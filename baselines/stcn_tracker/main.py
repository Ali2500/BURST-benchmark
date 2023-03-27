"""
Author: Ali Athar
Adapted from: https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/baselines/stp.py
"""
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from PIL import Image
from tqdm import tqdm
from time import time as current_time
from functools import partial
from datetime import timedelta, datetime

import cv2
import os
import os.path as osp
import multiprocessing as mp
import json
import pycocotools.mask as mt
import numpy as np
from multiprocessing.pool import Pool
from scipy.optimize import linear_sum_assignment
from utils import overlay_mask_on_image, create_color_map

config = {
    'DETECTION_THRESHOLD': 0.5,
    'ASSOCIATION_THRESHOLD': 1e-10,
    'MAX_FRAMES_SKIP': 4,
    'VOS_ASSOCIATION_NUM_FRAMES': 1,
    'IOU_MODE': 'mask'
}


# global variables for multiprocessing
TOTAL_COMPLETED = mp.Value('i', 0)
TOTAL_FRAMES_PROCESSED = mp.Value('i', 0)
TOTAL_SEQUENCES = None
START_TIME = None


def match(match_scores):
    match_rows, match_cols = linear_sum_assignment(-match_scores)
    return match_rows, match_cols


def parse_vos_result_dir(dirpath, iid_mapping):
    mask_files = sorted(glob(osp.join(dirpath, "*.png")))
    assert 0 < len(mask_files) <= (config["MAX_FRAMES_SKIP"] + 2), f"Num of mask files = {len(mask_files)}, dirpath: {dirpath}"

    iids = list(iid_mapping.keys())

    all_masks = []
    for t in range(len(mask_files)):
        masks_t = dict()

        mask = np.array(Image.open(mask_files[t]))
        for iid in iids:
            masks_t[iid_mapping[iid]] = mt.encode(np.asfortranarray((mask == iid).astype(np.uint8)))

        all_masks.append(masks_t)

    return all_masks


def compute_ious(current_timestep_det_idxes, current_timestep_rles, current_timestep_subclip_info, prev_masks,
                 prev_timesteps):
    assert len(prev_timesteps) == len(prev_masks)
    ious = np.zeros((len(prev_timesteps), len(current_timestep_rles)), np.float32)

    if len(current_timestep_rles) == 0 or len(prev_timesteps) == 0:
        return ious

    propagated_masks = parse_vos_result_dir(
        current_timestep_subclip_info["dirname"],
        current_timestep_subclip_info["iid_to_det_idx_mapping"]
    )

    for idx_1, (timestep_diff, prev_mask) in enumerate(zip(prev_timesteps.tolist(), prev_masks)):
        mask_t = propagated_masks[timestep_diff]

        for idx_2, (current_det_idx, rle_obj) in enumerate(zip(current_timestep_det_idxes, current_timestep_rles)):
            mask_t_det_idx = mask_t[current_det_idx]

            if config["IOU_MODE"] == "box":
                entity_1 = mt.toBbox(prev_mask)
                entity_2 = mt.toBbox(mask_t_det_idx)
            elif config["IOU_MODE"] == "mask":
                entity_1, entity_2 = prev_mask, mask_t_det_idx
            else:
                raise ValueError("Should not be here")

            ious[idx_1, idx_2] = mt.iou([entity_1], [entity_2], [False])

    return ious


def track_sequence(seq_obj, det_file_contents, seq_detections, subclip_info, mp_enabled):
    # To ensure IDs are unique per object across all classes.
    curr_max_id = 0
    seq_len = len(det_file_contents["image_paths"])

    output_segs = [dict() for _ in range(seq_len)]
    track_id_to_cls_id = dict()

    # Run tracker for each class.
    for cls, cls_data in tqdm(seq_detections.items(), total=len(seq_detections), leave=False, desc="Classes", disable=mp_enabled):
        # Initialize container for holding previously tracked objects.
        prev = {'boxes': np.empty((0, 4)),
                'masks': [],  # list of RLE objs
                'ids': np.array([], np.int32),
                'timesteps': np.array([])}

        # Run tracker for each frame.
        for timestep, t_data in enumerate(tqdm(cls_data, leave=False, desc="frame", disable=mp_enabled)):
            # Calculate IoU between previous and current frame dets.
            t_det_idxes = list(t_data.keys())
            t_mask_rles = [det["mask_rle"] for det in t_data.values()]
            t_subclip_info = subclip_info.get(timestep, None)

            if t_subclip_info is None and timestep != 0:
                keep_rows = [
                    i for i in range(len(prev['ids'])) if
                    (prev['timesteps'][i] + 1 <= config['MAX_FRAMES_SKIP'])
                ]

                # Update the set of previous tracking results to include the newly tracked detections.
                prev['ids'] = prev['ids'][keep_rows]
                prev["masks"] = [prev["masks"][i] for i in keep_rows]
                prev['timesteps'] = prev['timesteps'][keep_rows] + 1
                continue

            ious = compute_ious(
                current_timestep_det_idxes=t_det_idxes,
                current_timestep_rles=t_mask_rles,
                current_timestep_subclip_info=t_subclip_info,
                prev_timesteps=(prev["timesteps"] + 1).astype(int),
                prev_masks=prev['masks'],
            )

            # Find best matching between current dets and previous tracks.
            match_rows, match_cols = match(ious)

            # Remove matches that have an IoU below a certain threshold.
            actually_matched_mask = ious[match_rows, match_cols] > config['ASSOCIATION_THRESHOLD']
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            # Assign the prev track ID to the current dets if they were matched.
            ids = np.nan * np.ones((len(t_mask_rles),), np.int32)
            ids[match_cols] = prev['ids'][match_rows]

            # Create new track IDs for dets that were not matched to previous tracks.
            num_not_matched = len(ids) - len(match_cols)
            new_ids = np.arange(curr_max_id + 1, curr_max_id + num_not_matched + 1)
            ids[np.isnan(ids)] = new_ids

            for new_id in new_ids.tolist():
                track_id_to_cls_id[int(new_id)] = cls

            # Update maximum ID to ensure future added tracks have a unique ID value.
            curr_max_id += num_not_matched

            # Drop tracks from 'previous tracks' if they have not been matched in the last MAX_FRAMES_SKIP frames.
            unmatched_rows = [i for i in range(len(prev['ids'])) if
                              i not in match_rows and (prev['timesteps'][i] + 1 <= config['MAX_FRAMES_SKIP'])]

            # Update the set of previous tracking results to include the newly tracked detections.
            prev['ids'] = np.concatenate((ids, prev['ids'][unmatched_rows]), axis=0)
            prev["masks"] = t_mask_rles + [prev["masks"][i] for i in unmatched_rows]
            prev['timesteps'] = np.concatenate((np.zeros((len(ids),)), prev['timesteps'][unmatched_rows] + 1), axis=0)

            for i in range(len(t_mask_rles)):
                output_segs[timestep][int(ids[i])] = {
                    "rle": t_mask_rles[i]["counts"].decode("utf-8"),
                    "is_gt": False
                }

    seq_obj["segmentations"] = output_segs
    seq_obj["annotated_image_paths"] = det_file_contents["image_paths"]
    seq_obj["track_category_ids"] = track_id_to_cls_id

    return seq_obj


def process_seq(seq_obj, seq_subclip_info, detections_dir, class_id_mapping, mp_enabled):
    seq_key = f"{seq_obj['dataset']}_{seq_obj['seq_name']}"

    # parse detections
    detections_file = osp.join(detections_dir, seq_key + ".json")
    assert osp.exists(detections_file), f"Detections file not found: {detections_file}"

    with open(detections_file, 'r') as fh:
        detections_content = json.load(fh)

    seq_len = len(detections_content["image_paths"])
    img_dims = int(seq_obj["height"]), int(seq_obj["width"])

    detections_parsed = defaultdict(lambda: [dict() for _ in range(seq_len)])

    for t, preds_t in enumerate(detections_content["predictions"]):
        for det_idx, (mask_rle, box, score, cls_id) in enumerate(zip(
                preds_t["masks"], preds_t["boxes"], preds_t["scores"], preds_t["classes"]
        )):
            if score < config["DETECTION_THRESHOLD"]:
                continue

            if cls_id not in class_id_mapping:
                continue

            cls_id = class_id_mapping[cls_id]

            detections_parsed[cls_id][t][det_idx] = {
                "mask_rle": {"counts": mask_rle.encode("utf-8"), "size": img_dims},
                "box": box,
                "score": score
            }

    detections_parsed = dict(detections_parsed)

    try:
        seq_obj = track_sequence(seq_obj, detections_content, detections_parsed, seq_subclip_info, mp_enabled)
    except Exception as _:
        print(f"Error occurred while processing {seq_obj['dataset']}/{seq_obj['seq_name']}")
        seq_obj["segmentations"] = [dict() for _ in range(len(seq_obj["segmentations"]))]
        seq_obj["track_category_ids"] = dict()

    return seq_obj


def vis_sequence_tracks(seq_obj, split):
    suffix = split
    if split == "validation":
        suffix = "val"
    images_base_dir = f"/globalwork/data/TAO/frames/{suffix}"

    output_segs = seq_obj["segmentations"]
    assert len(output_segs) == len(seq_obj["annotated_image_paths"])
    cmap = create_color_map().tolist()
    img_dims = int(seq_obj["height"]), int(seq_obj["width"])
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for img_fname, segs_t in zip(seq_obj["annotated_image_paths"], output_segs):
        image_path = osp.join(images_base_dir, seq_obj['dataset'], seq_obj['seq_name'], img_fname)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        assert image is not None, f"Image not found: {image_path}"

        for track_id, seg in segs_t.items():
            mask = mt.decode({
                "counts": seg["rle"].encode("utf-8"),
                "size": img_dims
            }).astype(np.uint8)
            image = overlay_mask_on_image(image, mask, 0.5, tuple(cmap[track_id % 256]))

        cv2.imshow("Image", image)
        if cv2.waitKey(0) == 113:
            return


def get_detectron2_to_burst_category_id_mapping():
    coco_id_mapping_file = osp.join(osp.dirname(__file__), "class_mappings", "burst_coco_class_mapping.json")
    coco_id_mapping_file = osp.realpath(coco_id_mapping_file)
    assert osp.exists(coco_id_mapping_file), f"File not found: {coco_id_mapping_file}"

    with open(coco_id_mapping_file, 'r') as fh:
        category_info = json.load(fh)

    return {
        x['coco_id'] - 1: x['id'] for x in category_info
    }


def get_detectron2_longtail_category_id_mapping():
    lvis_cats_file = osp.join(osp.dirname(__file__), "class_mappings", "lvis_classes.json")
    lvis_cats_file = osp.realpath(lvis_cats_file)
    assert osp.exists(lvis_cats_file), f"File not found: {lvis_cats_file}"

    with open(lvis_cats_file, 'r') as fh:
        category_info = json.load(fh)

    return {
        x['id'] - 1: x['id'] for x in category_info
    }


def worker_fn(mp_enabled, subclip_info_dict, detections_dir, seq_to_process, class_id_mapping, seq):
    global TOTAL_COMPLETED
    if seq_to_process is not None and f"{seq['dataset']}/{seq['seq_name']}" != seq_to_process:
        return

    seq_key = f"{seq['dataset']}_{seq['seq_name']}"
    try:
        updated_seq_obj = process_seq(seq, subclip_info_dict[seq_key], detections_dir, class_id_mapping, mp_enabled)

    except Exception as err:
        print(f"Failed to process sequence: {seq['dataset']}/{seq['seq_name']}.")
        raise err

    if mp_enabled:
        TOTAL_COMPLETED.value += 1
        TOTAL_FRAMES_PROCESSED.value += len(seq['annotated_image_paths'])
        time_per_seq = (current_time() - START_TIME) / float(TOTAL_COMPLETED.value)
        eta = timedelta(seconds=(TOTAL_SEQUENCES - TOTAL_COMPLETED.value) * time_per_seq)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed {TOTAL_COMPLETED.value}/{TOTAL_SEQUENCES} sequences. ETA: {str(eta)}")

    return updated_seq_obj


def main(args):
    vos_subclip_output_dir = osp.join(args.base_dir, "STCN_output")
    subclip_info_file = osp.join(args.base_dir, "info.json")
    detections_dir = osp.join(args.base_dir, "detections")

    assert osp.exists(vos_subclip_output_dir)
    assert osp.exists(subclip_info_file)
    assert osp.exists(detections_dir)
    assert args.detector_type

    with open(subclip_info_file, 'r') as fh:
        subclip_info_content = json.load(fh)

    global config, START_TIME, TOTAL_SEQUENCES
    config["IOU_MODE"] = args.iou_mode

    # organize into dict
    subclip_info_dict = defaultdict(dict)

    for subclip in subclip_info_content:
        seq_key = f"{subclip['dataset']}_{subclip['seq_name']}"
        start_t = subclip["frame_idxes"][0]
        end_t = subclip['frame_idxes'][1]

        dirname = subclip["dirname"]

        assert start_t > end_t
        subclip_info_dict[seq_key][start_t] = {
            "dirname": osp.join(vos_subclip_output_dir, dirname),
            "iid_to_det_idx_mapping": {int(k): v for k, v in subclip["iid_to_det_idx_mapping"].items()}
        }

    with open(args.gt_anns_file, 'r') as fh:
        content = json.load(fh)

    if args.detector_type == "coco":
        class_id_mapping = get_detectron2_to_burst_category_id_mapping()

    elif args.detector_type == "lvis":
        class_id_mapping = get_detectron2_longtail_category_id_mapping()

    else:
        raise ValueError("Should not be here")

    input_sequences = content["sequences"]
    TOTAL_SEQUENCES = len(input_sequences)

    print(f"Output path: {args.output_path}")

    if args.num_procs <= 0 or args.seq:
        _worker_fn = partial(worker_fn, False, subclip_info_dict, detections_dir, args.seq, class_id_mapping)
        updated_sequences = []

        pbar = tqdm(input_sequences, leave=False, total=len(input_sequences))
        for seq in pbar:
            pbar.set_description(f"{seq['dataset']}/{seq['seq_name']}")
            updated_sequences.append(_worker_fn(seq))

    else:
        print(f"Running on {args.num_procs} parallel processes.")
        _worker_fn = partial(worker_fn, True, subclip_info_dict, detections_dir, args.seq, class_id_mapping)
        START_TIME = current_time()
        with Pool(args.num_procs) as p:
            updated_sequences = list(p.map(_worker_fn, input_sequences))

    content["sequences"] = updated_sequences
    os.makedirs(osp.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w') as fh:
        json.dump(content, fh)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "--base_dir", '-i', required=True, help="Path to directory containing per-frame detections "
        "and STCN mask propagations"
    )
    parser.add_argument(
        "--detector_type", '-dt', choices=("coco", "lvis"), help="Use 'coco' for common-class and "
        "open-world tasks and 'lvis' for longtail"
    )
    parser.add_argument(
        "--iou_mode", default='mask', choices=('mask', 'box'), help="Whether to use box or mask IoU when"
        "computing affinity scores between object proposals and mask propagation"
    )
    parser.add_argument(
        "--gt_anns_file", '-gt', required=True, help="Path to ground-truth annotations file"
    )
    parser.add_argument(
        "--output_path", '-o', required=True, help="Path to location where the predicted track output will be saved"
    )
    parser.add_argument(
        "--num_procs", type=int, default=mp.cpu_count(), help="Number of parallel processes to use. "
        "Default value is the number of CPU cores on the system"
    )
    parser.add_argument(
        "--seq", required=False, help="If given, only the specified sequence will be evaluated. "
        "E.g. to only run the tracker for seq 'xyz' in the 'HACS' dataset, set this argument to"
        " 'HACS_xyz'"
    )

    main(parser.parse_args())

from typing import Dict, List, Any, Tuple

import numpy as np
import pycocotools.mask as cocomask


def intify_track_ids(video_dict: Dict[str, Any]):
    video_dict["categories"] = {
        int(track_id): category_id for track_id, category_id in video_dict["categories"].items()
    }

    for t in range(len(video_dict["segmentations"])):
        video_dict["segmentations"][t] = {
            int(track_id): seg
            for track_id, seg in video_dict["segmentations"][t].items()
        }

    return video_dict


def rle_ann_to_mask(rle: str, image_size: Tuple[int, int]) -> np.ndarray:
    return cocomask.decode({
        "size": image_size,
        "counts": rle.encode("utf-8")
    }).astype(bool)

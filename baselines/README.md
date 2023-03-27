# Class-guided Baselines

## STCN Tracker

Download the required data. This includes:

* Per-frame object proposals generated using an off-the-shelf instance segmentation model. The detections are post-processed to be non-overlapping.
* Mask propagation for each of the per-frame object proposals into neighboring frames using an off-the-shelf STCN model trained for DAVIS/YouTube-VOS.

|                   | val | test |
|-------------------|-----|------|
| longtail          | [Link](https://omnomnom.vision.rwth-aachen.de/data/BURST_baselines/val/STCN_tracker/longtail.zip) | [Link](https://omnomnom.vision.rwth-aachen.de/data/BURST_baselines/test/STCN_tracker/longtail.zip)  |
| common/open_world | [Link](https://omnomnom.vision.rwth-aachen.de/data/BURST_baselines/val/STCN_tracker/common_open_world.zip) | [Link](https://omnomnom.vision.rwth-aachen.de/data/BURST_baselines/test/STCN_tracker/common_open_world.zip)  |

For longtail, the object proposals are obtained from a MaskRCNN model (X-101-FPN backbone) trained on LVIS.
For common/open_world, the object proposals are obtained from a Mask2Former model  (Swin-L backbone) trained on COCO.

To run the baseline, extract any of zip files linked above, and then run:

```
cd baselines/stcn_tracker
python main.py -i /path/to/extracted/zip/directory --detector_type {lvis,coco} --gt_anns_file <dataset_root>/annotations/{val,test}/all_classes.json -o /path/to/output.json
```

Set `--detector_type` to `lvis` for longtail and `coco` for common/open_world. For further argument details, run the script with `--help`.

## Box Tracker

Coming soon
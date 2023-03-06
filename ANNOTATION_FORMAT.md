# Annotation Format

The object masks are provided in JSON files as RLE encoded strings. We use `pycocotools` to encode/decode these masks.

The JSON file contains a dictionary which is organized as follows:

```
sequences:
    - width: <int>  # image_dims
      height: <int>
      id: <int>
      seq_name: <str>
      dataset: <str>  # LaSOT, BDD, ArgoVerse, HACS, ...
      fps: <int>
      all_image_paths:
        - <str>
        ...
      annotated_image_paths:
        - <str>
        ...
      neg_category_ids: 
        - <int>
        ...
      not_exhaustive_category_ids:
        - <int>
        ...
      track_category_ids:
        ...  # see below for details
      segmentations:
        ...  # see below for details
categories:
    - id: <int>
      name: <str>
      synset: <str>
      def: <str>
      synonyms: 
        - <str>
        ...
    ...
    ...
split: <str>  # train/val/test
```

- `categories`: List of all object categories in the dataset. We use the same category IDs as the LVIS dataset.
- `sequences`: List of all video sequences in the datasets. Each list entry is a dictionary with basic attributes e.g. image size, video ID, etc, and also the mask annotations for the object tracks in this video.
- `split`: which split (train/val/test) the annotations belong to.

#### Mask Annotations and Category IDs Per Sequence

The `track_category_ids` is a dict which conveys the category ID for each object track in that sequence:

```
track_category_ids:
    track_id: category_id
    ...
``` 

The `segmentations` field is a list with one entry per annotated video frame. Each list element is a dict with track IDs as keys and encoded masks and other attributes as values

```
segmentations:
    - rle: <str>
      is_gt: <bool>
      score: <float>
      bbox:               # only present in `first_frame_annotations` file
        - x coord <int>
        - y coord <int>
        - width <int>
        - height <int>
      point:              # only present in `first_frame_annotations` file
        - x coord <int>
        - y coord <int>
``` 

For the training set, we adopted a semi-automated workflow for annotating temporally dense object masks. The `is_gt` field conveys whether the given mask was annotated automatically or by a human annotator. The `score` field conveys the confidence for an automatically annotated mask.

- For the val and test sets, all annotations were done by humans, so the `is_gt` and `score` fields can be ignored. 

- For the `first_frame_annotations` files (useful for exemplar-guided tasks), there are two additional fields `bbox` and `point` which convey the bounding box and a random point on the object for the first frame in which it occurs.

- In the `segmentations` and `track_category_ids` fields, track IDs are encoded as strings (the JSON file format enforces that dict keys must be strings). Remember to cast them as int when parsing the annotations.


#### Format for Evaluation Code

For evaluating your predicted results, the code expects a single JSON file with the same format as the ground-truth format explained above. When generating this file, we recommnd simply loading the ground-truth file (available for both val and test sets) and replacing the `track_category_ids` and `segmentations` fields for every video sequence with your predicted results. Note that predictions for `track_category_ids` are only needed for the common and long-tail class-guided tasks. For the exemplar-guided and open-world tasks, `track_category_ids` is irrelevant and can safely be set to any value without effecting the score.

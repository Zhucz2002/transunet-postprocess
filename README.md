# transunet-postprocess

Post-processing utilities for medical image segmentation results from TransUNet.

## Features

- Read 3D hard-label `.nii` / `.nii.gz` segmentations
- Extract a single organ label from multi-class outputs
- Slice along the z-axis and export:
  - Original slices
  - Binary masks
  - Overlays with contours drawn on each slice

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python process_segmentation.py \
  --input /path/to/segmentation.nii.gz \
  --label 1 \
  --output outputs
```

The script will create the following structure under the output directory:

```
outputs/
  slices/     # grayscale slice images
  masks/      # binary masks for the selected label
  overlays/   # contour overlays on each slice
```

## Notes

- The overlay draws the organ contour in red on the corresponding slice.
- Input segmentation must be a 3D volume (z-axis slicing).

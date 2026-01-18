#!/usr/bin/env python3
"""Post-process 3D segmentation results and generate slice visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
from skimage import measure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an organ label from a 3D hard-label segmentation, "
            "slice along z, and export slices/masks/overlays."
        )
    )
    parser.add_argument("--image", required=True, help="Path to input nii/nii.gz image volume")
    parser.add_argument("--seg", required=True, help="Path to input nii/nii.gz segmentation")
    parser.add_argument("--label", type=int, required=True, help="Organ label to extract")
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for slices, masks, overlays",
    )
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2, help="Axis to slice along")
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for intensity clipping")
    parser.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for intensity clipping")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(slice_data: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    if slice_data.size == 0:
        return slice_data.astype(np.uint8)
    if pmin >= pmax:
        raise ValueError("pmin must be less than pmax")
    lower, upper = np.percentile(slice_data, [pmin, pmax])
    if upper <= lower:
        return np.zeros_like(slice_data, dtype=np.uint8)
    clipped = np.clip(slice_data, lower, upper)
    normalized = (clipped - lower) / (upper - lower) * 255.0
    return normalized.astype(np.uint8)


def overlay_contours(base_gray: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    if base_gray.ndim != 2:
        raise ValueError("base_gray must be a 2D array")
    rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
    for contour in contours:
        for point in contour:
            row, col = int(round(point[0])), int(round(point[1]))
            if 0 <= row < rgb.shape[0] and 0 <= col < rgb.shape[1]:
                rgb[row, col] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    seg_path = Path(args.seg)
    output_dir = Path(args.output)

    image_nii = nib.load(str(image_path))
    image_data = np.asarray(image_nii.get_fdata(), dtype=np.float32)

    seg_nii = nib.load(str(seg_path))
    seg_data = np.asarray(seg_nii.get_fdata(), dtype=np.int32)

    if image_data.ndim != 3:
        raise ValueError(f"Expected a 3D image volume, got shape {image_data.shape}")
    if seg_data.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation volume, got shape {seg_data.shape}")
    if image_data.shape != seg_data.shape:
        raise ValueError(
            "Image and segmentation shapes must match, "
            f"got {image_data.shape} and {seg_data.shape}"
        )

    organ_mask = seg_data == args.label

    slices_dir = output_dir / "slices"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    ensure_dir(slices_dir)
    ensure_dir(masks_dir)
    ensure_dir(overlays_dir)

    depth = image_data.shape[args.axis]
    for index in range(depth):
        slice_data = np.take(image_data, index, axis=args.axis)
        mask_slice = np.take(organ_mask, index, axis=args.axis)

        slice_uint8 = normalize_to_uint8(slice_data, args.pmin, args.pmax)
        mask_uint8 = mask_slice.astype(np.uint8) * 255

        contours = measure.find_contours(mask_slice.astype(np.uint8), 0.5)
        overlay = overlay_contours(slice_uint8, contours)

        slice_path = slices_dir / f"slice_{index:04d}.png"
        mask_path = masks_dir / f"mask_{index:04d}.png"
        overlay_path = overlays_dir / f"overlay_{index:04d}.png"

        imageio.imwrite(slice_path, slice_uint8)
        imageio.imwrite(mask_path, mask_uint8)
        imageio.imwrite(overlay_path, overlay)

    print(f"Saved {depth} slices to {output_dir}")


if __name__ == "__main__":
    main()

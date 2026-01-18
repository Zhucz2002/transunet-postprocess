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
    parser.add_argument("--input", required=True, help="Path to input nii/nii.gz segmentation")
    parser.add_argument("--label", type=int, required=True, help="Organ label to extract")
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for slices, masks, overlays",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(slice_data: np.ndarray) -> np.ndarray:
    if slice_data.dtype == np.uint8:
        return slice_data
    max_value = slice_data.max() if slice_data.size > 0 else 0
    if max_value == 0:
        return slice_data.astype(np.uint8)
    normalized = (slice_data.astype(np.float32) / max_value) * 255.0
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
    input_path = Path(args.input)
    output_dir = Path(args.output)

    nii = nib.load(str(input_path))
    data = np.asarray(nii.get_fdata(), dtype=np.int32)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation volume, got shape {data.shape}")

    organ_mask = data == args.label

    slices_dir = output_dir / "slices"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    ensure_dir(slices_dir)
    ensure_dir(masks_dir)
    ensure_dir(overlays_dir)

    depth = data.shape[2]
    for z in range(depth):
        slice_data = data[:, :, z]
        mask_slice = organ_mask[:, :, z]

        slice_uint8 = normalize_to_uint8(slice_data)
        mask_uint8 = (mask_slice.astype(np.uint8) * 255)

        contours = measure.find_contours(mask_slice.astype(np.uint8), 0.5)
        overlay = overlay_contours(slice_uint8, contours)

        slice_path = slices_dir / f"slice_{z:04d}.png"
        mask_path = masks_dir / f"mask_{z:04d}.png"
        overlay_path = overlays_dir / f"overlay_{z:04d}.png"

        imageio.imwrite(slice_path, slice_uint8)
        imageio.imwrite(mask_path, mask_uint8)
        imageio.imwrite(overlay_path, overlay)

    print(f"Saved {depth} slices to {output_dir}")


if __name__ == "__main__":
    main()

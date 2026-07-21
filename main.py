"""Unified command-line pipeline for D2S-PACT and comparison methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from artifact_removal import (
    remove_with_a2a,
    remove_with_dip,
    remove_with_proposed,
    remove_with_svd,
)
from reconstruction import (
    DETECTOR_STRIDE,
    load_measurements,
    reconstruct_dual_domain,
    reconstruct_ubp,
    sample_independent_measurements,
)
from visualization import save_comparison_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("proposed", "a2a", "dip", "svd"),
        help="Artifact-removal method to run.",
    )
    parser.add_argument("--signals", default="data/sensor_signal.npz")
    parser.add_argument("--locations", default="data/sensor_location.txt")
    parser.add_argument(
        "--output",
        help="Output .npy path. Defaults to output/<method>.npy.",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--figure-dir", default="figure")
    args = parser.parse_args()

    full_signals, full_locations = load_measurements(
        args.signals,
        args.locations,
        detector_stride=1,
    )
    sparse_signals = full_signals[::DETECTOR_STRIDE]
    sparse_locations = full_locations[::DETECTOR_STRIDE]
    print(
        f"Sparse signals: {sparse_signals.shape}; "
        f"detector locations: {sparse_locations.shape}"
    )

    fdm_volume, ubp_volume, _ = reconstruct_dual_domain(
        sparse_signals,
        sparse_locations,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.method == "proposed":
        result = remove_with_proposed(ubp_volume, fdm_volume, device)
    elif args.method == "a2a":
        first_signals, first_locations, second_signals, second_locations = (
            sample_independent_measurements(full_signals, full_locations)
        )
        print("Reconstructing the first A2A input")
        first_volume = reconstruct_ubp(first_signals, first_locations)
        print("Reconstructing the second A2A input")
        second_volume = reconstruct_ubp(second_signals, second_locations)
        result = remove_with_a2a(first_volume, second_volume, device)
    elif args.method == "dip":
        result = remove_with_dip(ubp_volume, device)
    else:
        result = remove_with_svd(ubp_volume, fdm_volume)

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.output_dir) / f"{args.method}.npy"
    )
    if output_path.suffix.lower() != ".npy":
        output_path = output_path.with_suffix(".npy")
    figure_path = Path(args.figure_dir) / f"{args.method}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.astype(np.float32))
    save_comparison_figure(
        fdm_volume,
        ubp_volume,
        result,
        args.method,
        figure_path,
    )
    print(f"Saved volume to {output_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()

"""D2S-PACT and comparison methods for the public hemispherical simulation.

Examples: python main.py --method proposed --lambda-cross 0.25
          python main.py --method n2i
          python main.py --method mbpd --mbpd-table-dir MB-PD/data --no-figure
          python main.py --method pnp --mbpd-table-dir MB-PD/data --no-figure
Frangi requires scikit-image; PnP requires bm4d (install separately with pip).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

from artifact_removal import (
    A2A_SAMPLE_SIZE,
    LAMBDA_CROSS,
    MBPD_DAMP,
    MBPD_VOXEL_CHUNK,
    MBPDOperator,
    N2I_SPLITS,
    RANDOM_SEED,
    SVD_AXIS,
    SVD_RANK,
    load_mbpd_tables,
    remove_with_a2a,
    remove_with_dip,
    remove_with_frangi,
    remove_with_mbpd,
    remove_with_n2i,
    remove_with_pnp,
    remove_with_proposed,
    remove_with_svd,
)
from reconstruction import normalize_volume
from visualization import save_comparison_figure


def prepare_n2i_splits(signals, locations, reconstructor=None):
    """Use a common full-array denominator for four signed, unnormalized splits.

    A custom reconstructor must also return signed values without normalization.
    """
    if len(signals) < N2I_SPLITS or len(signals) % N2I_SPLITS:
        raise ValueError("Noise2Inverse needs a detector count divisible by four.")
    if reconstructor is None:
        from reconstruction import reconstruct_ubp_raw

        reconstructor = reconstruct_ubp_raw

    full_volume = reconstructor(signals, locations)
    rng = np.random.default_rng(RANDOM_SEED)
    groups = np.array_split(rng.permutation(len(signals)), N2I_SPLITS)
    splits = []
    for index, group in enumerate(groups):
        masked = np.zeros_like(signals)
        masked[group] = signals[group]
        splits.append(reconstructor(masked, locations) * N2I_SPLITS)
        print(f"Noise2Inverse split {index + 1}/{N2I_SPLITS} reconstructed")
    mean = sum(volume / N2I_SPLITS for volume in splits)
    relative_error = np.linalg.norm((mean - full_volume).astype(np.float64).ravel()) / max(
        np.linalg.norm(full_volume.astype(np.float64).ravel()), 1e-30
    )
    if not np.isfinite(relative_error) or relative_error > 5e-5:
        raise RuntimeError(f"Noise2Inverse split-mean linearity check failed: {relative_error}")
    return splits, full_volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("proposed", "a2a", "dip", "svd", "frangi", "mbpd", "pnp", "n2i"),
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
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip the comparison figure and its additional reconstructions.",
    )
    parser.add_argument(
        "--lambda-cross",
        "--lamdacross",
        dest="lambda_cross",
        type=float,
        default=LAMBDA_CROSS,
        help="Cross-loss and inference-mixing weight in [0, 1] (default: 0.25).",
    )
    parser.add_argument("--svd-rank", type=int, default=SVD_RANK)
    parser.add_argument("--svd-axis", type=int, choices=(0, 1, 2), default=SVD_AXIS)
    parser.add_argument(
        "--mbpd-table-dir",
        default="MB-PD/data",
        help="Directory containing the two GPAIR sphere_integral*.mat tables.",
    )
    parser.add_argument(
        "--mbpd-damp",
        type=float,
        default=MBPD_DAMP,
        help="Simulation LSQR damping; the L2 coefficient is damp squared (default: 500).",
    )
    args = parser.parse_args()
    if not 0 <= args.lambda_cross <= 1:
        parser.error("--lambda-cross / --lamdacross must lie in [0, 1].")
    if not np.isfinite(args.mbpd_damp) or args.mbpd_damp < 0:
        parser.error("--mbpd-damp must be finite and nonnegative.")
    if args.svd_rank < 1:
        parser.error("--svd-rank must be positive.")

    optional = {"frangi": ("skimage.feature", "scikit-image"), "pnp": ("bm4d", "bm4d")}
    if args.method in optional:
        module, package = optional[args.method]
        try:
            importlib.import_module(module)
        except ImportError as error:
            parser.error(f"{args.method} requires {package}: pip install {package}. {error}")
    tables = None
    if args.method in ("mbpd", "pnp"):
        tables = load_mbpd_tables(args.mbpd_table_dir)

    from reconstruction import (
        DETECTOR_STRIDE,
        GRID_SIZE,
        X_RANGE,
        Y_RANGE,
        Z_RANGE,
        VOXEL_SIZE,
        SOUND_SPEED,
        SAMPLING_RATE,
        load_measurements,
        reconstruct_dual_domain,
        reconstruct_ubp,
        reconstruct_ubp_raw,
    )

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    full_signals, full_locations = load_measurements(
        args.signals,
        args.locations,
        detector_stride=1,
    )
    sparse_signals = full_signals[::DETECTOR_STRIDE]
    sparse_locations = full_locations[::DETECTOR_STRIDE]
    print(
        f"Sparse signals: {sparse_signals.shape}; " f"detector locations: {sparse_locations.shape}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fdm_volume = ubp_volume = None
    if args.method in ("proposed", "svd"):
        fdm_volume, ubp_volume, _ = reconstruct_dual_domain(sparse_signals, sparse_locations)
    elif args.method == "a2a":
        ubp_volume = reconstruct_ubp_raw(sparse_signals, sparse_locations)
    elif args.method in ("dip", "frangi"):
        ubp_volume = reconstruct_ubp(sparse_signals, sparse_locations)

    if args.method == "proposed":
        result = remove_with_proposed(ubp_volume, fdm_volume, device, args.lambda_cross)
    elif args.method == "a2a":
        if len(sparse_signals) < A2A_SAMPLE_SIZE:
            parser.error("A2A requires at least 400 selected sparse detectors.")
        rng = np.random.default_rng(RANDOM_SEED)
        first_indices, second_indices = [
            np.sort(rng.choice(len(sparse_signals), A2A_SAMPLE_SIZE, replace=False))
            for _ in range(2)
        ]
        print("Reconstructing the first A2A input")
        first_volume = reconstruct_ubp_raw(
            sparse_signals[first_indices], sparse_locations[first_indices]
        )
        print("Reconstructing the second A2A input")
        second_volume = reconstruct_ubp_raw(
            sparse_signals[second_indices], sparse_locations[second_indices]
        )
        result = remove_with_a2a(first_volume, second_volume, device, ubp_volume)
    elif args.method == "dip":
        result = remove_with_dip(ubp_volume, device)
    elif args.method == "svd":
        result = remove_with_svd(ubp_volume, fdm_volume, args.svd_rank, args.svd_axis)
    elif args.method == "frangi":
        result = remove_with_frangi(ubp_volume)
    elif args.method == "n2i":
        splits, ubp_volume = prepare_n2i_splits(sparse_signals, sparse_locations)
        result = remove_with_n2i(splits, device)
    else:
        operator = MBPDOperator(
            sparse_locations,
            GRID_SIZE,
            (X_RANGE, Y_RANGE, Z_RANGE),
            VOXEL_SIZE,
            SOUND_SPEED,
            SAMPLING_RATE,
            sparse_signals.shape[1],
            *tables,
            device=device,
            voxel_chunk=MBPD_VOXEL_CHUNK,
        )
        if args.method == "mbpd":
            result = remove_with_mbpd(operator, sparse_signals, damp=args.mbpd_damp)
        else:
            result = remove_with_pnp(operator, sparse_signals)
        del operator

    output_path = Path(args.output) if args.output else Path(args.output_dir) / f"{args.method}.npy"
    if output_path.suffix.lower() != ".npy":
        output_path = output_path.with_suffix(".npy")
    figure_path = Path(args.figure_dir) / f"{args.method}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Every method uses one 3D normalization before saving or projecting.
    result = normalize_volume(result)
    np.save(output_path, result)
    print(f"Saved volume to {output_path}")
    if not args.no_figure:
        # These analytical reconstructions are only for the comparison figure;
        # MB-PD and PnP consume measurements directly.
        if fdm_volume is None:
            from reconstruction import reconstruct_fdm

            fdm_volume, _ = reconstruct_fdm(sparse_signals, sparse_locations)
        if ubp_volume is None:
            ubp_volume = reconstruct_ubp(sparse_signals, sparse_locations)
        save_comparison_figure(
            fdm_volume,
            normalize_volume(ubp_volume),
            result,
            args.method,
            figure_path,
        )
        print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()

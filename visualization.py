"""MAP visualization for PACT reconstruction comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def maximum_amplitude_projection(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return the transposed maximum amplitude projection used by the notebooks."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}.")
    return np.max(volume, axis=axis).T


def save_comparison_figure(
    fdm_volume: np.ndarray,
    ubp_volume: np.ndarray,
    result_volume: np.ndarray,
    method: str,
    output_path: str | Path,
) -> None:
    """Save the original three-panel Z-axis MAP comparison style."""
    projections = [
        maximum_amplitude_projection(fdm_volume),
        maximum_amplitude_projection(ubp_volume),
        maximum_amplitude_projection(result_volume),
    ]
    titles = ["image3d_fft", "image3d_ubp", method]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, projection, title in zip(axes, projections, titles):
        axis.imshow(projection, cmap="hot", origin="lower", aspect="equal")
        axis.set_title(f"MAP ({title})")
        axis.invert_yaxis()
        axis.axis("off")

    figure.subplots_adjust(wspace=0.05, hspace=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

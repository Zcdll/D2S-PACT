"""Artifact-removal methods and defaults for the public 3D PACT simulation."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from torch.cuda.amp import GradScaler, autocast

from reconstruction import normalize_volume


MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.6
PATCH_SIZE = (64, 64, 64)
BATCH_SIZE = 2
ITERATIONS_PER_EPOCH = 20
LAMBDA_CROSS = 0.25  # Used in both the cross-loss and inference mixture.
LAMBDA_CONSISTENCY = 10.0
UNCERTAINTY_ALPHA = 12.0
EMBEDDING_CHANNELS = 32
INFERENCE_TILE_SIZE = 96
INFERENCE_HALO = 3
RANDOM_SEED = 2026

A2A_EPOCHS = 3000
A2A_LEARNING_RATE = 1e-3
A2A_CHANNELS = 48
A2A_SAMPLE_SIZE = 400
A2A_SCHEDULER_STEP_SIZE = 1000

DIP_ITERATIONS = 2500
DIP_LEARNING_RATE = 1e-3
DIP_BASE_CHANNELS = 8

SVD_RANK = 19
SVD_AXIS = 1
N2I_SPLITS = 4
N2I_EFFECTIVE_EPOCHS = 100
N2I_BATCH_SIZE = 8
N2I_DATA_SCALING = 1 / 4096.0
MBPD_DAMP = 500.0  # Simulation setting in Table II; LSQR uses damp squared.
MBPD_MAX_ITERATIONS = 100
MBPD_TOLERANCE = 1e-5
MBPD_VOXEL_CHUNK = 1048576
MBPD_TABLE_SOURCE = (
    "https://github.com/ddffwyb/GPAIR/tree/" "c05dc94c206bdc0796cb51440d47fe8e6e80940d/data"
)
MBPD_TABLE_HASHES = {
    "sphere_integral_table.mat": "f1c66855e7c47a1d65cf514d33dd6248072298f0782fbf63468e9b9799655fa0",
    "sphere_integral_gradd_table.mat": "15b1c3649550d343919f6c3a4c6a1ca3ad9c84c547990c4574788ee06c51f1ab",
}


class ArtifactNet3D(nn.Module):
    """Lightweight shared-weight residual network from the proposed method."""

    def __init__(
        self,
        input_channels: int = 1,
        embedding_channels: int = EMBEDDING_CHANNELS,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.convolution_1 = nn.Conv3d(input_channels, embedding_channels, 3, padding=1)
        self.convolution_2 = nn.Conv3d(embedding_channels, embedding_channels, 3, padding=1)
        self.convolution_3 = nn.Conv3d(embedding_channels, embedding_channels, 3, padding=1)
        self.output = nn.Conv3d(embedding_channels, input_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.convolution_1(inputs))
        features = self.activation(self.convolution_2(features))
        features = self.activation(self.convolution_3(features))
        return self.output(features)


class A2ANet2D(nn.Module):
    """Three-layer residual predictor for slice-wise ZS-A2A."""

    def __init__(self, channels: int = A2A_CHANNELS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConvBlock3D(nn.Module):
    """Convolution, batch normalization and activation used by the DIP U-Net."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DIPUNet3D(nn.Module):
    """Compact 3D U-Net used as the deep image prior comparison."""

    def __init__(self, base_channels: int = DIP_BASE_CHANNELS):
        super().__init__()
        self.encoder_1 = ConvBlock3D(1, base_channels)
        self.encoder_2 = ConvBlock3D(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool3d(2)
        self.decoder_2 = ConvBlock3D(base_channels * 6, base_channels * 2)
        self.decoder_1 = ConvBlock3D(base_channels * 3, base_channels)
        self.output = nn.Sequential(nn.Conv3d(base_channels, 1, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_1 = self.encoder_1(inputs)
        encoded_2 = self.encoder_2(self.pool(encoded_1))
        bottleneck = self.bottleneck(self.pool(encoded_2))
        up_2 = functional.interpolate(
            bottleneck, size=encoded_2.shape[2:], mode="trilinear", align_corners=False
        )
        decoded_2 = self.decoder_2(torch.cat([up_2, encoded_2], dim=1))
        up_1 = functional.interpolate(
            decoded_2, size=encoded_1.shape[2:], mode="trilinear", align_corners=False
        )
        decoded_1 = self.decoder_1(torch.cat([up_1, encoded_1], dim=1))
        return self.output(decoded_1)


def dual_domain_loss(
    ubp: torch.Tensor,
    fdm: torch.Tensor,
    model: nn.Module,
    lambda_cross: float = LAMBDA_CROSS,
    lambda_consistency: float = LAMBDA_CONSISTENCY,
    alpha: float = UNCERTAINTY_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the proposed cross-domain and uncertainty-weighted losses."""
    if not 0 <= lambda_cross <= 1:
        raise ValueError("lambda_cross must lie in [0, 1].")
    ubp_artifact = model(ubp)
    fdm_artifact = model(fdm)
    clean_ubp = ubp - ubp_artifact
    clean_fdm = fdm - fdm_artifact

    cross_loss = lambda_cross * functional.mse_loss(clean_ubp, fdm)
    cross_loss += (1 - lambda_cross) * functional.mse_loss(clean_fdm, ubp)
    uncertainty = torch.exp(-alpha * torch.abs(ubp - fdm))
    consistency_loss = (uncertainty * (clean_ubp - clean_fdm) ** 2).mean()
    total_loss = cross_loss + lambda_consistency * consistency_loss
    return total_loss, clean_ubp


def random_paired_crops(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample aligned patches and apply the original random x-axis flip."""
    _, _, depth, height, width = volume_a.shape
    patch_depth, patch_height, patch_width = patch_size
    crops_a: list[torch.Tensor] = []
    crops_b: list[torch.Tensor] = []

    for _ in range(batch_size):
        z_start = np.random.randint(0, depth - patch_depth + 1)
        y_start = np.random.randint(0, height - patch_height + 1)
        x_start = np.random.randint(0, width - patch_width + 1)
        crop_a = volume_a[
            :,
            :,
            z_start : z_start + patch_depth,
            y_start : y_start + patch_height,
            x_start : x_start + patch_width,
        ]
        crop_b = volume_b[
            :,
            :,
            z_start : z_start + patch_depth,
            y_start : y_start + patch_height,
            x_start : x_start + patch_width,
        ]
        if np.random.rand() > 0.5:
            crop_a = torch.flip(crop_a, [4])
            crop_b = torch.flip(crop_b, [4])
        crops_a.append(crop_a)
        crops_b.append(crop_b)

    return torch.cat(crops_a, dim=0), torch.cat(crops_b, dim=0)


def prepare_proposed_volumes(
    ubp_volume: np.ndarray,
    fdm_volume: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the FDM and UBP volumes and convert them to NCDHW tensors."""
    if ubp_volume.shape != fdm_volume.shape:
        raise ValueError("UBP and FDM volumes must have identical shapes.")
    ubp_tensor = torch.from_numpy(normalize_volume(ubp_volume)).unsqueeze(0).unsqueeze(0)
    fdm_tensor = torch.from_numpy(normalize_volume(fdm_volume)).unsqueeze(0).unsqueeze(0)
    return (
        ubp_tensor.to(device, dtype=torch.float32),
        fdm_tensor.to(device, dtype=torch.float32),
    )


def remove_with_proposed(
    ubp_volume: np.ndarray,
    fdm_volume: np.ndarray,
    device: torch.device,
    lambda_cross: float = LAMBDA_CROSS,
) -> np.ndarray:
    """Train and apply the proposed dual-domain artifact network."""
    if not 0 <= lambda_cross <= 1:
        raise ValueError("lambda_cross must lie in [0, 1].")
    ubp_tensor, fdm_tensor = prepare_proposed_volumes(ubp_volume, fdm_volume, device)
    model = ArtifactNet3D(input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA,
    )
    model.train()
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = 0.0
        for _ in range(ITERATIONS_PER_EPOCH):
            ubp_batch, fdm_batch = random_paired_crops(ubp_tensor, fdm_tensor)
            loss, _ = dual_domain_loss(ubp_batch, fdm_batch, model, lambda_cross=lambda_cross)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            average_loss = epoch_loss / ITERATIONS_PER_EPOCH
            print(f"Epoch {epoch}/{MAX_EPOCHS}, average loss={average_loss:.6f}")

    print(f"Proposed training completed in {time.time() - start_time:.2f} s")
    model.eval()
    inference_input = (1 - lambda_cross) * ubp_tensor + lambda_cross * fdm_tensor
    return tiled_artifact_correction(model, inference_input)


@torch.no_grad()
def tiled_artifact_correction(
    model: nn.Module,
    inputs: torch.Tensor,
    tile_size: int = INFERENCE_TILE_SIZE,
    halo: int = INFERENCE_HALO,
) -> np.ndarray:
    """Apply Eq. (11) in 96-cubed cores with the network's 3-voxel context."""
    if inputs.ndim != 5 or inputs.shape[:2] != (1, 1):
        raise ValueError("Tiled inference expects a single-channel NCDHW volume.")
    if tile_size < 1 or halo < INFERENCE_HALO:
        raise ValueError("Use a positive tile size and at least 3 context voxels.")
    shape = inputs.shape[2:]
    result = np.empty(shape, dtype=np.float32)
    for z in range(0, shape[0], tile_size):
        for y in range(0, shape[1], tile_size):
            for x in range(0, shape[2], tile_size):
                start = (z, y, x)
                stop = tuple(min(s + tile_size, n) for s, n in zip(start, shape))
                padded_start = tuple(max(0, s - halo) for s in start)
                padded_stop = tuple(min(n, e + halo) for e, n in zip(stop, shape))
                outer = tuple(slice(s, e) for s, e in zip(padded_start, padded_stop))
                inner = tuple(slice(s - p, e - p) for s, e, p in zip(start, stop, padded_start))
                core = tuple(slice(s, e) for s, e in zip(start, stop))
                patch = inputs[(slice(None), slice(None)) + outer]
                corrected = patch - model(patch)
                result[core] = corrected[0, 0][inner].cpu().numpy()
    return result


def remove_with_a2a(
    first_volume: np.ndarray,
    second_volume: np.ndarray,
    device: torch.device,
    full_ubp_volume: np.ndarray,
) -> np.ndarray:
    """Use nonnegative per-slice peak scaling on raw UBP inputs.

    Train on two 400/512 subsets and infer on full sparse UBP slices. Restore
    each full-input slice's peak before the final whole-volume save processing.
    """
    first = np.asarray(first_volume, dtype=np.float32)
    second = np.asarray(second_volume, dtype=np.float32)
    full = np.asarray(full_ubp_volume, dtype=np.float32)
    if first.shape != second.shape or first.shape != full.shape:
        raise ValueError("A2A input volumes must have identical shapes.")
    if first.ndim != 3 or first.size == 0 or any(
        not np.isfinite(volume).all() for volume in (first, second, full)
    ):
        raise ValueError("A2A requires finite, nonempty 3D volumes.")

    denoised = np.empty_like(first, dtype=np.float32)
    slice_count = first.shape[0]
    print(f"Processing {slice_count} A2A slices on {device}")

    for slice_index in range(slice_count):
        torch.manual_seed(RANDOM_SEED + slice_index)
        model = A2ANet2D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=A2A_LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=A2A_SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
        )
        images = [np.maximum(v[slice_index], 0) for v in (full, first, second)]
        peaks = [float(image.max()) for image in images]
        full_slice, first_slice, second_slice = [
            torch.from_numpy(image / max(peak, 1e-30))[None, None].to(device)
            for image, peak in zip(images, peaks)
        ]
        model.train()

        for _ in range(A2A_EPOCHS):
            clean_first = first_slice - model(first_slice)
            clean_second = second_slice - model(second_slice)
            loss = 0.5 * (
                functional.mse_loss(clean_first, second_slice)
                + functional.mse_loss(clean_second, first_slice)
            ) + functional.mse_loss(clean_first, clean_second)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            clean_slice = full_slice - model(full_slice)
        # Restore slice amplitude; output clipping/normalization happens at save time.
        denoised[slice_index] = clean_slice[0, 0].cpu().numpy() * peaks[0]
        print(f"A2A slice {slice_index + 1}/{slice_count} completed")

    return denoised


def remove_with_dip(
    ubp_volume: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Fit a 3D deep image prior directly to the normalized UBP volume."""
    target = torch.from_numpy(normalize_volume(ubp_volume).astype(np.float32))
    target = target.unsqueeze(0).unsqueeze(0).to(device)
    torch.manual_seed(RANDOM_SEED)
    network_input = torch.randn_like(target)
    model = DIPUNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=DIP_LEARNING_RATE)
    scaler = GradScaler(enabled=device.type == "cuda")
    model.train()

    for iteration in range(1, DIP_ITERATIONS + 1):
        optimizer.zero_grad()
        with autocast(enabled=device.type == "cuda"):
            prediction = model(network_input)
            loss = functional.mse_loss(prediction, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if iteration % 100 == 0 or iteration == 1:
            print(f"DIP iteration {iteration}/{DIP_ITERATIONS}, loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        result = model(network_input)
    return result.float().clamp(0, 1).cpu().squeeze().numpy().astype(np.float32)


def remove_with_svd(
    ubp_volume: np.ndarray,
    fdm_volume: np.ndarray,
    rank: int = SVD_RANK,
    axis: int = SVD_AXIS,
) -> np.ndarray:
    """Joint truncated SVD of nonnegative UBP/FDM; Y unfolding, rank 19.

    Both inputs use the common whole-volume normalization. The projected volume
    is normalized again by main.py before saving, like all other method outputs.
    """
    if ubp_volume.ndim != 3 or ubp_volume.shape != fdm_volume.shape:
        raise ValueError("SVD requires equal-shaped 3D volumes.")
    if (
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        or axis not in (0, 1, 2)
    ):
        raise ValueError("SVD axis must be 0, 1 or 2.")
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or not 1 <= rank <= ubp_volume.shape[axis]
    ):
        raise ValueError("SVD rank must be an integer within the unfolded dimension.")
    matrices = []
    for volume in (ubp_volume, fdm_volume):
        encoded = normalize_volume(volume)
        matrices.append(
            np.ascontiguousarray(
                np.moveaxis(encoded, axis, 0).reshape(volume.shape[axis], -1),
                dtype=np.float64,
            )
        )
    au, af = matrices
    _, vectors = np.linalg.eigh(au @ au.T + af @ af.T)
    basis = vectors[:, -rank:]
    mean = 0.5 * (au + af)
    projected = basis @ (basis.T @ mean)
    shape = (ubp_volume.shape[axis],) + tuple(ubp_volume.shape[i] for i in range(3) if i != axis)
    return np.moveaxis(projected.reshape(shape), 0, axis).astype(np.float32)


class N2IDnCNN(nn.Module):
    """20-layer DnCNN with batch normalization; directly predicts the image."""

    def __init__(self, depth: int = 20, width: int = 64):
        super().__init__()
        if depth < 3 or width < 1:
            raise ValueError("DnCNN needs at least three layers and positive width.")
        self.input = nn.Conv2d(1, width, 3, padding=1, bias=False)
        self.hidden = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(depth - 2)
            ]
        )
        self.output = nn.Conv2d(width, 1, 3, padding=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = functional.relu(self.input(inputs))
        return self.output(self.hidden(features))


def n2i_pair(splits, slice_index, target_group):
    """X:1 pair: mean of the other three splits predicts the held-out split."""
    inputs = np.stack(
        [
            np.asarray(volume[slice_index], dtype=np.float32)
            for group, volume in enumerate(splits)
            if group != target_group
        ]
    )
    target = np.asarray(splits[target_group][slice_index], dtype=np.float32)
    return inputs.mean(axis=0), target


def remove_with_n2i(
    split_volumes: list[np.ndarray],
    device: torch.device,
    effective_epochs: int = N2I_EFFECTIVE_EPOCHS,
    batch_size: int = N2I_BATCH_SIZE,
    data_scaling: float = N2I_DATA_SCALING,
) -> np.ndarray:
    """X:1 simulation adaptation using four signed, unnormalized UBP splits.

    Apply the same fixed 1/4096 input scale to every split, without clipping
    or independent min-max normalization. Average the four network predictions
    and undo the scale before final whole-volume clipping/normalization on save.
    """
    if len(split_volumes) != N2I_SPLITS:
        raise ValueError("Noise2Inverse requires exactly four disjoint splits.")
    shape = split_volumes[0].shape
    if len(shape) != 3 or any(v.shape != shape or not np.isfinite(v).all() for v in split_volumes):
        raise ValueError("Noise2Inverse requires equal-shaped finite 3D splits.")
    if effective_epochs < 1 or batch_size < 1 or not np.isfinite(data_scaling) or data_scaling <= 0:
        raise ValueError("Invalid Noise2Inverse budget or intensity scale.")
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    model = N2IDnCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pair_count = shape[0] * N2I_SPLITS
    passes = max(effective_epochs // N2I_SPLITS, 1)
    model.train()
    for epoch in range(passes):
        order = rng.permutation(pair_count)
        for begin in range(0, pair_count, batch_size):
            pairs = [
                n2i_pair(split_volumes, int(j // N2I_SPLITS), int(j % N2I_SPLITS))
                for j in order[begin : begin + batch_size]
            ]
            inputs = torch.from_numpy(
                np.stack([a for a, b in pairs])[:, None] * data_scaling
            ).to(device)
            targets = torch.from_numpy(
                np.stack([b for a, b in pairs])[:, None] * data_scaling
            ).to(device)
            optimizer.zero_grad()
            loss = functional.mse_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Noise2Inverse loss.")
            loss.backward()
            optimizer.step()
        print(f"Noise2Inverse pass {epoch + 1}/{passes}, loss={loss.item():.6f}")
    result = np.empty(shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for index in range(shape[0]):
            inputs = (
                np.stack([n2i_pair(split_volumes, index, j)[0] for j in range(N2I_SPLITS)])[:, None]
                * data_scaling
            )
            prediction = model(torch.from_numpy(inputs).to(device))
            result[index] = prediction.mean(dim=0)[0].cpu().numpy() / data_scaling
    return result


# The physical operator below is adapted from GPAIR and the supplied notebook.
# GPAIR source: https://github.com/ddffwyb/GPAIR
# MIT License (GPAIR operator)
#
# Copyright (c) 2025 Yibing Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class MBPDOperator(LinearOperator):
    """A: (x,y,z) C-order volume -> detector-major, time-minor pressure samples.

    Same lookup weights as GPAIR/MB-PD/mbpd_Sphere_256.ipynb. Changes:
    bounded voxel chunks, atan2 for zero denominators, matched time-range masks.
    """

    def __init__(
        self,
        detector_locations,
        grid_shape,
        ranges,
        res,
        vs,
        fs,
        num_times,
        integral_table,
        gradient_table,
        device="cpu",
        voxel_chunk=MBPD_VOXEL_CHUNK,
    ):
        self.device = torch.device(device)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.ranges = tuple(tuple(float(v) for v in r) for r in ranges)
        self.res, self.vs, self.fs = float(res), float(vs), float(fs)
        self.dt, self.num_times = 1.0 / self.fs, int(num_times)
        self.voxel_chunk = int(voxel_chunk)
        if len(self.grid_shape) != 3 or min(self.grid_shape) <= 0 or self.voxel_chunk <= 0:
            raise ValueError("Invalid grid/chunk size.")
        if min(self.res, self.vs, self.fs) <= 0 or self.num_times < 2:
            raise ValueError("Invalid acoustic/temporal sampling parameters.")
        if len(self.ranges) != 3 or any(len(r) != 2 or r[1] <= r[0] for r in self.ranges):
            raise ValueError("Expected three increasing spatial ranges.")
        if any(
            not np.isclose((r[1] - r[0]) / self.res, n, rtol=1e-6)
            for r, n in zip(self.ranges, self.grid_shape)
        ):
            raise ValueError("Grid shape/range/resolution mismatch.")
        self.locations = torch.as_tensor(
            np.asarray(detector_locations), dtype=torch.float32, device=self.device
        )
        if self.locations.ndim != 2 or self.locations.shape[1] != 3 or len(self.locations) == 0:
            raise ValueError("Detector locations must have shape [N,3].")
        if not torch.isfinite(self.locations).all():
            raise ValueError("Non-finite detector locations.")
        self.num_detectors = len(self.locations)
        self.integral = torch.as_tensor(
            integral_table, dtype=torch.float32, device=self.device
        ).contiguous()
        self.gradient = torch.as_tensor(
            gradient_table, dtype=torch.float32, device=self.device
        ).contiguous()
        if (
            self.integral.ndim != 3
            or len(set(self.integral.shape)) != 1
            or self.integral.shape != self.gradient.shape
        ):
            raise ValueError("Lookup tables must be equal-sized cubic arrays.")
        if not torch.isfinite(self.integral).all() or not torch.isfinite(self.gradient).all():
            raise ValueError("Non-finite lookup table.")
        self.n_lookup = int(self.integral.shape[0])
        if self.n_lookup < 4 or self.n_lookup % 2:
            raise ValueError("Lookup size must be even and >= 4.")
        self.alpha_interval = 3.1415926536 / 2.0 / self.n_lookup
        self.beta_interval = self.alpha_interval
        self.d_interval = 4.0 * self.res / self.n_lookup
        self.time_interval_length = int(2.0 * 1.7321 * self.res / self.vs * self.fs) + 1
        self.offsets = torch.arange(
            self.time_interval_length, device=self.device, dtype=torch.int64
        )
        self.axes = [
            torch.linspace(r[0], r[1] - self.res, n, device=self.device, dtype=torch.float32)
            for r, n in zip(self.ranges, self.grid_shape)
        ]
        self.num_voxels = int(np.prod(self.grid_shape))
        super().__init__(
            dtype=np.dtype("float32"), shape=(self.num_detectors * self.num_times, self.num_voxels)
        )

    def coordinates(self, start, stop):
        ids = torch.arange(start, stop, device=self.device)
        ny, nz = self.grid_shape[1:]
        return torch.stack(
            (
                self.axes[0][ids // (ny * nz)],
                self.axes[1][(ids // nz) % ny],
                self.axes[2][ids % nz],
            ),
            dim=1,
        )

    def weights_and_indices(self, coordinates, detector_index):
        vectors = self.locations[detector_index] - coordinates
        distances = torch.linalg.vector_norm(vectors, dim=1)
        first = torch.floor((distances - 1.7321 * self.res) / self.vs * self.fs).to(torch.int64)
        # Equivalent to abs(atan(y/x)) away from singular axes; well defined at x=y=0.
        alpha = (
            (torch.atan2(vectors[:, 1].abs(), vectors[:, 0].abs()) / self.alpha_interval)
            .to(torch.int64)
            .clamp(0, self.n_lookup - 1)
        )
        beta = (
            (
                torch.atan2(vectors[:, 2].abs(), torch.linalg.vector_norm(vectors[:, :2], dim=1))
                / self.beta_interval
            )
            .to(torch.int64)
            .clamp(0, self.n_lookup - 1)
        )
        times = first[:, None] + self.offsets[None, :]
        valid = (times > 0) & (times < self.num_times)
        # Use a safe denominator before masking; t=0 is singular in the source formula.
        safe_times = times.clamp(min=1)
        radius = self.vs * safe_times * self.dt
        # Truncation toward zero, NOT floor: retains the original lookup discretization.
        d_index = ((distances[:, None] - radius) / self.d_interval).to(torch.int64)
        d_index = (d_index + self.n_lookup // 2).clamp(0, self.n_lookup - 1)
        area = self.integral[d_index, alpha[:, None], beta[:, None]]
        derivative = self.gradient[d_index, alpha[:, None], beta[:, None]]
        weights = -self.vs / radius * (area / radius + derivative)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        indices = times.clamp(0, self.num_times - 1).long()
        return weights, indices

    @torch.inference_mode()
    def _apply(self, vector, transpose):
        array = np.asarray(vector).reshape(-1)
        expected = self.shape[0] if transpose else self.shape[1]
        if array.size != expected or not np.isfinite(array).all():
            raise ValueError("Operator input has wrong length or non-finite values.")
        values = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if transpose:
            result = torch.zeros(self.num_voxels, dtype=torch.float32, device=self.device)
            measurements = values.reshape(self.num_detectors, self.num_times)
        else:
            result = torch.zeros(
                (self.num_detectors, self.num_times), dtype=torch.float32, device=self.device
            )
        for start in range(0, self.num_voxels, self.voxel_chunk):
            stop = min(start + self.voxel_chunk, self.num_voxels)
            coordinates = self.coordinates(start, stop)
            for detector in range(self.num_detectors):
                weights, indices = self.weights_and_indices(coordinates, detector)
                if transpose:
                    result[start:stop] += (weights * measurements[detector, indices]).sum(dim=1)
                else:
                    result[detector].scatter_add_(
                        0, indices.reshape(-1), (weights * values[start:stop, None]).reshape(-1)
                    )
        output = result.reshape(-1).cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Non-finite MB-PD operator output.")
        return output

    def _matvec(self, vector):
        return self._apply(vector, False)

    def _rmatvec(self, vector):
        return self._apply(vector, True)


def load_mbpd_tables(table_dir: str | Path, resolution: float = 0.1e-3):
    """Load the verified GPAIR integration tables used by the simulation notebooks."""
    from scipy.io import loadmat

    arrays = []
    for filename, expected_hash in MBPD_TABLE_HASHES.items():
        path = Path(table_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Obtain both sphere_integral*.mat files from "
                f"{MBPD_TABLE_SOURCE} and set --mbpd-table-dir."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Lookup-table checksum mismatch: {path}")
        array = np.asarray(loadmat(path)[path.stem], dtype=np.float32)
        if array.shape != (100, 100, 100) or not np.isfinite(array).all():
            raise ValueError(f"Invalid integration table: {path}")
        arrays.append(array)
    ratio = resolution / 0.1e-3
    return np.ascontiguousarray(arrays[0] * ratio**2), np.ascontiguousarray(arrays[1] * ratio)


def remove_with_mbpd(
    operator: LinearOperator,
    signals: np.ndarray,
    damp: float = MBPD_DAMP,
    max_iterations: int = MBPD_MAX_ITERATIONS,
) -> np.ndarray:
    """Solve ||Ax-b||^2 + damp^2 ||x||^2 with damped LSQR; retain signed output."""
    if not np.isfinite(damp) or damp < 0 or max_iterations < 1:
        raise ValueError("Invalid MB-PD damping or iteration budget.")
    solution = lsqr(
        operator,
        np.asarray(signals, dtype=np.float32).reshape(-1),
        damp=damp,
        atol=MBPD_TOLERANCE,
        btol=MBPD_TOLERANCE,
        conlim=1e8,
        iter_lim=max_iterations,
        show=True,
    )
    result = solution[0].reshape(operator.grid_shape).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite MB-PD reconstruction.")
    print(f"MB-PD completed: {solution[2]} iterations, LSQR stop code {solution[1]}")
    return result


def bm4d_denoise(volume, sigma, threads=4):
    if not np.isfinite(sigma) or sigma <= 0 or threads < 1:
        raise ValueError("BM4D sigma and thread count must be positive.")
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3 or min(array.shape) < 5 or not np.isfinite(array).all():
        raise ValueError("BM4D requires a finite 3-D volume, each dimension >=5.")
    import bm4d
    from bm4d.profiles import BM4DProfile

    profile = BM4DProfile()
    profile.num_threads = int(threads)
    result = np.asarray(bm4d.bm4d(array, sigma_psd=float(sigma), profile=profile), dtype=np.float32)
    if result.shape != array.shape or not np.isfinite(result).all():
        raise FloatingPointError("BM4D returned an invalid volume.")
    return result


def norm64(array):
    a = np.asarray(array).reshape(-1)
    return float(np.sqrt(np.sum(np.square(a, dtype=np.float64), dtype=np.float64)))


def prepare_pnp_scaling(operator, b, power_iters=6, seed=2026):
    """Data-only, one-time scale selection; no reference image is accepted."""
    if power_iters < 1:
        raise ValueError("power_iters must be positive.")
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(operator.shape[1]).astype(np.float32)
    q /= norm64(q)
    for _ in range(power_iters):
        aq = operator.matvec(q)
        eigenvalue = norm64(aq) ** 2
        normal_q = operator.rmatvec(aq)
        denominator = norm64(normal_q)
        if denominator <= 0 or not np.isfinite(eigenvalue):
            raise ValueError("Degenerate forward operator during scale estimation.")
        q = np.asarray(normal_q / denominator, dtype=np.float32)
    atb = np.asarray(operator.rmatvec(b), dtype=np.float32)
    positive_peak = float(np.max(atb))
    if positive_peak <= 0 or eigenvalue <= 0:
        raise ValueError("No positive backprojection support; check geometry/sign conventions.")
    # z = physical_volume / volume_scale; A_normalized = A/sqrt(L).
    volume_scale = positive_peak / eigenvalue
    atbn = atb / (eigenvalue * volume_scale)
    metadata = {"normal_operator_scale": eigenvalue, "volume_scale": volume_scale}
    return np.asarray(atbn, dtype=np.float32), metadata


def data_consistency_update(operator, atbn, target, x0, rho, normal_scale, max_iter=10, rtol=1e-4):
    """Solve (A.T A / L + rho I)x = A.T b/(L*s) + rho*(v-u)."""
    if rho <= 0 or normal_scale <= 0 or max_iter < 1 or rtol <= 0:
        raise ValueError("Invalid PnP/CG parameters.")

    def normal(vector):
        projected = operator.matvec(vector)
        return np.asarray(
            operator.rmatvec(projected) / normal_scale + rho * vector, dtype=np.float32
        )

    normal_operator = LinearOperator(
        (operator.shape[1], operator.shape[1]), matvec=normal, dtype=np.float32
    )
    rhs = np.asarray(atbn + rho * np.asarray(target).reshape(-1), dtype=np.float32)
    count = [0]

    def tick(_):
        count[0] += 1

    solution, info = cg(
        normal_operator,
        rhs,
        x0=np.asarray(x0, dtype=np.float32).reshape(-1),
        maxiter=max_iter,
        rtol=rtol,
        atol=0.0,
        callback=tick,
    )
    if info < 0 or not np.isfinite(solution).all():
        raise RuntimeError(f"CG breakdown: info={info}")
    # Explicit inner residual: recorded even when the inner budget is exhausted.
    relative = norm64(normal(solution) - rhs) / max(norm64(rhs), 1e-30)
    return np.asarray(solution, dtype=np.float32), {
        "cg_iterations": count[0],
        "cg_info": int(info),
        "cg_relative_residual": relative,
        "cg_converged": info == 0,
    }


def remove_with_pnp(
    operator: LinearOperator,
    signals: np.ndarray,
    outer_iterations: int = 20,
    rho: float = 0.1,
    sigma: float = 0.02,
) -> np.ndarray:
    """Measurement-domain PnP-ADMM with fixed-scale, full-volume BM4D.

    Requires the optional author package: pip install bm4d.
    Forward/adjoint operations run on the operator device; BM4D runs on CPU.
    """
    if (
        outer_iterations < 1
        or not np.isfinite(rho)
        or rho <= 0
        or not np.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError("Invalid PnP parameters.")
    b = np.asarray(signals, dtype=np.float32).reshape(-1)
    if b.size != operator.shape[0] or not np.isfinite(b).all() or norm64(b) == 0:
        raise ValueError("Invalid PnP measurements.")
    atbn, scaling = prepare_pnp_scaling(operator, b, power_iters=6, seed=RANDOM_SEED)
    v = np.maximum(atbn, 0)
    x, u = v.copy(), np.zeros_like(v)
    for iteration in range(1, outer_iterations + 1):
        x, stats = data_consistency_update(
            operator,
            atbn,
            v - u,
            x,
            rho,
            scaling["normal_operator_scale"],
            max_iter=10,
            rtol=1e-4,
        )
        v = np.maximum(
            bm4d_denoise((x + u).reshape(operator.grid_shape), sigma, threads=4), 0
        ).reshape(-1)
        u += x - v
        print(
            f"PnP {iteration}/{outer_iterations}, CG info={stats['cg_info']}, "
            f"relative residual={stats['cg_relative_residual']:.3g}"
        )
    return (v.reshape(operator.grid_shape) * scaling["volume_scale"]).astype(np.float32)


def frangi_response_from_eigenvalues(eigenvalues, alpha=0.5, beta=0.5, gamma=0.1):
    if min(alpha, beta, gamma) <= 0:
        raise ValueError("Frangi alpha/beta/gamma must be positive.")
    eigenvalues = np.asarray(eigenvalues, dtype=np.float32)
    if eigenvalues.shape[0] != 3 or not np.isfinite(eigenvalues).all():
        raise ValueError("Expected three finite Hessian eigenvalue fields.")
    order = np.argsort(np.abs(eigenvalues), axis=0)
    ordered = np.take_along_axis(eigenvalues, order, axis=0)
    l1, l2, l3 = ordered
    # Bright tubes on dark background: the two dominant curvatures must be negative.
    valid = (l2 < 0) & (l3 < 0)
    tiny = np.finfo(np.float32).tiny
    ra = np.abs(l2) / np.maximum(np.abs(l3), tiny)
    rb = np.abs(l1) / np.sqrt(np.maximum(np.abs(l2 * l3), tiny))
    strength = np.sum(ordered**2, axis=0)
    response = (
        -np.expm1(-(ra**2) / (2 * alpha**2))
        * np.exp(-(rb**2) / (2 * beta**2))
        * (-np.expm1(-strength / (2 * gamma**2)))
    )
    return np.where(valid, response, 0).astype(np.float32)


def remove_with_frangi(
    ubp_volume: np.ndarray,
    sigmas_mm: tuple[float, ...] = (0.1, 0.2, 0.3),
    spacing_mm: float = 0.1,
) -> np.ndarray:
    """3D bright-vessel Frangi response, with fixed alpha=beta=0.5 and gamma=0.1.

    Requires: pip install scikit-image. Scales are Gaussian sigmas in mm.
    Output is dimensionless vesselness, not reconstructed pressure.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    if (
        not np.isfinite(spacing_mm)
        or spacing_mm <= 0
        or not sigmas_mm
        or any(not np.isfinite(s) or s <= 0 for s in sigmas_mm)
    ):
        raise ValueError("Frangi spacing and Gaussian scales must be finite and positive.")
    image = normalize_volume(ubp_volume)
    if min(image.shape) < 7:
        raise ValueError("Frangi requires at least seven voxels along each axis.")
    vesselness = np.zeros_like(image)
    for sigma_mm in sorted(set(sigmas_mm)):
        sigma = sigma_mm / spacing_mm
        hessian = hessian_matrix(
            image, sigma=sigma, mode="reflect", order="rc", use_gaussian_derivatives=True
        )
        eigenvalues = hessian_matrix_eigvals(hessian) * sigma**2
        del hessian
        response = frangi_response_from_eigenvalues(eigenvalues)
        np.maximum(vesselness, response, out=vesselness)
        print(f"Frangi Gaussian sigma={sigma_mm:g} mm completed")
    return vesselness

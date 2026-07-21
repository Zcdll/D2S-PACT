"""Proposed and comparison artifact-removal methods for 3D PACT."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast


MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.6
PATCH_SIZE = (64, 64, 64)
BATCH_SIZE = 2
ITERATIONS_PER_EPOCH = 20
LAMBDA_CROSS = 1.0
LAMBDA_CONSISTENCY = 10.0
UNCERTAINTY_ALPHA = 12.0
EMBEDDING_CHANNELS = 32

A2A_EPOCHS = 300
A2A_LEARNING_RATE = 1e-3
A2A_CHANNELS = 64

DIP_ITERATIONS = 5500
DIP_LEARNING_RATE = 1e-3
DIP_BASE_CHANNELS = 8

SVD_RANK = 30
SVD_INTERSECTION_MODE = "multiply"


class ArtifactNet3D(nn.Module):
    """Lightweight shared-weight residual network from the proposed method."""

    def __init__(
        self,
        input_channels: int = 1,
        embedding_channels: int = EMBEDDING_CHANNELS,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.convolution_1 = nn.Conv3d(
            input_channels, embedding_channels, 3, padding=1
        )
        self.convolution_2 = nn.Conv3d(
            embedding_channels, embedding_channels, 3, padding=1
        )
        self.convolution_3 = nn.Conv3d(
            embedding_channels, embedding_channels, 3, padding=1
        )
        self.output = nn.Conv3d(embedding_channels, input_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.convolution_1(inputs))
        features = self.activation(self.convolution_2(features))
        features = self.activation(self.convolution_3(features))
        return self.output(features)


class A2ANet2D(nn.Module):
    """Slice-wise convolutional network for the A2A comparison."""

    def __init__(self, channels: int = A2A_CHANNELS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ConvBlock3D(nn.Module):
    """Two-layer convolution block used by the DIP U-Net."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1),
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
        self.up_2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.decoder_2 = ConvBlock3D(base_channels * 4, base_channels * 2)
        self.up_1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder_1 = ConvBlock3D(base_channels * 2, base_channels)
        self.output = nn.Sequential(nn.Conv3d(base_channels, 1, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_1 = self.encoder_1(inputs)
        encoded_2 = self.encoder_2(self.pool(encoded_1))
        bottleneck = self.bottleneck(self.pool(encoded_2))
        decoded_2 = self.decoder_2(torch.cat([self.up_2(bottleneck), encoded_2], dim=1))
        decoded_1 = self.decoder_1(torch.cat([self.up_1(decoded_2), encoded_1], dim=1))
        return self.output(decoded_1)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Apply the min-max normalization used by the notebooks."""
    return (volume - volume.min()) / (volume.max() - volume.min())


def dual_domain_loss(
    ubp: torch.Tensor,
    fdm: torch.Tensor,
    model: nn.Module,
    lambda_cross: float = LAMBDA_CROSS,
    lambda_consistency: float = LAMBDA_CONSISTENCY,
    alpha: float = UNCERTAINTY_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the proposed cross-domain and uncertainty-weighted losses."""
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
) -> np.ndarray:
    """Train and apply the proposed dual-domain artifact network."""
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
            loss, _ = dual_domain_loss(ubp_batch, fdm_batch, model)
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
    with torch.no_grad():
        clean_volume = ubp_tensor - model(ubp_tensor)
    return clean_volume.clamp(0, 1).cpu().squeeze().numpy().astype(np.float32)


def remove_with_a2a(
    first_volume: np.ndarray,
    second_volume: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run slice-wise Artifact2Artifact training on two independent UBP volumes."""
    first = normalize_volume(first_volume).astype(np.float32)
    second = normalize_volume(second_volume).astype(np.float32)
    if first.shape != second.shape:
        raise ValueError("A2A input volumes must have identical shapes.")

    denoised = np.empty_like(first, dtype=np.float32)
    slice_count = first.shape[0]
    print(f"Processing {slice_count} A2A slices on {device}")

    for slice_index in range(slice_count):
        model = A2ANet2D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=A2A_LEARNING_RATE)
        first_slice = torch.from_numpy(first[slice_index]).unsqueeze(0).unsqueeze(0).to(device)
        second_slice = torch.from_numpy(second[slice_index]).unsqueeze(0).unsqueeze(0).to(device)
        model.train()

        for _ in range(A2A_EPOCHS):
            prediction = model(first_slice)
            loss = functional.mse_loss(prediction, second_slice)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            clean_slice = model(first_slice)
        denoised[slice_index] = clean_slice.clamp(0, 1).cpu().squeeze().numpy()
        print(f"A2A slice {slice_index + 1}/{slice_count} completed")

    return denoised


def remove_with_dip(
    ubp_volume: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Fit a 3D deep image prior directly to the normalized UBP volume."""
    target = torch.from_numpy(normalize_volume(ubp_volume).astype(np.float32))
    target = target.unsqueeze(0).unsqueeze(0).to(device)
    network_input = torch.rand_like(target)
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
    intersection_mode: str = SVD_INTERSECTION_MODE,
) -> np.ndarray:
    """Apply low-rank SVD to the shared component of UBP and FDM volumes."""
    if ubp_volume.shape != fdm_volume.shape:
        raise ValueError(
            f"Volume shapes must match: {ubp_volume.shape} and {fdm_volume.shape}."
        )
    if ubp_volume.ndim != 3:
        raise ValueError("SVD inputs must be 3D volumes.")

    normalized_ubp = normalize_volume(ubp_volume)
    normalized_fdm = normalize_volume(fdm_volume)
    if intersection_mode == "multiply":
        intersection = normalized_ubp * normalized_fdm
    elif intersection_mode == "binary":
        ubp_mask = normalized_ubp > 0.1
        fdm_mask = normalized_fdm > 0.1
        intersection = (ubp_mask & fdm_mask).astype(np.float32)
    else:
        raise ValueError(f"Unsupported intersection mode: {intersection_mode}")

    depth, height, width = intersection.shape
    matrix = intersection.reshape(depth, -1)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix, full_matrices=False
    )
    retained_rank = min(rank, len(singular_values))
    low_rank = (
        left_vectors[:, :retained_rank] * singular_values[:retained_rank]
    ) @ right_vectors[:retained_rank, :]
    return np.clip(low_rank.reshape(depth, height, width), 0, 1).astype(np.float32)

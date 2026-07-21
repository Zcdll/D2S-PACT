"""Dual-domain 3D PACT reconstruction for spherical detector arrays."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import time
from pathlib import Path

import numpy as np
import scipy.integrate
import torch
from gapat.algorithms import recon
from gapat.processings import negetive_processing
from scipy.special import sph_harm, spherical_jn, spherical_yn


SOUND_SPEED = 1500.0
SAMPLING_RATE = 40e6
TIME_SAMPLES = 2048
VOXEL_SIZE = 0.1e-3
GRID_SIZE = (256, 256, 256)
FOV_SIZE = (2.56e-2, 2.56e-2, 2.56e-2)
X_RANGE = (-12.8e-3, 12.8e-3)
Y_RANGE = (-12.8e-3, 12.8e-3)
Z_RANGE = (-12.8e-3, 12.8e-3)
DETECTOR_STRIDE = 8
SPHERICAL_HARMONIC_ORDER = 85
VIRTUAL_DIRECTIONS = 15000
A2A_SAMPLE_SIZE = 800
SIGNAL_ARCHIVE_KEY = "signals"


def load_measurements(
    signal_path: str | Path,
    location_path: str | Path,
    detector_stride: int = DETECTOR_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load NPZ or text measurements and apply sparse detector sampling."""
    signal_file = Path(signal_path)
    if signal_file.suffix.lower() == ".npz":
        with np.load(signal_file) as archive:
            signals = archive[SIGNAL_ARCHIVE_KEY].astype(np.float32, copy=False)
    else:
        signals = np.loadtxt(signal_file).astype(np.float32)
    locations = np.loadtxt(location_path).astype(np.float32)

    if signals.ndim != 2:
        raise ValueError(f"Expected a 2D signal matrix, got {signals.shape}.")
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError(f"Expected detector coordinates with shape (N, 3), got {locations.shape}.")
    if signals.shape[0] != locations.shape[0]:
        raise ValueError("The signal and detector counts do not match.")
    if detector_stride < 1:
        raise ValueError("detector_stride must be at least 1.")

    return signals[::detector_stride], locations[::detector_stride]


def sample_independent_measurements(
    signals: np.ndarray,
    locations: np.ndarray,
    sample_size: int = A2A_SAMPLE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the two independent sparse acquisitions used by A2A."""
    if sample_size > signals.shape[0]:
        raise ValueError("sample_size cannot exceed the number of detectors.")

    first_indices = np.random.choice(signals.shape[0], sample_size, replace=False)
    second_indices = np.random.choice(signals.shape[0], sample_size, replace=False)
    return (
        signals[first_indices],
        locations[first_indices],
        signals[second_indices],
        locations[second_indices],
    )


def spherical_hankel1(order: int, values: np.ndarray) -> np.ndarray:
    """Evaluate the spherical Hankel function of the first kind."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return spherical_jn(order, values) + 1j * spherical_yn(order, values)


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_radius = np.hypot(x, y)
    radius = np.hypot(horizontal_radius, z)
    polar_angle = np.arctan2(horizontal_radius, z)
    azimuth = np.arctan2(y, x)
    azimuth[azimuth < 0] += 2 * np.pi
    return radius, polar_angle, azimuth


def fibonacci_sphere(samples: int = 10000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []

    for index in range(samples):
        z_value = 1 - (index / float(samples - 1)) * 2
        horizontal_radius = np.sqrt(1 - z_value * z_value)
        azimuth = golden_angle * index
        x_values.append(np.cos(azimuth) * horizontal_radius)
        y_values.append(np.sin(azimuth) * horizontal_radius)
        z_values.append(z_value)

    return np.array(x_values), np.array(y_values), np.array(z_values)


def spherical_harmonic_tensor(
    maximum_order: int,
    azimuth: np.ndarray,
    polar_angle: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Build the spherical harmonic basis used by the FDM inversion."""
    start_time = time.time()
    harmonic_count = (maximum_order + 1) ** 2
    harmonic_matrix = np.zeros((len(azimuth), harmonic_count), dtype=np.complex64)

    orders: list[int] = []
    modes: list[int] = []
    for order in range(maximum_order + 1):
        for mode in range(-order, order + 1):
            orders.append(order)
            modes.append(mode)

    order_array = np.array(orders)
    mode_array = np.array(modes)
    core_count = multiprocessing.cpu_count()
    chunk_size = max(1, len(order_array) // (core_count * 4))

    def calculate_chunk(start: int, end: int) -> np.ndarray:
        order_chunk = order_array[start:end][np.newaxis, :]
        mode_chunk = mode_array[start:end][np.newaxis, :]
        azimuth_grid = azimuth[:, np.newaxis]
        polar_grid = polar_angle[:, np.newaxis]
        return sph_harm(mode_chunk, order_chunk, azimuth_grid, polar_grid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=core_count) as executor:
        futures = {}
        for start in range(0, len(order_array), chunk_size):
            end = min(start + chunk_size, len(order_array))
            futures[executor.submit(calculate_chunk, start, end)] = (start, end)

        for future in concurrent.futures.as_completed(futures):
            start, end = futures[future]
            harmonic_matrix[:, start:end] = future.result()

    print(f"Spherical harmonic matrix computed in {time.time() - start_time:.2f} s")
    return torch.from_numpy(harmonic_matrix).to(device)


@torch.jit.script
def _grid_kernel(
    index_x: torch.Tensor,
    index_y: torch.Tensor,
    index_z: torch.Tensor,
    values: torch.Tensor,
    size_x: int,
    size_y: int,
    size_z: int,
    sigma: float,
):
    base_x = torch.floor(index_x).long()
    base_y = torch.floor(index_y).long()
    base_z = torch.floor(index_z).long()

    delta_x = index_x - base_x
    delta_y = index_y - base_y
    delta_z = index_z - base_z
    inverse_sigma = 1.0 / (2 * sigma**2)

    weight_x = torch.empty(
        (index_x.shape[0], 3), dtype=index_x.dtype, device=index_x.device
    )
    weight_y = torch.empty(
        (index_x.shape[0], 3), dtype=index_x.dtype, device=index_x.device
    )
    weight_z = torch.empty(
        (index_x.shape[0], 3), dtype=index_x.dtype, device=index_x.device
    )

    for kernel_index in range(3):
        offset = kernel_index - 1
        weight_x[:, kernel_index] = torch.exp(-(delta_x - offset) ** 2 * inverse_sigma)
        weight_y[:, kernel_index] = torch.exp(-(delta_y - offset) ** 2 * inverse_sigma)
        weight_z[:, kernel_index] = torch.exp(-(delta_z - offset) ** 2 * inverse_sigma)

    grid = torch.zeros(
        size_x * size_y * size_z, dtype=torch.complex64, device=index_x.device
    )
    grid_weights = torch.zeros(
        size_x * size_y * size_z, dtype=torch.float32, device=index_x.device
    )
    stride_x = size_z * size_y
    stride_y = size_z

    for x_offset in range(3):
        current_x_weight = weight_x[:, x_offset]
        current_x = base_x + (x_offset - 1)
        for y_offset in range(3):
            current_y_weight = weight_y[:, y_offset]
            current_y = base_y + (y_offset - 1)
            xy_weight = current_x_weight * current_y_weight
            xy_index = current_x * stride_x + current_y * stride_y
            for z_offset in range(3):
                current_z_weight = weight_z[:, z_offset]
                current_z = base_z + (z_offset - 1)
                weight = xy_weight * current_z_weight
                flat_index = xy_index + current_z
                grid.index_add_(0, flat_index, values * weight.to(torch.complex64))
                grid_weights.index_add_(0, flat_index, weight)

    return grid, grid_weights


def grid_spherical_spectrum(
    points: torch.Tensor,
    values: torch.Tensor,
    grid_size: tuple[int, int, int],
    fov_size: tuple[float, float, float],
) -> torch.Tensor:
    size_x, size_y, size_z = grid_size
    fov_x, fov_y, fov_z = fov_size

    delta_k_x = 2 * np.pi / fov_x
    delta_k_y = 2 * np.pi / fov_y
    delta_k_z = 2 * np.pi / fov_z
    index_x = (points[:, 0] / delta_k_x) + (size_x / 2)
    index_y = (points[:, 1] / delta_k_y) + (size_y / 2)
    index_z = (points[:, 2] / delta_k_z) + (size_z / 2)

    kernel_width = 2
    valid = (
        (index_x >= kernel_width)
        & (index_x < size_x - kernel_width)
        & (index_y >= kernel_width)
        & (index_y < size_y - kernel_width)
        & (index_z >= kernel_width)
        & (index_z < size_z - kernel_width)
    )

    grid, weights = _grid_kernel(
        index_x[valid],
        index_y[valid],
        index_z[valid],
        values[valid],
        size_x,
        size_y,
        size_z,
        0.5,
    )
    populated = weights > 1e-6
    grid[populated] /= weights[populated]
    return grid.reshape(size_x, size_y, size_z)


def reconstruct_fdm(
    signal_data: np.ndarray,
    probe_coordinates: np.ndarray,
    sampling_rate: float = SAMPLING_RATE,
    sound_speed: float = SOUND_SPEED,
    grid_size: tuple[int, int, int] = GRID_SIZE,
    fov_size: tuple[float, float, float] = FOV_SIZE,
    maximum_order: int = SPHERICAL_HARMONIC_ORDER,
    virtual_directions: int = VIRTUAL_DIRECTIONS,
    device_name: str = "cuda",
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reconstruct a 3D volume using the spherical Fourier-domain method."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    probe_count, time_count = signal_data.shape
    size_x, size_y, size_z = grid_size
    probe_radius, probe_polar, probe_azimuth = cartesian_to_spherical(
        probe_coordinates[:, 0],
        probe_coordinates[:, 1],
        probe_coordinates[:, 2],
    )
    radius = np.mean(probe_radius)
    angular_weights = np.full(probe_count, (4 * np.pi) / probe_count)

    padded_time_count = time_count * 2
    spectrum = torch.from_numpy(
        np.conj(np.fft.rfft(signal_data, n=padded_time_count, axis=1))
    ).to(device)
    wave_numbers = (
        2
        * np.pi
        * np.fft.rfftfreq(padded_time_count, d=1 / sampling_rate)
        / sound_speed
    )

    print(f"Analyzing spherical harmonics with l_max={maximum_order}")
    probe_harmonics = spherical_harmonic_tensor(
        maximum_order,
        probe_azimuth,
        probe_polar,
        device,
    )
    weight_tensor = torch.from_numpy(angular_weights).float().to(device).unsqueeze(1)
    modal_pressure = torch.matmul(
        probe_harmonics.to(torch.complex64).conj().T,
        spectrum.to(torch.complex64) * weight_tensor.to(torch.complex64),
    )

    modal_zero = modal_pressure[0, :].cpu().numpy()
    zero_factor = 1j * (radius**2) / (np.sqrt(2) * (np.pi**2.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc_term = np.sin(wave_numbers * radius) / (wave_numbers * radius)
        zero_mask = wave_numbers == 0
        sinc_term[zero_mask] = 1.0
        bracket_term = sinc_term - np.cos(wave_numbers * radius)
        integrand = (
            (1.0 / wave_numbers)
            * modal_zero
            * np.exp(-1j * wave_numbers * radius)
            * bracket_term
        )
    integrand[zero_mask] = 0.0
    zero_value = zero_factor * scipy.integrate.trapezoid(integrand, x=wave_numbers)
    zero_value_tensor = torch.tensor(zero_value, dtype=torch.complex64, device=device)

    print("Applying the radial filter")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        filtered_modes = np.zeros(modal_pressure.shape, dtype=np.complex64)
        modal_pressure_cpu = modal_pressure.cpu().numpy()
        for order in range(maximum_order + 1):
            radial_position = wave_numbers * radius
            hankel_values = spherical_hankel1(order, radial_position)
            valid_wave = radial_position >= max(0, order - 5)
            denominator = (wave_numbers**2) * hankel_values
            numerator = np.sqrt(2 / np.pi) * (1j) ** order
            denominator_magnitude = np.real(denominator * np.conj(denominator))
            radial_filter = numerator * (
                np.conj(denominator) / (denominator_magnitude + 1e-10)
            )
            radial_filter[~valid_wave] = 0
            radial_filter[np.isnan(radial_filter)] = 0
            start = order**2
            end = (order + 1) ** 2
            filtered_modes[start:end, :] = (
                modal_pressure_cpu[start:end, :] * radial_filter
            )

    print(f"Synthesizing {virtual_directions} virtual directions")
    direction_x, direction_y, direction_z = fibonacci_sphere(virtual_directions)
    _, direction_polar, direction_azimuth = cartesian_to_spherical(
        direction_x,
        direction_y,
        direction_z,
    )
    virtual_harmonics = spherical_harmonic_tensor(
        maximum_order,
        direction_azimuth,
        direction_polar,
        device,
    )
    virtual_spectrum = torch.matmul(
        virtual_harmonics,
        torch.from_numpy(filtered_modes).cfloat().to(device),
    )

    wave_number_tensor = torch.from_numpy(wave_numbers).float().to(device)
    direction_x_tensor = torch.from_numpy(direction_x).float().to(device).unsqueeze(1)
    direction_y_tensor = torch.from_numpy(direction_y).float().to(device).unsqueeze(1)
    direction_z_tensor = torch.from_numpy(direction_z).float().to(device).unsqueeze(1)
    points = torch.stack(
        [
            (direction_x_tensor * wave_number_tensor).flatten(),
            (direction_y_tensor * wave_number_tensor).flatten(),
            (direction_z_tensor * wave_number_tensor).flatten(),
        ],
        dim=1,
    )
    values = virtual_spectrum.flatten()
    nonzero = torch.abs(values) > 1e-14

    print(f"Gridding {int(nonzero.sum().item())} spectral samples")
    k_space = grid_spherical_spectrum(
        points[nonzero], values[nonzero], grid_size, fov_size
    )
    k_space[size_x // 2, size_y // 2, size_z // 2] = zero_value_tensor

    image = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k_space)))
    volume = torch.abs(image).cpu().numpy()[::-1, ::-1, ::-1]
    axes = (
        np.linspace(-fov_size[0] / 2, fov_size[0] / 2, size_x),
        np.linspace(-fov_size[1] / 2, fov_size[1] / 2, size_y),
        np.linspace(-fov_size[2] / 2, fov_size[2] / 2, size_z),
    )
    return volume, axes


def reconstruct_ubp(
    signal_data: np.ndarray,
    probe_coordinates: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 3D volume using GAPAT universal back-projection."""
    volume = recon(
        signal_data,
        probe_coordinates,
        -probe_coordinates,
        list(X_RANGE),
        list(Y_RANGE),
        list(Z_RANGE),
        VOXEL_SIZE,
        SOUND_SPEED,
        SAMPLING_RATE,
        delay=0,
        method="ubp",
        device="gpu",
    )
    return negetive_processing(volume, method="zero")


def reconstruct_dual_domain(
    signal_data: np.ndarray,
    probe_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run the FDM and UBP reconstruction stages."""
    start = time.time()
    fdm_volume, axes = reconstruct_fdm(signal_data, probe_coordinates)
    print(f"FDM reconstruction completed in {time.time() - start:.2f} s")

    start = time.time()
    ubp_volume = reconstruct_ubp(signal_data, probe_coordinates)
    print(f"UBP reconstruction completed in {time.time() - start:.2f} s")
    return fdm_volume, ubp_volume, axes

# lib/processing/data_augment.py
"""
Data Augmentation Module for Polarimetry Data

This module provides data augmentation capabilities for polarimetry datasets,
handling both Mueller matrices (4x4) and regular 2D arrays with appropriate
transformations for each data type.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import logging
import random


class PolarimetryDataAugmenter:
    """
    Data augmentation class for polarimetry data with specialized handling
    for Mueller matrices and regular 2D arrays.
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the data augmenter.

        Args:
            log_level: Logging level
        """
        self.logger = self._setup_logger(log_level)

        # Import the specialized Mueller matrix transforms
        try:
            # Try multiple import paths
            try:
                from polar_augment.flip_mm import RandomMuellerFlip
                from polar_augment.rotation_mm import RandomMuellerRotation
            except ImportError:
                # Try adding lib directory to path
                import sys
                from pathlib import Path

                # Get current directory and look for lib
                current_dir = Path(__file__).parent if hasattr(Path(__file__), 'parent') else Path.cwd()
                lib_dir = current_dir.parent / "lib" if current_dir.name == "processing" else current_dir / "lib"

                if lib_dir.exists() and str(lib_dir) not in sys.path:
                    sys.path.insert(0, str(lib_dir))

                from polar_augment.flip_mm import RandomMuellerFlip
                from polar_augment.rotation_mm import RandomMuellerRotation

            self.RandomMuellerFlip = RandomMuellerFlip
            self.RandomMuellerRotation = RandomMuellerRotation
            self.logger.info("Successfully imported Mueller matrix augmentation functions")
        except ImportError as e:
            self.logger.warning(f"Mueller matrix functions not available: {e}")
            self.logger.info("Will use fallback augmentation (regular transforms for all data)")
            self.RandomMuellerFlip = None
            self.RandomMuellerRotation = None

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup and configure logger."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _is_mueller_matrix(self, array: np.ndarray) -> bool:
        """Check if array is a Mueller matrix (H, W, 4, 4)."""
        return len(array.shape) == 4 and array.shape[-2:] == (4, 4)

    def _is_2d_array(self, array: np.ndarray) -> bool:
        """Check if array is a 2D spatial array (H, W)."""
        return len(array.shape) == 2

    def _numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor with appropriate format."""
        # CRITICAL FIX: Convert to float32 to match PyTorch expectations
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        if self._is_mueller_matrix(array):
            # Mueller matrix: (H, W, 4, 4) -> (16, H, W) for the transforms
            return torch.from_numpy(array).permute(2, 3, 0, 1).reshape(16, array.shape[0], array.shape[1])
        elif self._is_2d_array(array):
            # 2D array: (H, W) -> (1, H, W) for transforms
            return torch.from_numpy(array).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")

    def _torch_to_numpy(self, tensor: torch.Tensor, original_shape: Tuple) -> np.ndarray:
        """Convert torch tensor back to numpy with original format."""
        if len(original_shape) == 4 and original_shape[-2:] == (4, 4):
            # Mueller matrix: (16, H, W) -> (H, W, 4, 4)
            h, w = original_shape[:2]
            return tensor.reshape(4, 4, h, w).permute(2, 3, 0, 1).cpu().numpy()
        elif len(original_shape) == 2:
            # 2D array: (1, H, W) -> (H, W)
            return tensor.squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unsupported original shape: {original_shape}")

    def rotate_data(self, data_dict: Dict[str, np.ndarray], angle: float,
                    mueller_keys: List[str] = None, regular_keys: List[str] = None) -> Dict[str, np.ndarray]:
        """Apply rotation to specified keys."""
        rotated_dict = data_dict.copy()

        if mueller_keys is None:
            mueller_keys = []
        if regular_keys is None:
            regular_keys = []

        for key in mueller_keys + regular_keys:
            if key not in data_dict:
                continue

            array = data_dict[key]
            is_mueller = self._is_mueller_matrix(array)

            if is_mueller and self.RandomMuellerRotation is not None:
                # Use specialized Mueller rotation
                tensor = self._numpy_to_torch(array)
                rotation_transform = self.RandomMuellerRotation(angle)
                rotated_tensor = rotation_transform(tensor)
                rotated_dict[key] = self._torch_to_numpy(rotated_tensor, array.shape)
            elif self._is_2d_array(array):
                # Regular 2D rotation
                tensor = self._numpy_to_torch(array)
                rotated_tensor = TF.rotate(tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                rotated_dict[key] = self._torch_to_numpy(rotated_tensor, array.shape)
            else:
                self.logger.warning(f"Skipping rotation for {key} with shape {array.shape}")

        return rotated_dict

    def flip_data(self, data_dict: Dict[str, np.ndarray], orientation: int,
                  mueller_keys: List[str] = None, regular_keys: List[str] = None) -> Dict[str, np.ndarray]:
        """Apply flip to specified keys."""
        flipped_dict = data_dict.copy()

        if mueller_keys is None:
            mueller_keys = []
        if regular_keys is None:
            regular_keys = []

        for key in mueller_keys + regular_keys:
            if key not in data_dict:
                continue

            array = data_dict[key]
            is_mueller = self._is_mueller_matrix(array)

            if is_mueller and self.RandomMuellerFlip is not None:
                # Use specialized Mueller flip
                tensor = self._numpy_to_torch(array)
                flip_transform = self.RandomMuellerFlip(orientation)
                flipped_tensor = flip_transform(tensor)
                flipped_dict[key] = self._torch_to_numpy(flipped_tensor, array.shape)
            elif self._is_2d_array(array):
                # Regular 2D flip
                tensor = self._numpy_to_torch(array)
                if orientation == 0:  # Horizontal
                    flipped_tensor = TF.hflip(tensor)
                elif orientation == 1:  # Vertical
                    flipped_tensor = TF.vflip(tensor)
                elif orientation == 2:  # Both
                    flipped_tensor = TF.vflip(TF.hflip(tensor))
                else:
                    raise ValueError(f"Invalid flip orientation: {orientation}")
                flipped_dict[key] = self._torch_to_numpy(flipped_tensor, array.shape)
            else:
                self.logger.warning(f"Skipping flip for {key} with shape {array.shape}")

        return flipped_dict

    def add_gaussian_noise(self, data_dict: Dict[str, np.ndarray], noise_level: float,
                           exclude_keys: List[str] = None) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to all arrays."""
        noisy_dict = data_dict.copy()

        if exclude_keys is None:
            exclude_keys = []

        for key, value in data_dict.items():
            if key in exclude_keys or not isinstance(value, np.ndarray):
                continue

            noise = np.random.normal(0, noise_level, value.shape).astype(value.dtype)
            noisy_dict[key] = value + noise

        return noisy_dict

    def load_and_augment_npz(self, npz_path: Union[str, Path],
                             augmentation_params: Dict) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Load an NPZ file and apply augmentations."""
        npz_path = Path(npz_path)

        # Load the NPZ file
        with np.load(npz_path, allow_pickle=True) as data:
            data_dict = {key: data[key] for key in data.keys()}

        # Apply augmentations
        augmented_data = self.augment_sample(data_dict, augmentation_params)

        return augmented_data

    def save_augmented_data(self, data: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]],
                            output_path: Union[str, Path]):
        """Save augmented data to NPZ file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, list):
            # Multiple variants (e.g., from cropping)
            for i, variant in enumerate(data):
                variant_path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
                np.savez_compressed(variant_path, **variant)
                self.logger.debug(f"Saved variant to {variant_path}")
        else:
            # Single variant
            np.savez_compressed(output_path, **data)
            self.logger.debug(f"Saved to {output_path}")

    def get_augmentation_combinations(self) -> List[Dict]:
        """Get default augmentation combinations."""
        return [
            {'name': 'original', 'rotation_angle': None, 'flip_orientation': None, 'noise_level': None,
             'crop_to_quarters': False},
            {'name': 'rot_90', 'rotation_angle': 90, 'flip_orientation': None, 'noise_level': None,
             'crop_to_quarters': False},
            {'name': 'rot_180', 'rotation_angle': 180, 'flip_orientation': None, 'noise_level': None,
             'crop_to_quarters': False},
            {'name': 'rot_270', 'rotation_angle': 270, 'flip_orientation': None, 'noise_level': None,
             'crop_to_quarters': False},
            {'name': 'flip_h', 'rotation_angle': None, 'flip_orientation': 0, 'noise_level': None,
             'crop_to_quarters': False},
            {'name': 'flip_v', 'rotation_angle': None, 'flip_orientation': 1, 'noise_level': None,
             'crop_to_quarters': False},
        ]

    def crop_to_regions(self, data_dict: Dict[str, np.ndarray],
                        crop_type: str = "quarters",
                        exclude_keys: List[str] = None) -> List[Dict[str, np.ndarray]]:
        """
        Crop all 2D/4D arrays into multiple regions.
        """
        if exclude_keys is None:
            exclude_keys = []

        # Get dimensions from first 2D array
        crop_height, crop_width = None, None
        for key, value in data_dict.items():
            if key not in exclude_keys and isinstance(value, np.ndarray):
                if len(value.shape) == 2:
                    crop_height, crop_width = value.shape
                    break
                elif len(value.shape) == 4 and value.shape[-2:] == (4, 4):
                    crop_height, crop_width = value.shape[:2]
                    break

        if crop_height is None or crop_width is None:
            self.logger.error("Could not determine dimensions for cropping")
            return [data_dict]

        # Calculate quarter sizes
        quarter_h = crop_height // 2
        quarter_w = crop_width // 2

        # Calculate center positions for the middle crops
        center_h_start = crop_height // 4
        center_w_start = crop_width // 4

        # Define crop regions based on type
        if crop_type == "quarters":
            crop_regions = [
                (0, quarter_h, 0, quarter_w, 'top_left'),
                (0, quarter_h, quarter_w, crop_width, 'top_right'),
                (quarter_h, crop_height, 0, quarter_w, 'bottom_left'),
                (quarter_h, crop_height, quarter_w, crop_width, 'bottom_right')
            ]
        elif crop_type == "centers":
            crop_regions = [
                (0, quarter_h, center_w_start, center_w_start + quarter_w, 'top_center'),
                (quarter_h, crop_height, center_w_start, center_w_start + quarter_w, 'bottom_center'),
                (center_h_start, center_h_start + quarter_h, 0, quarter_w, 'left_center'),
                (center_h_start, center_h_start + quarter_h, quarter_w, crop_width, 'right_center')
            ]
        elif crop_type == "all":
            crop_regions = [
                (0, quarter_h, 0, quarter_w, 'top_left'),
                (0, quarter_h, quarter_w, crop_width, 'top_right'),
                (quarter_h, crop_height, 0, quarter_w, 'bottom_left'),
                (quarter_h, crop_height, quarter_w, crop_width, 'bottom_right'),
                (0, quarter_h, center_w_start, center_w_start + quarter_w, 'top_center'),
                (quarter_h, crop_height, center_w_start, center_w_start + quarter_w, 'bottom_center'),
                (center_h_start, center_h_start + quarter_h, 0, quarter_w, 'left_center'),
                (center_h_start, center_h_start + quarter_h, quarter_w, crop_width, 'right_center')
            ]
        else:
            raise ValueError(f"Invalid crop_type: {crop_type}. Use 'quarters', 'centers', or 'all'")

        self.logger.info(f"Cropping to {crop_type} - creating {len(crop_regions)} regions")
        self.logger.info(f"Original size: ({crop_height}, {crop_width}), Crop size: ({quarter_h}, {quarter_w})")

        regions = []
        for y1, y2, x1, x2, region_name in crop_regions:
            region_dict = {}

            for key, value in data_dict.items():
                if key in exclude_keys or not isinstance(value, np.ndarray):
                    region_dict[key] = value
                    continue

                try:
                    if len(value.shape) == 2:
                        region_dict[key] = value[y1:y2, x1:x2]
                    elif len(value.shape) == 4 and value.shape[-2:] == (4, 4):
                        region_dict[key] = value[y1:y2, x1:x2, :, :]
                    else:
                        region_dict[key] = value
                        self.logger.debug(f"Kept original shape for {key}: {value.shape}")
                except Exception as e:
                    self.logger.error(f"Failed to crop {key}: {e}")
                    region_dict[key] = value

            # Add metadata
            region_dict['crop_region'] = region_name
            region_dict['crop_coordinates'] = (y1, y2, x1, x2)
            region_dict['crop_type'] = crop_type
            region_dict['original_size'] = (crop_height, crop_width)
            region_dict['crop_size'] = (quarter_h, quarter_w)
            regions.append(region_dict)

        self.logger.info(f"Created {len(regions)} cropped regions, all size ({quarter_h}, {quarter_w})")
        return regions

    def crop_to_quarters(self, data_dict: Dict[str, np.ndarray],
                         exclude_keys: List[str] = None) -> List[Dict[str, np.ndarray]]:
        """Legacy method - crop all 2D/4D arrays into 4 corner quarters."""
        return self.crop_to_regions(data_dict, crop_type="quarters", exclude_keys=exclude_keys)

    def augment_sample(self, data_dict: Dict[str, np.ndarray],
                       augmentation_params: Dict) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Apply multiple augmentations to a sample."""
        augmented_dict = data_dict.copy()

        # Extract parameters
        rotation_angle = augmentation_params.get('rotation_angle')
        flip_orientation = augmentation_params.get('flip_orientation')
        noise_level = augmentation_params.get('noise_level')

        # Handle different cropping options
        crop_to_quarters = augmentation_params.get('crop_to_quarters', False)
        crop_type = augmentation_params.get('crop_type', 'quarters')

        mueller_keys = augmentation_params.get('mueller_keys')
        regular_keys = augmentation_params.get('regular_keys')

        # Apply rotation if specified
        if rotation_angle is not None and rotation_angle != 0:
            augmented_dict = self.rotate_data(augmented_dict, rotation_angle, mueller_keys, regular_keys)

        # Apply flip if specified
        if flip_orientation is not None:
            augmented_dict = self.flip_data(augmented_dict, flip_orientation, mueller_keys, regular_keys)

        # Apply noise if specified
        if noise_level is not None:
            augmented_dict = self.add_gaussian_noise(augmented_dict, noise_level)

        # Apply cropping if specified
        if crop_to_quarters:
            return self.crop_to_regions(augmented_dict, crop_type=crop_type)
        else:
            return augmented_dict


def augment_dataset_directory(source_directory: Union[str, Path],
                              output_directory: Union[str, Path],
                              augmentation_combinations: List[Dict] = None,
                              log_level: str = "INFO") -> Dict[str, int]:
    """
    Augment all NPZ files in a directory with multiple augmentation combinations.

    FIXED: Now discovers and preserves all modality types (SAMMM, TRIMM, etc.)
    instead of only processing SAMMM.npz files.

    Args:
        source_directory: Directory containing NPZ files to augment
        output_directory: Directory to save augmented files
        augmentation_combinations: List of augmentation parameter dicts (if None, use defaults)
        log_level: Logging level

    Returns:
        Dictionary with augmentation statistics
    """
    source_path = Path(source_directory)
    output_path = Path(output_directory)

    # Initialize augmenter
    augmenter = PolarimetryDataAugmenter(log_level=log_level)

    # Get augmentation combinations
    if augmentation_combinations is None:
        augmentation_combinations = augmenter.get_augmentation_combinations()

    # FIXED: Find ALL NPZ files regardless of name
    all_files = list(source_path.rglob("*.npz"))

    # Filter out macOS hidden files (._*)
    npz_files = [f for f in all_files if not f.name.startswith('._')]

    if not npz_files:
        augmenter.logger.warning(f"No NPZ files found in {source_path}")
        return {'files_processed': 0, 'augmentations_created': 0, 'errors': 0}

    augmenter.logger.info(
        f"Found {len(npz_files)} NPZ files to augment (filtered {len(all_files) - len(npz_files)} hidden files)")

    # Log modality types found
    modalities = set([f.stem for f in npz_files])
    augmenter.logger.info(f"Modality types found: {sorted(modalities)}")
    augmenter.logger.info(f"Will create {len(augmentation_combinations)} variants each")

    stats = {'files_processed': 0, 'augmentations_created': 0, 'errors': 0}

    for npz_file in npz_files:
        try:
            # Get relative path for output structure
            rel_path = npz_file.relative_to(source_path)
            sample_folder = rel_path.parent
            modality_name = npz_file.stem  # FIXED: Preserve original modality name

            augmenter.logger.info(f"Processing: {sample_folder}/{modality_name}.npz")

            for aug_params in augmentation_combinations:
                try:
                    # Create output folder
                    aug_name = aug_params['name']
                    output_folder = output_path / f"{sample_folder}_{aug_name}"
                    output_folder.mkdir(parents=True, exist_ok=True)

                    # Apply augmentation
                    augmented_data = augmenter.load_and_augment_npz(npz_file, aug_params)

                    # FIXED: Save with original modality name, not hardcoded SAMMM
                    output_file = output_folder / f"{modality_name}.npz"
                    augmenter.save_augmented_data(augmented_data, output_file)

                    stats['augmentations_created'] += 1

                except Exception as e:
                    augmenter.logger.error(
                        f"Failed to create {aug_params['name']} variant for {sample_folder}/{modality_name}: {e}")
                    stats['errors'] += 1

            stats['files_processed'] += 1

        except Exception as e:
            augmenter.logger.error(f"Failed to process {npz_file}: {e}")
            stats['errors'] += 1

    augmenter.logger.info(f"Augmentation complete: {stats}")
    return stats


# Convenience functions for common operations
def quick_rotate(npz_path: Union[str, Path], angle: float, output_path: Union[str, Path] = None):
    """Quickly rotate a single NPZ file."""
    augmenter = PolarimetryDataAugmenter()

    npz_path = Path(npz_path)
    modality_name = npz_path.stem  # FIXED: Preserve modality name

    if output_path is None:
        output_path = npz_path.parent / f"rotated_{angle}deg_{modality_name}.npz"

    augmented_data = augmenter.load_and_augment_npz(
        npz_path, {'rotation_angle': angle, 'flip_orientation': None}
    )

    augmenter.save_augmented_data(augmented_data, output_path)
    return output_path


def quick_flip(npz_path: Union[str, Path], orientation: int, output_path: Union[str, Path] = None):
    """Quickly flip a single NPZ file."""
    augmenter = PolarimetryDataAugmenter()

    npz_path = Path(npz_path)
    modality_name = npz_path.stem  # FIXED: Preserve modality name

    orientation_names = {0: 'hflip', 1: 'vflip', 2: 'bothflip'}
    if output_path is None:
        output_path = npz_path.parent / f"{orientation_names[orientation]}_{modality_name}.npz"

    augmented_data = augmenter.load_and_augment_npz(
        npz_path, {'rotation_angle': None, 'flip_orientation': orientation}
    )

    augmenter.save_augmented_data(augmented_data, output_path)
    return output_path
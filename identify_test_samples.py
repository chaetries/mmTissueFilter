"""
Script to identify which samples will be used for training, validation, and testing
in the m11_transfer_learning notebook.

Run this BEFORE training to see the data split.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict

# === CONFIGURATION - MATCH YOUR NOTEBOOK SETTINGS ===
DATA_DIR = Path("E:/MPL_Data/mmNoTissueFilter/TRIMMM")  # UPDATE THIS
NESTED_STRUCTURE = True  # Set to False for flat structure
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def discover_samples_nested(data_dir: Path) -> List[Dict]:
    """Discover samples in nested directory structure (Day folders)"""
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    samples = []
    parent_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    for parent_dir in parent_dirs:
        sample_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        sample_groups = {}
        for sample_dir in sample_dirs:
            name = sample_dir.name
            base_name = name
            for suffix in ['_original', '_rot90', '_rot180', '_rot270', '_flip_h', '_flip_v']:
                base_name = base_name.replace(suffix, '')

            unique_name = f"{parent_dir.name}_{base_name}"
            if unique_name not in sample_groups:
                sample_groups[unique_name] = []
            sample_groups[unique_name].append(sample_dir)

        for unique_name, dirs in sample_groups.items():
            selected_dir = next((d for d in dirs if '_original' in d.name), dirs[0])
            npz_files = [f for f in selected_dir.glob("*.npz") if not f.name.startswith('.')]

            if npz_files:
                samples.append({
                    'sample_name': unique_name,
                    'parent_folder': parent_dir.name,
                    'sample_dir': selected_dir,
                    'npz_path': npz_files[0]
                })

    return samples


def discover_samples_flat(data_dir: Path) -> List[Dict]:
    """Discover samples in flat directory structure (original version)"""
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    sample_groups = {}

    for sample_dir in all_dirs:
        name = sample_dir.name
        base_name = name
        for suffix in ['_original', '_rot90', '_rot180', '_rot270', '_flip_h', '_flip_v']:
            base_name = base_name.replace(suffix, '')

        if base_name not in sample_groups:
            sample_groups[base_name] = []
        sample_groups[base_name].append(sample_dir)

    original_samples = []
    for base_name, dirs in sample_groups.items():
        original_dir = next((d for d in dirs if '_original' in d.name), dirs[0])
        npz_files = [f for f in original_dir.glob("*.npz") if not f.name.startswith('.')]
        if npz_files:
            original_samples.append({
                'sample_name': base_name,
                'sample_dir': original_dir,
                'npz_path': npz_files[0]
            })

    return original_samples


def main():
    print("=" * 80)
    print("SAMPLE SPLIT IDENTIFICATION")
    print("=" * 80)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Structure Mode: {'NESTED' if NESTED_STRUCTURE else 'FLAT'}")
    print(f"Split Ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"\n{'=' * 80}\n")

    # Set random seed (same as notebook)
    np.random.seed(RANDOM_SEED)

    # Discover samples
    try:
        if NESTED_STRUCTURE:
            samples = discover_samples_nested(DATA_DIR)
        else:
            samples = discover_samples_flat(DATA_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease update DATA_DIR in this script to match your data location.")
        return

    if not samples:
        print("ERROR: No valid samples found!")
        print(f"\nChecked directory: {DATA_DIR}")
        print("Make sure your NPZ files are in subdirectories of DATA_DIR")
        return

    print(f"Found {len(samples)} total samples\n")

    # Split data (same as notebook)
    train_samples, temp = train_test_split(
        samples,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED
    )
    val_samples, test_samples = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED
    )

    # Display results
    print(f"{'=' * 80}")
    print(f"TRAINING SET ({len(train_samples)} samples)")
    print(f"{'=' * 80}")
    for i, sample in enumerate(sorted(train_samples, key=lambda x: x['sample_name']), 1):
        print(f"{i:3d}. {sample['sample_name']:40s} | {sample['npz_path']}")

    print(f"\n{'=' * 80}")
    print(f"VALIDATION SET ({len(val_samples)} samples)")
    print(f"{'=' * 80}")
    for i, sample in enumerate(sorted(val_samples, key=lambda x: x['sample_name']), 1):
        print(f"{i:3d}. {sample['sample_name']:40s} | {sample['npz_path']}")

    print(f"\n{'=' * 80}")
    print(f"TEST SET ({len(test_samples)} samples) - ISOLATED FOR FINAL VALIDATION")
    print(f"{'=' * 80}")
    for i, sample in enumerate(sorted(test_samples, key=lambda x: x['sample_name']), 1):
        print(f"{i:3d}. {sample['sample_name']:40s} | {sample['npz_path']}")

    # Save to file
    output_file = Path("data_split_info.txt")
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA SPLIT INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data Directory: {DATA_DIR}\n")
        f.write(f"Split Ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Total Samples: {len(samples)}\n\n")

        f.write(f"TRAINING SET ({len(train_samples)} samples)\n")
        f.write("-" * 80 + "\n")
        for sample in sorted(train_samples, key=lambda x: x['sample_name']):
            f.write(f"{sample['sample_name']}\n")
            f.write(f"  Path: {sample['npz_path']}\n\n")

        f.write(f"\nVALIDATION SET ({len(val_samples)} samples)\n")
        f.write("-" * 80 + "\n")
        for sample in sorted(val_samples, key=lambda x: x['sample_name']):
            f.write(f"{sample['sample_name']}\n")
            f.write(f"  Path: {sample['npz_path']}\n\n")

        f.write(f"\nTEST SET ({len(test_samples)} samples)\n")
        f.write("-" * 80 + "\n")
        for sample in sorted(test_samples, key=lambda x: x['sample_name']):
            f.write(f"{sample['sample_name']}\n")
            f.write(f"  Path: {sample['npz_path']}\n\n")

    print(f"\n{'=' * 80}")
    print(f" Split information saved to: {output_file.absolute()}")
    print(f"{'=' * 80}\n")

    print("IMPORTANT NOTES:")
    print("- The TEST SET samples are NEVER seen during training")
    print("- These are used ONLY for final evaluation after training completes")
    print("- The split is deterministic (same RANDOM_SEED = same split)")
    print("- Make sure your notebook uses the SAME RANDOM_SEED value")


if __name__ == "__main__":
    main()

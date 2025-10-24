"""
Data management for tissue annotation
"""

import numpy as np
from datetime import datetime


class DataManager:
    """Manages NPZ file data and mask operations"""

    def __init__(self):
        self.filepath = None
        self.data_dict = None
        self.param_names = []
        self.current_param = None
        self.current_mask_type = 'tissue'
        self.masks = {}  # Dictionary to store multiple masks by type
        self.mask_history = {}  # Dictionary to store history for each mask type
        self.available_mask_types = ['tissue']  # Start with 'tissue' as default

    def load_file(self, filepath):
        """Load NPZ file"""
        try:
            self.filepath = filepath
            self.data_dict = np.load(filepath, allow_pickle=True)

            available_params = ['M11s', 'linrs', 'Morientations',
                                'Mdepols', 'Mdiattenuations']
            self.param_names = [p for p in available_params
                                if p in self.data_dict]

            if not self.param_names:
                return False, "No recognized Mueller parameters found in file"

            first_param = self.data_dict[self.param_names[0]]
            height, width = first_param.shape

            # Load all existing masks from file
            self.masks = {}
            self.available_mask_types = ['tissue']

            for key in self.data_dict.files:
                if key.endswith('_mask') and key != 'tissue_mask':
                    mask_type = key.replace('_mask', '')
                    self.masks[mask_type] = self.data_dict[key].astype(bool)
                    if mask_type not in self.available_mask_types:
                        self.available_mask_types.append(mask_type)

            if 'tissue_mask' in self.data_dict:
                self.masks['tissue'] = self.data_dict['tissue_mask'].astype(bool)
                message = "File loaded successfully. Existing masks found."
            else:
                self.masks['tissue'] = np.zeros((height, width), dtype=bool)
                message = "File loaded successfully."

            self.current_mask_type = 'tissue'
            self.mask_history = {mask_type: [] for mask_type in self.available_mask_types}

            return True, message

        except Exception as e:
            return False, f"Error loading file: {str(e)}"

    def add_mask_type(self, mask_type):
        """Add a new mask type"""
        if mask_type not in self.available_mask_types:
            if self.masks:
                # Use shape from existing mask
                shape = next(iter(self.masks.values())).shape
                self.masks[mask_type] = np.zeros(shape, dtype=bool)
            self.available_mask_types.append(mask_type)
            self.mask_history[mask_type] = []
            return True
        return False

    def get_current_mask(self):
        """Get the current mask"""
        return self.masks.get(self.current_mask_type)

    def set_current_mask_type(self, mask_type):
        """Switch to a different mask type"""
        if mask_type in self.available_mask_types:
            self.current_mask_type = mask_type
            return True
        return False

    def save_mask(self):
        """Save all masks to NPZ file"""
        if not self.masks or self.filepath is None:
            return False, "No data to save"

        try:
            import os

            # Load all existing data
            data_dict = {}
            for key in self.data_dict.files:
                data_dict[key] = self.data_dict[key]

            # Update with all mask data
            for mask_type, mask in self.masks.items():
                mask_key = f"{mask_type}_mask"
                data_dict[mask_key] = mask
                data_dict[f"{mask_type}_mask_date"] = np.array(datetime.now().isoformat())
                data_dict[f"{mask_type}_mask_pixels"] = np.array(np.sum(mask))

            # Create temp file in the SAME directory as the original file
            directory = os.path.dirname(self.filepath)
            filename = os.path.basename(self.filepath)
            temp_path = os.path.join(directory, f'.tmp_{filename}')

            try:
                # Save to temporary file
                np.savez(temp_path, **data_dict)

                # Replace original file with temp file
                if os.path.exists(self.filepath):
                    os.remove(self.filepath)
                os.rename(temp_path, self.filepath)

            except Exception as e:
                # Clean up temp file if something goes wrong
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

            # Generate message for all masks
            message_lines = ["Saved all masks:"]
            for mask_type, mask in self.masks.items():
                n_pixels = np.sum(mask)
                pct = 100 * n_pixels / mask.size
                message_lines.append(
                    f"  {mask_type}: {n_pixels} pixels ({pct:.1f}%)"
                )

            message = f"Saved to {self.filepath}\n" + "\n".join(message_lines)
            return True, message

        except Exception as e:
            return False, f"Error saving masks: {str(e)}"

    def save_mask_history(self):
        """Save current mask state for undo"""
        mask = self.get_current_mask()
        if mask is not None:
            if self.current_mask_type not in self.mask_history:
                self.mask_history[self.current_mask_type] = []
            self.mask_history[self.current_mask_type].append(mask.copy())
            if len(self.mask_history[self.current_mask_type]) > 20:
                self.mask_history[self.current_mask_type].pop(0)

    def undo(self):
        """Undo last mask operation"""
        if (self.current_mask_type in self.mask_history and
            self.mask_history[self.current_mask_type]):
            self.masks[self.current_mask_type] = self.mask_history[self.current_mask_type].pop()
            return True
        else:
            return False

    def clear_mask(self):
        """Clear all annotations for current mask type"""
        mask = self.get_current_mask()
        if mask is not None:
            self.save_mask_history()
            self.masks[self.current_mask_type] = np.zeros_like(mask, dtype=bool)

    def get_statistics(self):
        """Get statistics string"""
        mask = self.get_current_mask()
        if mask is None:
            return "No data loaded"

        n_tissue = np.sum(mask)
        total = mask.size
        pct = 100 * n_tissue / total

        stats = (
            f"Mask Type: {self.current_mask_type}\n"
            f"Parameter: {self.current_param}\n"
            f"Image size: {mask.shape[1]} x {mask.shape[0]}\n"
            f"Tissue pixels: {n_tissue}\n"
            f"Coverage: {pct:.2f}%"
        )

        return stats
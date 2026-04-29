import os
import numpy as np
import nibabel
from scipy.ndimage import zoom, gaussian_filter


class Dataloader:
    def __init__(self, data_path, pt_ids, isTraining=True):
        self.data_path = "./" + data_path
        self.pt_ids = pt_ids
        self.isTraining = isTraining
        self.crop_size = (128, 128, 128)

    def load(self):
        dataset = []
        for pt_id in self.pt_ids:
            img_path = os.path.join(
                self.data_path, f"BraTS2021_{pt_id}", f"BraTS2021_{pt_id}.nii.gz"
            )
            lbl_path = os.path.join(
                self.data_path, f"BraTS2021_{pt_id}", f"BraTS2021_{pt_id}_seg.nii.gz"
            )

            image = nibabel.load(img_path).get_fdata().astype(np.float32)
            label = nibabel.load(lbl_path).get_fdata().astype(np.uint8)

            # APPLY AUGMENTATIONS
            if self.isTraining:
                # 1. Biased Crop
                # image, label = self._biased_crop(image, label)
                # 2. Zoom
                image, label = self._apply_zoom(image, label)
                # 3. Flips
                image, label = self._apply_flips(image, label)
                # 4. Gaussian Noise
                image = self._apply_gaussian_noise(image)
                # 5. Gaussian Blur
                image = self._apply_gaussian_blur(image)
                # 6. Brightness
                image = self._apply_brightness(image)
                # 7. Contrast
                image = self._apply_contrast(image)
            else:
                # If testing/validating, apply standard center crop to match dimensions
                image, label = self._center_crop(image, label)

            dataset.append({"x": image, "y": label})
        return dataset

    def _biased_crop(self, image, label):
        """1. Biased crop with 0.4 prob to guarantee foreground."""
        _, d, h, w = image.shape
        cd, ch, cw = self.crop_size

        if np.random.rand() < 0.4 and np.any(label > 0):
            fg_indices = np.argwhere(label > 0)
            center_voxel = fg_indices[np.random.choice(len(fg_indices))]

            # Handle label shapes: (D, H, W) vs (1, D, H, W)
            if len(center_voxel) == 4:
                _, z, y, x = center_voxel
            else:
                z, y, x = center_voxel

            z_start = max(0, min(z - cd // 2, d - cd))
            y_start = max(0, min(y - ch // 2, h - ch))
            x_start = max(0, min(x - cw // 2, w - cw))
        else:
            z_start = np.random.randint(0, d - cd + 1)
            y_start = np.random.randint(0, h - ch + 1)
            x_start = np.random.randint(0, w - cw + 1)

        cropped_img = image[
            :, z_start : z_start + cd, y_start : y_start + ch, x_start : x_start + cw
        ]

        if label.ndim == 4:
            cropped_lbl = label[
                :,
                z_start : z_start + cd,
                y_start : y_start + ch,
                x_start : x_start + cw,
            ]
        else:
            cropped_lbl = label[
                z_start : z_start + cd, y_start : y_start + ch, x_start : x_start + cw
            ]

        return cropped_img, cropped_lbl

    def _center_crop(self, image, label):
        """Helper for validation consistency."""
        _, d, h, w = image.shape
        cd, ch, cw = self.crop_size

        z_start = (d - cd) // 2
        y_start = (h - ch) // 2
        x_start = (w - cw) // 2

        cropped_img = image[
            :, z_start : z_start + cd, y_start : y_start + ch, x_start : x_start + cw
        ]
        if label.ndim == 4:
            cropped_lbl = label[
                :,
                z_start : z_start + cd,
                y_start : y_start + ch,
                x_start : x_start + cw,
            ]
        else:
            cropped_lbl = label[
                z_start : z_start + cd, y_start : y_start + ch, x_start : x_start + cw
            ]

        return cropped_img, cropped_lbl

    def _apply_zoom(self, image, label):
        """2. Zoom (0.15 prob, 1.0 to 1.4)."""
        if np.random.rand() < 0.15:
            zoom_factor = np.random.uniform(1.0, 1.4)
            # Do not zoom the channel dimension
            zoom_tuple = (1.0, zoom_factor, zoom_factor, zoom_factor)

            # Cubic interpolation (order=3) for image, Nearest Neighbor (order=0) for label
            image = zoom(image, zoom_tuple, order=3, mode="nearest")

            if label.ndim == 4:
                label = zoom(label, zoom_tuple, order=0, mode="nearest")
            else:
                label = zoom(
                    label,
                    (zoom_factor, zoom_factor, zoom_factor),
                    order=0,
                    mode="nearest",
                )

            # Crop back down to original size after zooming
            image, label = self._center_crop(image, label)

        return image, label

    def _apply_flips(self, image, label):
        """3. Flips (0.5 prob per axis)."""
        # Axes for D, H, W (skipping Channel at index 0)
        axes = (
            [1, 2, 3] if label.ndim == 4 else [1, 2, 3]
        )  # assuming label spatial dims map correctly

        for axis_idx, axis in enumerate([1, 2, 3]):
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=axis)
                lbl_axis = axis if label.ndim == 4 else axis - 1
                label = np.flip(label, axis=lbl_axis)

        # Ensure memory continuity after flipping
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def _apply_gaussian_noise(self, image):
        """4. Gaussian Noise (0.15 prob, stddev 0 to 0.33)."""
        if np.random.rand() < 0.15:
            std = np.random.uniform(0.0, 0.33)
            noise = np.random.normal(0, std, image.shape)
            image = image + noise
        return image.astype(np.float32)

    def _apply_gaussian_blur(self, image):
        """5. Gaussian Blur (0.15 prob, sigma 0.5 to 1.5)."""
        if np.random.rand() < 0.15:
            sigma = np.random.uniform(0.5, 1.5)
            # Apply blur to spatial dimensions, not the channel dimension
            image = gaussian_filter(image, sigma=(0, sigma, sigma, sigma))
        return image

    def _apply_brightness(self, image):
        """6. Brightness (0.15 prob, factor 0.7 to 1.3)."""
        if np.random.rand() < 0.15:
            factor = np.random.uniform(0.7, 1.3)
            image = image * factor
        return image

    def _apply_contrast(self, image):
        """7. Contrast (0.15 prob, factor 0.65 to 1.5)."""
        if np.random.rand() < 0.15:
            factor = np.random.uniform(0.65, 1.5)
            img_min, img_max = image.min(), image.max()
            image = image * factor
            # Clip back to original min/max values
            image = np.clip(image, img_min, img_max)
        return image

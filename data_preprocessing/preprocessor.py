import os  # For interacting with the file system
from joblib import Parallel, delayed  # For parallel processing of tasks
import nibabel  # For loading medical imaging data
import numpy as np  # For numerical operations
import itertools  # For chaining lists together
import monai  # For medical imaging transformations and utilities
import skimage  # For image processing tasks

from utils.utils import get_patient_ids


class Preprocessor:
    def __init__(self, pt_ids=None):
        self.data_path = "./.raw/BraTS2021_Training_Data"
        self.output_path = "./.processed"
        self.pt_ids = pt_ids
        self.modalities = ["t1", "t1ce", "t2", "flair"]
        self.target_spacing = None
        self.ct_min = {"t1": 0, "t1ce": 0, "t2": 0, "flair": 0}
        self.ct_max = {"t1": 0, "t1ce": 0, "t2": 0, "flair": 0}
        self.ct_mean = {"t1": 0, "t1ce": 0, "t2": 0, "flair": 0}
        self.ct_std = {"t1": 0, "t1ce": 0, "t2": 0, "flair": 0}

    def run(self):
        """Entry point for the preprocessor. This method will be called to start the preprocessing steps."""

        # Create directory for output of processed files if it does not exist.
        if not os.path.exists(".processed"):
            os.makedirs(".processed")

        # Get the list of patients ids from the data directory.
        self.pt_ids = get_patient_ids(self.data_path)

        # Collect intensity statistics from the medical images to determine normalization parameters.
        self.collect_intensities()

        self.run_parallel(self.preprocess_images)

    def collect_intensities(self):
        intensities_dicts = self.run_parallel(self.get_intensities)
        intensities = {}
        for mod in self.modalities:
            # Chain the intensity values from all patients for the current modality into a single list.
            intensities[mod] = list(
                itertools.chain(*[d[mod] for d in intensities_dicts])
            )
            # Calculate the 0.5th and 99.5th percentiles to determine the minimum and maximum intensity values for normalization,
            # and calculate the mean and standard deviation for further normalization steps.
            self.ct_min[mod] = np.percentile(intensities[mod], [0.5, 99.5])[0]
            self.ct_max[mod] = np.percentile(intensities[mod], [0.5, 99.5])[1]
            self.ct_mean[mod] = np.mean(intensities[mod])
            self.ct_std[mod] = np.std(intensities[mod])

    def get_intensities(self, pt_id):
        intensity = {}
        # Loop through each modality for the patient and load the corresponding image and label to extract intensity values.
        for mod in self.modalities:
            image = (
                nibabel.load(
                    os.path.join(
                        self.data_path,
                        f"BraTS2021_{pt_id}",
                        f"BraTS2021_{pt_id}_{mod}.nii.gz",
                    )
                )
                .get_fdata()
                .astype(np.float32)
            )
            label = (
                nibabel.load(
                    os.path.join(
                        self.data_path,
                        f"BraTS2021_{pt_id}",
                        f"BraTS2021_{pt_id}_seg.nii.gz",
                    )
                )
                .get_fdata()
                .astype(np.uint8)
            )
            # Extract brain region
            foreground_area = np.where(label > 0)
            # Store the intensity values of the brain region for the current modality in the intensity dictionary as a list.
            intensity[mod] = image[foreground_area].tolist()
        return intensity

    def preprocess_images(self, pt_id):

        os.makedirs(os.path.join(self.output_path, f"BraTS2021_{pt_id}"))

        images = {}
        for mod in self.modalities:
            # Load the image for the current modality and patient
            image = nibabel.load(
                os.path.join(
                    self.data_path,
                    f"BraTS2021_{pt_id}",
                    f"BraTS2021_{pt_id}_{mod}.nii.gz",
                )
            )
            images[mod] = image
            
        img_header = images["t1"].header
        img_affine = images["t1"].affine
        img_data = image.get_fdata().astype(np.float32)

            normalized_img_data = self.normalize(img_data, mod)

            normalized_image = nibabel.Nifti1Image(
                normalized_img_data, img_affine, header=img_header
            )

            nibabel.save(
                normalized_image,
                os.path.join(
                    self.output_path,
                    f"BraTS2021_{pt_id}",
                    f"BraTS2021_{pt_id}_{mod}.nii.gz",
                ),
            )

        # Load the label image for the patient
        label = nibabel.load(
            os.path.join(
                self.data_path,
                f"BraTS2021_{pt_id}",
                f"BraTS2021_{pt_id}_seg.nii.gz",
            )
        )
        lbl_header = label.header
        lbl_affine = label.affine
        lbl_data = label.get_fdata().astype(np.uint8)

        # This ensures classes are [0, 1, 2, 3] instead of [0, 1, 2, 4]
        lbl_data[lbl_data == 4] = 3

        ohc_label = nibabel.Nifti1Image(lbl_data, lbl_affine, header=lbl_header)
        nibabel.save(
            ohc_label,
            os.path.join(
                self.output_path, f"BraTS2021_{pt_id}", f"BraTS2021_{pt_id}_seg.nii.gz"
            ),
        )

    def normalize(self, image, mod):
        image = np.clip(image, self.ct_min[mod], self.ct_max[mod])
        image = (image - self.ct_mean[mod]) / self.ct_std[mod]
        return image

    # Runs a function in parallel across multiple patient IDs.
    def run_parallel(self, func):
        return Parallel(n_jobs=os.cpu_count())(
            delayed(func)(pt_id) for pt_id in self.pt_ids
        )

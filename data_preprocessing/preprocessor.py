import os  # For interacting with the file system
from joblib import Parallel, delayed  # For parallel processing of tasks
import nibabel  # For loading medical imaging data
import numpy as np  # For numerical operations
import itertools  # For chaining lists together
import monai  # For medical imaging transformations and utilities
import skimage  # For image processing tasks

from utils.utils import get_patient_ids


class Preprocessor:
    def __init__(self):
        self.data_path = "./.raw/BraTS2021_Training_Data"
        self.pt_ids = []
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

        """ This step is not required for BraTS2021 as all images are already at the same spacing [1.0, 1.0, 1.0],
            but it is included for completeness and to handle potential future datasets with varying spacings.
        """
        # Collect spacing information from the medical images to determine the target spacing for resampling.
        self.collect_spacings()
        # Collect intensity statistics from the medical images to determine normalization parameters.
        self.collect_intensities()

        self.run_parallel(self.preprocess_images())

    def preprocess_images(self, pt_id):
        # Load the label image for the patient
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
        for mod in self.modalities:
            # Load the image for the current modality and patient, and convert it to a float32 numpy array for processing.
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

            ## Extract the brain region by cropping the image and label to the bounding box of the brain.
            # Calculate the bounding box of the brain region in the image using MONAI's utility function, which returns the start and end coordinates of the bounding box.
            bounding_box = monai.transforms.utils.get_bounding_box(image)
            # Crop the image and label to the bounding box of the brain region.
            image = monai.transforms.SpacialCrop(
                roi_start=bounding_box[0], roi_end=bounding_box[1]
            )(image)
            # Store metadata about the original shape of the image, the cropped shape, and the bounding box coordinates.
            original_shape = image.shape[1:]
            cropped_shape = image.shape[1:]
            image_metadata = np.vstack([bounding_box, original_shape, cropped_shape])
            label = monai.transforms.SpacialCrop(
                roi_start=bounding_box[0], roi_end=bounding_box[1]
            )(label)

            # Resize the image and label to the target spacing determined from the collected spacings.
            image, label = self.resize(image, label)

            image = self.normalize(image, mod)

    def resize(self, image, label):
        image_spacing = self.get_image_spacing(image)
        if image_spacing == self.target_spacing:
            return image, label

        spacing_ratio = np.array(image_spacing) / np.array(self.target_spacing)
        new_shape = (
            np.round(np.array(image.shape[1:]) * spacing_ratio).astype(int).tolist()
        )
        # RESIZE IMAGE
        resized_image = skimage.transform.resize(
            image, new_shape, order=3, mode="edge", clip=True, anti_aliasing=False
        )

        # RESIZE LABEL
        resized_label = np.zeros(new_shape, dtype=np.uint8)
        n_classes = np.max(label) + 1
        for class_id in range(n_classes):
            class_mask = (label == class_id).astype(np.float32)
            resized_class_mask = skimage.transform.resize(
                class_mask,
                new_shape,
                order=0,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            resized_label[resized_class_mask > 0.5] = class_id

        return resized_image, resized_label

    def normalize(self, image, mod):
        image = np.clip(image, self.ct_min[mod], self.ct_max[mod])
        image = (image - self.ct_mean[mod]) / self.ct_std[mod]
        return image

    # Collects spacing information from the medical images of all patients.
    def collect_spacings(self):
        # Get spacing information for all patients in parallel.
        spacing = self.run_parallel(self.get_spacing)
        spacing = np.array(spacing).reshape(-1, 3)
        # Calculate the median spacing across all patients.
        target_spacing = np.median(spacing, axis=0)
        # Store the target spacing as a list for later use.
        self.target_spacing = list(target_spacing)

    # Gets spacing information from the T1-weighted image of a patient.
    def get_spacing(self, pt_id):
        patient_spacings = []
        # Loop through each modality for the patient and load the corresponding image to extract spacing information.
        for mod in self.modalities:
            image = nibabel.load(
                os.path.join(
                    self.data_path,
                    f"BraTS2021_{pt_id}",
                    f"BraTS2021_{pt_id}_{mod}.nii.gz",
                )
            )
            # Extract spacing and append to the patient's list
            patient_spacings.append(self.get_image_spacing(image))

        return patient_spacings  # Returns a list of 4 spacings

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

    # Runs a function in parallel across multiple patient IDs.
    def run_parallel(self, func):
        return Parallel(n_jobs=os.cpu_count())(
            delayed(func)(pt_id) for pt_id in self.pt_ids
        )

    @staticmethod
    def get_image_spacing(image):
        return image.header["pixdim"][1:4].tolist()[::-1]

import os  # For interacting with the file system
import nibabel  # For loading medical imaging data


class Preprocessor:
    # Entry point for the preprocessor. This method will be called to start the preprocessing steps.
    def run(self):
        # Create directory for output of processed files if it does not exist.
        if not os.path.exists(".processed"):
            os.makedirs(".processed")

    # Method to load the spacing information from a medical image file.
    def get_spacing(self, pair):
        image = nibabel.load(os.path.join(self.data_path, pair["image"]))
        spacing = self.load_spacing(image)
        return spacing

    def collect_spacings(self):
        spacing = self.run_parallel(self.get_spacing, "training")
        spacing = np.array(spacing)
        target_spacing = np.median(spacing, axis=0)
        if max(target_spacing) / min(target_spacing) >= 3:
            lowres_axis = np.argmin(target_spacing)
            target_spacing[lowres_axis] = np.percentile(spacing[:, lowres_axis], 10)
        self.target_spacing = list(target_spacing)

import torch  # Import PyTorch library
import time  # Import time library for measuring execution time
import os  # Import os library for file and directory operations
import sklearn  # Import scikit-learn for data splitting

from data_preprocessing.preprocessor import Preprocessor
from data_augmentation.data_loader import BraTSDataset
from utils.utils import get_patient_ids


def main():
    initial_setup()

    raw_path = ".raw"
    data_path = ".processed"
    pt_ids = get_patient_ids(raw_path)

    # Create directory for output of processed files if it does not exist.
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_dirs = [
        d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))
    ]
    out_dirs = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]
    if len(data_dirs) != len(out_dirs):
        start = time.time()
        Preprocessor(raw_path, data_path, pt_ids).run()
        end = time.time()
        # 5 minutes
        print(f"Preprocessing time: {end - start} seconds")
    else:
        print("Preprocessed data already exists. Skipping preprocessing step.")

    train_ids, val_ids = sklearn.model_selection.train_test_split(
        pt_ids, test_size=0.2, random_state=123
    )

    training_dataset = BraTSDataset(data_path, train_ids, isTraining=True)
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=2, shuffle=True
    )
    validation_dataset = BraTSDataset(data_path, val_ids, isTraining=False)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=2, shuffle=False
    )

    for batch_image, batch_label in train_loader:
        pass


def initial_setup():
    # Enable TensorFloat-32 (TF32) for faster matrix operations on compatible NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Make the main function run when the script is executed
if __name__ == "__main__":
    main()

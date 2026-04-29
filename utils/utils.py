import json
import os


def get_patient_ids(data_path):
    # List all patient directories in the data path.
    patients = [
        d.split("_")[-1]  # Extract the patient ID from the directory name
        for d in os.listdir(data_path)  # List all entries in the data path
        if os.path.isdir(
            os.path.join(data_path, d)
        )  # Check if the entry is a directory
    ]
    return sorted(patients)

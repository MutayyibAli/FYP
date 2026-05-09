import os
import nibabel as nib
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Required to generate plots without a display server
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# Optional: Import your model class from your model.py file
# from model import nnUNet

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload limit
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------------------------------------------------
# MODEL INITIALIZATION (Mocked for demonstration)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Replace this with your actual model initialization and weights loading
# model = nnUNet(in_channels=4, out_channels=4).to(device)
# model.load_state_dict(torch.load('best_model.pth', map_location=device))
# model.eval()


def save_nifti_slice_as_png(nifti_path, output_filename, is_mask=False):
    """Loads a 3D NIfTI file, extracts the middle axial slice, and saves it as a PNG."""
    if not nifti_path or not os.path.exists(nifti_path):
        return None

    img_data = nib.load(nifti_path).get_fdata()

    # Grab the middle slice along the Z-axis (Depth)
    z_mid = img_data.shape[2] // 2
    slice_2d = img_data[:, :, z_mid]

    # Plot and save as PNG
    plt.figure(figsize=(5, 5))
    if is_mask:
        # Use a distinct colormap for masks
        plt.imshow(slice_2d.T, cmap="viridis", origin="lower")
    else:
        # Use grayscale for standard MRI scans
        plt.imshow(slice_2d.T, cmap="gray", origin="lower")

    plt.axis("off")
    plt.tight_layout(pad=0)
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    return output_filename


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = {
            "flair": request.files.get("flair"),
            "t1": request.files.get("t1"),
            "t1ce": request.files.get("t1ce"),
            "t2": request.files.get("t2"),
            "gt": request.files.get("gt"),  # Ground truth is optional
        }

        # Save uploaded files to the uploads directory
        saved_paths = {}
        for key, file in files.items():
            if file and file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                saved_paths[key] = filepath
            else:
                saved_paths[key] = None

        # 1. Convert Modalities and GT to PNG for web display
        display_images = {}
        for key in ["flair", "t1", "t1ce", "t2"]:
            if saved_paths[key]:
                display_images[key] = save_nifti_slice_as_png(
                    saved_paths[key], f"{key}_slice.png", is_mask=False
                )

        if saved_paths["gt"]:
            display_images["gt"] = save_nifti_slice_as_png(
                saved_paths["gt"], "gt_slice.png", is_mask=True
            )

        # 2. RUN INFERENCE (Pseudo-code for PyTorch)
        # In reality, you will load the 4 modalities, stack them, pad to 160 depth, and feed to the model
        """
        if all([saved_paths['flair'], saved_paths['t1'], saved_paths['t1ce'], saved_paths['t2']]):
            with torch.no_grad():
                # stack your data into shape (1, 4, H, W, D) -> feed to model -> argmax
                # prediction_mask = model(stacked_tensor).argmax(dim=1).squeeze().cpu().numpy()
                
                # Mocking the output save process:
                # pred_nifti = nib.Nifti1Image(prediction_mask, affine=np.eye(4))
                # nib.save(pred_nifti, 'static/uploads/pred.nii.gz')
                # display_images['pred'] = save_nifti_slice_as_png('static/uploads/pred.nii.gz', "pred_slice.png", is_mask=True)
        """

        # MOCK PREDICTION FOR NOW (So the app runs without crashing before your model is linked)
        if saved_paths["flair"]:
            # Pretending the ground truth or flair is the prediction for visual sake
            display_images["pred"] = save_nifti_slice_as_png(
                saved_paths["flair"], "pred_slice.png", is_mask=True
            )

        return render_template("index.html", images=display_images)

    # For GET requests, just show the form
    return render_template("index.html", images=None)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

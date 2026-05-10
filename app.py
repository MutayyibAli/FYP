import os
import shutil
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from model import get_results

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper function to check if the file has the correct extension
def allowed_file(filename):
    return filename.lower().endswith(".nii.gz")


# ---------------------------------------------------------
# Define your analyze function here
# ---------------------------------------------------------
def analyse():

    get_results()

    return [
        url_for("static", filename="t1.png"),
        url_for("static", filename="t1ce.png"),
        url_for("static", filename="t2.png"),
        url_for("static", filename="flair.png"),
        url_for("static", filename="ground_truth.png"),
        url_for("static", filename="prediction.png"),
    ]


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    # 1. Check if all 4 required modalities are present and validate file types
    required_mods = ["t1", "t1ce", "t2", "flair"]

    # First pass: Validate everything before making any folder changes
    for mod in required_mods:
        if mod not in request.files or request.files[mod].filename == "":
            return jsonify({"error": f"Missing required modality: {mod.upper()}"}), 400

        file = request.files[mod]
        if not allowed_file(file.filename):
            return jsonify(
                {
                    "error": f"Invalid file type for {mod.upper()}. Only .nii.gz files are allowed."
                }
            ), 400

    # Validate optional ground truth if provided
    if "ground_truth" in request.files and request.files["ground_truth"].filename != "":
        gt_file = request.files["ground_truth"]
        if not allowed_file(gt_file.filename):
            return jsonify(
                {
                    "error": "Invalid file type for GROUND_TRUTH. Only .nii.gz files are allowed."
                }
            ), 400

    # 2. Clear the uploads folder before saving
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

    # 3. Save the required files safely
    for mod in required_mods:
        file = request.files[mod]
        filename = secure_filename(f"{mod}.nii.gz")
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    # 4. Save optional ground truth safely
    if "ground_truth" in request.files and request.files["ground_truth"].filename != "":
        gt_file = request.files["ground_truth"]
        filename = secure_filename(f"ground_truth.nii.gz")
        gt_file.save(os.path.join(UPLOAD_FOLDER, filename))

    return jsonify({"message": "Files uploaded successfully!"}), 200


@app.route("/analyze", methods=["POST"])
def run_analysis():
    try:
        # Call the custom function
        result_image_urls = analyse()
        return jsonify({"image_urls": result_image_urls}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

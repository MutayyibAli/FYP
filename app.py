import os
import shutil
import time
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Define your analyze function here
# ---------------------------------------------------------
def analyse():
    """
    Your custom analysis function.
    Currently, it simulates a long-running process and returns a path to an image.
    """
    time.sleep(3)  # Simulating processing time

    # Return the path to the resulting image.
    # For now, it points to a placeholder image in the static folder.
    return url_for("static", filename="result.png")


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    # 1. Check if all 4 required modalities are present
    required_mods = ["flair", "t1", "t1ce", "t2"]
    for mod in required_mods:
        if mod not in request.files or request.files[mod].filename == "":
            return jsonify({"error": f"Missing required modality: {mod.upper()}"}), 400

    # 2. Clear the uploads folder before saving
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

    # 3. Save the required files
    for mod in required_mods:
        file = request.files[mod]
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    # 4. Save optional ground truth if provided
    if "ground_truth" in request.files and request.files["ground_truth"].filename != "":
        gt_file = request.files["ground_truth"]
        gt_file.save(os.path.join(UPLOAD_FOLDER, gt_file.filename))

    return jsonify({"message": "Files uploaded successfully!"}), 200


@app.route("/analyze", methods=["POST"])
def run_analysis():
    try:
        # Call the custom function
        result_image_url = analyse()
        return jsonify({"image_url": result_image_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

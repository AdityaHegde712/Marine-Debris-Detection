from glob import glob
import os
import base64
import json
from io import BytesIO

import rasterio
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np
from flask import send_file
from planet_functions import (
    allowed_file,
    merge_overlapping_boxes,
    process_tif,
    save_image,
    make_geojson
)
from sentinel_functions import (
    process_tif_file,
    make_uint8
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------------------------------------------------------------------------------
# FOR PLANETSCOPE DATA
# ----------------------------------------------------------------------------------------------


app.config['UPLOAD_FOLDER'] = 'uploads/'  # Define an upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['JSON_FOLDER'] = 'json/'  # Define a JSON directory
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)
app.config['PROCESSED_FOLDER'] = 'processed/'  # Define a processed directory
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
# Load the YOLO model
model_path = r"ai_models/PLANET.pt"
model = YOLO(model_path)


def make_json_path(x: str, mode: str = 'PLANET') -> str:
    folder = app.config['JSON_FOLDER'] if mode == 'PLANET' else app.config['SENTINEL_JSON_FOLDER']
    return os.path.join(folder, f"{os.path.splitext(os.path.basename(x))[0]}.geojson")


def make_image_path(x: str, mode: str = 'PLANET') -> str:
    folder = app.config['UPLOAD_FOLDER'] if mode == 'PLANET' else app.config['SENTINEL_UPLOAD_FOLDER']
    return os.path.join(folder, os.path.basename(x))


def save_rasterio_image(image: np.ndarray, filename: str, profile: dict):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = np.moveaxis(image, 2, 0)
    processed_path = os.path.join(app.config['SENTINEL_PROCESSED_FOLDER'], filename)
    with rasterio.open(processed_path, mode='w', **profile) as dst:
        dst.write(image)


def detect_marine_debris(image_path: str):
    # Run inference
    results = model(image_path, iou=0.8, verbose=False)
    result = results[0]  # Assume a single result

    # Open the original image
    bounds = None
    crs = None
    transform = None
    if image_path.lower().endswith(".jpg"):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        img, bounds, crs, transform = process_tif(image_path)  # Replace "_, _" with "crs, transform" for georeference
        if img is None:  # Handle failed TIFF processing
            return None, None  # Stop processing if TIFF conversion failed
        draw = ImageDraw.Draw(img)

    # Draw bounding boxes
    bboxes = []
    for box in result.boxes.xywh.tolist():
        x_center, y_center, width, height = box
        x0 = int(x_center - width / 2)
        y0 = int(y_center - height / 2)
        x1 = int(x_center + width / 2)
        y1 = int(y_center + height / 2)

        bboxes.append([x0, y0, x1, y1])

    # Merge overlapping boxes
    merged_boxes = merge_overlapping_boxes(bboxes)

    # Draw merged bounding boxes
    draw, geojson, final_boxes = make_geojson(
        draw=draw,
        merged_boxes=merged_boxes,
        bounds=bounds,
        img_size=(img.width, img.height)
    )

    # Save image with bounding boxes
    print(f"Saving image to {make_image_path(image_path)}")
    save_image(np.array(img), make_image_path(image_path), crs, transform)

    # Save the geojson as image_path.geojson
    geojson_path = make_json_path(image_path)
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)

    # Save the annotated image to a buffer
    img_data = BytesIO()
    img.save(img_data, format="JPEG")
    img_data.seek(0)

    # Encode image as Base64
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")
    final_boxes = sorted(final_boxes, key=lambda x: x[1], reverse=True)

    return img_base64, final_boxes, geojson_path

# TODO: Make an endpoint for the processed planet image
# Current downloaded version is not georeferenced, but the original image is


@app.route('/download_geojson/<filename>', methods=['GET'])
def download_geojson(filename):
    geojson_path = os.path.join(app.config['JSON_FOLDER'], filename)

    if not os.path.exists(geojson_path):
        return jsonify({"error": "GeoJSON file not found"}), 404

    return send_file(geojson_path, as_attachment=True, mimetype="application/json")

@app.route('/download_planetscope_processed/<filename>', methods=['GET'])
def download_planetscope_processed(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path):
        return jsonify({"error": "Processed image not found"}), 404
    return send_file(processed_path, as_attachment=True)

@app.route('/download_planetscope_original/<filename>', methods=['GET'])
def download_planetscope_original(filename):
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        return jsonify({"error": "Original image not found"}), 404
    return send_file(original_path, as_attachment=True)

@app.route('/marinedebris/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image_base64, detections, json_path = detect_marine_debris(file_path)
        os.remove(file_path)

        return jsonify({
            "image_base64": image_base64,
            "detections": detections,
            "json_path": json_path
        })

    return jsonify({"error": "Invalid file type"}), 400

# ----------------------------------------------------------------------------------
# FOR SENTINEL DATA
# ----------------------------------------------------------------------------------


app.config['SENTINEL_UPLOAD_FOLDER'] = 'sentinel_uploads/'  # Define an upload directory
os.makedirs(app.config['SENTINEL_UPLOAD_FOLDER'], exist_ok=True)
app.config['SENTINEL_JSON_FOLDER'] = 'sentinel_json/'  # Define a JSON directory
os.makedirs(app.config['SENTINEL_JSON_FOLDER'], exist_ok=True)
app.config['SENTINEL_PROCESSED_FOLDER'] = 'sentinel_processed/'  # Define a processed directory
os.makedirs(app.config['SENTINEL_PROCESSED_FOLDER'], exist_ok=True)
VISUALS = [i for i in glob(app.config['SENTINEL_UPLOAD_FOLDER'], recursive=True) if 'conf' not in i and 'cl' not in i]


@app.route('/sentinel', methods=['POST'])
def sentinel():
    print("Received request at /sentinel")

    if 'file' not in request.files:
        print("No file provided in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    print(f"Received file: {file.filename}")

    if not file.filename.endswith(".tif"):
        print("Invalid file type, only .tif allowed")
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['SENTINEL_UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"Saved file to {file_path}")

    # Process file
    bboxes, geojson, output_image, profile = process_tif_file(file_path)
    output_image = Image.fromarray(make_uint8(np.array(output_image)))

    # Save geojson to file
    with open(make_json_path(file_path, mode='SENTINEL'), "w") as f:
        json.dump(geojson, f)

    # Save image for inspection reasons
    save_rasterio_image(output_image, filename, profile)

    img_data = BytesIO()
    output_image.save(img_data, format="JPEG")
    img_data.seek(0)

    # Encode image as Base64
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")
    final_boxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    response = {
        'bboxes': final_boxes,
        'image': img_base64,
    }

    return jsonify(response)

@app.route('/download_sentinel_geojson/<filename>', methods=['GET'])
def download_sentinel_geojson(filename):
    geojson_path = os.path.join(app.config['SENTINEL_JSON_FOLDER'], filename)
    if not os.path.exists(geojson_path):
        return jsonify({"error": "GeoJSON file not found"}), 404
    return send_file(geojson_path, as_attachment=True, mimetype="application/json")

@app.route('/download_sentinel_processed_tif/<filename>', methods=['GET'])
def download_sentinel_processed_tif(filename):
    processed_path = os.path.join(app.config['SENTINEL_PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path):
        return jsonify({"error": "Processed TIFF file not found"}), 404
    return send_file(processed_path, as_attachment=True)

@app.route('/download_sentinel_original_tif/<filename>', methods=['GET'])
def download_sentinel_original_tif(filename):
    original_path = os.path.join(app.config['SENTINEL_UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        return jsonify({"error": "Original TIFF file not found"}), 404
    return send_file(original_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

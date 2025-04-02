import os
import base64
import json
from io import BytesIO

import rasterio
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask_cors import CORS
from PIL import Image, ImageDraw
import numpy as np
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config.update({
    # Planetscope configurations
    'UPLOAD_FOLDER': 'uploads/',
    'JSON_FOLDER': 'json/',
    'PROCESSED_FOLDER': 'processed/',
    
    # Sentinel configurations
    'SENTINEL_UPLOAD_FOLDER': 'sentinel_uploads/',
    'SENTINEL_JSON_FOLDER': 'sentinel_json/',
    'SENTINEL_PROCESSED_FOLDER': 'sentinel_processed/',
    
    # Model path
    'MODEL_PATH': r"ai_models/PLANET.pt"
})

# Create required directories
for folder in [
    app.config['UPLOAD_FOLDER'],
    app.config['JSON_FOLDER'],
    app.config['PROCESSED_FOLDER'],
    app.config['SENTINEL_UPLOAD_FOLDER'],
    app.config['SENTINEL_JSON_FOLDER'],
    app.config['SENTINEL_PROCESSED_FOLDER']
]:
    os.makedirs(folder, exist_ok=True)

# Load the YOLO model
model = YOLO(app.config['MODEL_PATH'])


def make_json_path(file_path: str, mode: str = 'PLANET') -> str:
    """Generate path for JSON output file."""
    folder = app.config['JSON_FOLDER'] if mode == 'PLANET' else app.config['SENTINEL_JSON_FOLDER']
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(folder, f"{base_name}.geojson")


def make_image_path(file_path: str, mode: str = 'PLANET') -> str:
    """Generate path for processed image output."""
    folder = app.config['PROCESSED_FOLDER'] if mode == 'PLANET' else app.config['SENTINEL_PROCESSED_FOLDER']
    return os.path.join(folder, os.path.basename(file_path))


def save_rasterio_image(image: np.ndarray, filename: str, profile: dict):
    """Save image using rasterio with given profile."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = np.moveaxis(image, 2, 0)
    processed_path = os.path.join(app.config['SENTINEL_PROCESSED_FOLDER'], filename)
    with rasterio.open(processed_path, mode='w', **profile) as dst:
        dst.write(image)


def detect_marine_debris(image_path: str):
    """Detect marine debris in an image and return results."""
    # Run inference
    results = model(image_path, iou=0.8, verbose=False)
    result = results[0]  # Assume a single result

    # Process image based on file type
    if image_path.lower().endswith(".jpg"):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        bounds = crs = transform = None
    else:
        img, bounds, crs, transform = process_tif(image_path)
        if img is None:  # Handle failed TIFF processing
            return None, None
        draw = ImageDraw.Draw(img)

    # Extract and merge bounding boxes
    bboxes = []
    for box in result.boxes.xywh.tolist():
        x_center, y_center, width, height = box
        x0 = int(x_center - width / 2)
        y0 = int(y_center - height / 2)
        x1 = int(x_center + width / 2)
        y1 = int(y_center + height / 2)
        bboxes.append([x0, y0, x1, y1])

    merged_boxes = merge_overlapping_boxes(bboxes)

    # Create GeoJSON and draw boxes
    draw, geojson, final_boxes = make_geojson(
        draw=draw,
        merged_boxes=merged_boxes,
        bounds=bounds,
        img_size=(img.width, img.height)
    )

    # Save outputs
    save_image(np.array(img), make_image_path(image_path), crs, transform)
    
    geojson_path = make_json_path(image_path)
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)

    # Prepare image response
    img_data = BytesIO()
    img.save(img_data, format="JPEG")
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")
    
    return img_base64, sorted(final_boxes, key=lambda x: x[1], reverse=True), geojson_path


# Planetscope endpoints
@app.route('/marinedebris/detect', methods=['POST'])
def detect():
    """Endpoint for marine debris detection in Planetscope images."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image_base64, detections, json_path = detect_marine_debris(file_path)
    
    return jsonify({
        "image_base64": image_base64,
        "detections": detections,
        "json_path": json_path
    })


@app.route('/download_geojson/<filename>', methods=['GET'])
def download_geojson(filename):
    """Download GeoJSON file for Planetscope results."""
    geojson_path = os.path.join(app.config['JSON_FOLDER'], filename)
    return _send_file_if_exists(geojson_path, "application/json")


@app.route('/download_planetscope_processed/<filename>', methods=['GET'])
def download_planetscope_processed(filename):
    """Download processed Planetscope image."""
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return _send_file_if_exists(processed_path)


@app.route('/download_planetscope_original/<filename>', methods=['GET'])
def download_planetscope_original(filename):
    """Download original Planetscope image."""
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return _send_file_if_exists(original_path)


# Sentinel endpoints
@app.route('/sentinel', methods=['POST'])
def sentinel():
    """Endpoint for processing Sentinel images."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.endswith(".tif"):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['SENTINEL_UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Process file
    bboxes, geojson, output_image, profile = process_tif_file(file_path)
    output_image = Image.fromarray(make_uint8(np.array(output_image)))

    # Save outputs
    with open(make_json_path(file_path, mode='SENTINEL'), "w") as f:
        json.dump(geojson, f)
    save_rasterio_image(output_image, filename, profile)

    # Prepare response
    img_data = BytesIO()
    output_image.save(img_data, format="JPEG")
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")

    return jsonify({
        'bboxes': sorted(bboxes, key=lambda x: x[1], reverse=True),
        'image': img_base64
    })


@app.route('/download_sentinel_geojson/<filename>', methods=['GET'])
def download_sentinel_geojson(filename):
    """Download GeoJSON file for Sentinel results."""
    geojson_path = os.path.join(app.config['SENTINEL_JSON_FOLDER'], filename)
    return _send_file_if_exists(geojson_path, "application/json")


@app.route('/download_sentinel_processed_tif/<filename>', methods=['GET'])
def download_sentinel_processed_tif(filename):
    """Download processed Sentinel TIFF."""
    processed_path = os.path.join(app.config['SENTINEL_PROCESSED_FOLDER'], filename)
    return _send_file_if_exists(processed_path)


@app.route('/download_sentinel_original_tif/<filename>', methods=['GET'])
def download_sentinel_original_tif(filename):
    """Download original Sentinel TIFF."""
    original_path = os.path.join(app.config['SENTINEL_UPLOAD_FOLDER'], filename)
    return _send_file_if_exists(original_path)


def _send_file_if_exists(file_path, mimetype=None):
    """Helper function to send file if it exists or return 404."""
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, mimetype=mimetype)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
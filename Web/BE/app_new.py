

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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from flask import send_file
from planet_functions import (
    allowed_file,
    merge_overlapping_boxes,
    init_geojson,
    make_feature,
    process_tif,
    save_image,
    label_image
)

from sentinel_functions import (process_tif_file, draw_yolo_boxes, make_uint8)

# Load a monospaced font (fallback to default if not available)
try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=12)
except Exception:
    font = ImageFont.load_default()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------------------------------------------------------------------------------

# FOR PLANETSCOPE DATA

app.config['UPLOAD_FOLDER'] = 'uploads/'  # Define an upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['JSON_FOLDER'] = 'json/'  # Define a JSON directory
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)
app.config['PROCESSED_FOLDER'] = 'processed/'  # Define a processed directory
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
# Load the YOLO model
model_path = r"ai_models/PLANET.pt"
model = YOLO(model_path)


def make_json_path(x: str) -> str:
    return os.path.join(app.config['JSON_FOLDER'], f"{os.path.splitext(os.path.basename(x))[0]}.geojson")


def make_image_path(x: str) -> str:
    return os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(x))


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
        longitude_width = bounds.right - bounds.left
        latitude_height = abs(bounds.top - bounds.bottom)

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
    geojson = init_geojson()
    final_boxes = []
    for i, box in enumerate(merged_boxes):
        x0, y0, x1, y1 = box

        # Create properties
        box_properties = {
            "area_m2": abs((x1 - x0) * (y1 - y0) * 9),
            "box_id": chr(97 + i)
        }

        # Latlong determination
        if bounds:
            x0 = (bounds.left + (x0 / img.width) * longitude_width)
            y0 = (bounds.top - (y0 / img.height) * latitude_height)
            x1 = (bounds.left + (x1 / img.width) * longitude_width)
            y1 = (bounds.top - (y1 / img.height) * latitude_height)
            final_boxes.append([box_properties["box_id"], box_properties["area_m2"], x0, y0, x1, y1])
        else:
            final_boxes.append([box_properties["box_id"], box_properties["area_m2"], x0, y0, x1, y1])
            y0 = -y0
            y1 = -y1

        geojson["features"].append(make_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                properties=box_properties
        ))

        draw.rectangle(box, outline="red", width=3)
        draw = label_image(draw, box, box_properties["box_id"], font=font)

    # Save image with bounding boxes
    save_image(np.array(img), make_image_path(image_path), crs, transform)

    # Save the geojson as image_path.geojson
    with open(make_json_path(image_path), "w") as f:
        json.dump(geojson, f)

    # Save the annotated image to a buffer
    img_data = BytesIO()
    img.save(img_data, format="JPEG")
    img_data.seek(0)

    # Encode image as Base64
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")
    final_boxes = sorted(final_boxes, key=lambda x: x[1], reverse=True)

    return img_base64, final_boxes


@app.route('/download_geojson/<filename>', methods=['GET'])
def download_geojson(filename):
    geojson_path = os.path.join(app.config['JSON_FOLDER'], filename)

    if not os.path.exists(geojson_path):
        return jsonify({"error": "GeoJSON file not found"}), 404

    return send_file(geojson_path, as_attachment=True, mimetype="application/json")


@app.route('/marinedebris/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image_base64, detections = detect_marine_debris(file_path)
        json_path = make_json_path(file_path)
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


def save_rasterio_image(image: np.ndarray, filename: str, profile: dict):
    image = np.moveaxis(image, 2, 0)
    processed_path = os.path.join(app.config['SENTINEL_PROCESSED_FOLDER'], filename)
    with rasterio.open(processed_path, mode='w', **profile) as dst:
        dst.write(image)


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
    bboxes, image, profile = process_tif_file(file_path)
    image = make_uint8(image)

    # Draw bounding boxes on the image
    output_image = draw_yolo_boxes(image, bboxes)  # Returns image shape (height, width, channels)

    # Save image for inspection reasons
    save_rasterio_image(output_image, filename, profile)

    img_data = BytesIO()
    Image.fromarray(output_image).save(img_data, format="JPEG")
    img_data.seek(0)

    # Encode image as Base64
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")
    final_boxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    response = {
        'bboxes': final_boxes,
        'image': img_base64,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

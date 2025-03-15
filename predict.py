'''
Prediction script for inference.
'''

import os
import json
from glob import glob
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from typing import List
import rasterio
import numpy as np
from BE.functions import (
    process_tif,
    make_feature,
    merge_overlapping_boxes,
    init_geojson
)

JSON_FOLDER = "./BE/json/"
DRAWN_FOLDER = "./BE/drawn/"
PREDICTION_FOLDER = "./BE/uploads/"

# Load a monospaced font (fallback to default if not available)
try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=12)
except Exception:
    font = ImageFont.load_default()

# Load the YOLO model
MODEL_PATH = "runs/train/train3/weights/best.pt"
model = YOLO(MODEL_PATH)


def make_json_path(x: str) -> str:
    return os.path.join(JSON_FOLDER, f"{os.path.splitext(os.path.basename(x))[0]}.geojson")


def make_image_path(x: str) -> str:
    return os.path.join(DRAWN_FOLDER, os.path.basename(x))


def save_tif(img: np.array, crs, transform, path: str):
    img = np.transpose(img, (2, 0, 1))
    print(img.shape)
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=3,
            dtype=img.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(img)


def label_image(draw: ImageDraw, box: List, label: str) -> ImageDraw:
    # Draw label (box ID) at top-right corner with offset
    offset_x = 2
    offset_y = -10

    # Get size of the text
    text_width = draw.textlength(label, font=font)
    text_height = font.size
    text_x = min(box[2] + offset_x, draw._image.width - text_width - 3)
    text_y = max(box[1] + offset_y, 0)

    # Draw filled rectangle as background
    background_box = [(text_x, text_y), (text_x + text_width + 2, text_y + text_height + 2)]
    draw.rectangle(background_box, fill="black")

    # Draw text label
    draw.text((text_x + 1, text_y + 1), label, fill="white", font=font)

    return draw


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
        img, bounds, crs, transform = process_tif(image_path)
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
    latlong_boxes = []
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
            latlong_boxes.append([x0, y0, x1, y1])
        else:
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
        draw = label_image(draw, box, box_properties["box_id"])

    # Save the image with bounding boxes
    if not image_path.lower().endswith(".tif"):
        img.save(make_image_path(image_path))
    else:
        save_tif(np.array(img), crs, transform, make_image_path(image_path))

    # Save the geojson as image_path.geojson
    with open(make_json_path(image_path), "w") as f:
        json.dump(geojson, f)


if __name__ == "__main__":
    images = glob(os.path.join(PREDICTION_FOLDER, "*"))
    for image in images:
        detect_marine_debris(image)
        print(f"Processed {image}")

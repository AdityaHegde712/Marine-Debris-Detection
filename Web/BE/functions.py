'''
Util functions for main backend file.
'''

import os
from typing import List, Dict, Any
import rasterio
from rasterio.plot import reshape_as_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'tif'}


def increment_path(path):
    """Ensure unique inference directory by appending a number if needed."""
    base_path = path
    counter = 1
    while os.path.exists(path):
        path = f"{base_path}_{counter}"
        counter += 1
    return path


def label_image(draw: ImageDraw, box: List, label: str, font: ImageFont) -> ImageDraw:
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


def remove_nested_boxes(boxes):
    new_boxes = []
    for box in boxes:
        is_nested = False
        for other_box in boxes:
            if box != other_box:
                if (box[0] >= other_box[0] and box[1] >= other_box[1] and
                        box[2] <= other_box[2] and box[3] <= other_box[3]):
                    is_nested = True
                    break
        if not is_nested:
            new_boxes.append(box)
    return new_boxes


def min_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(min(box1_area, box2_area))

    return iou


def merge_overlapping_boxes(boxes, iou_threshold=0.4):
    if len(boxes) == 1:
        return boxes
    while True:
        merged = False
        for idx, box in enumerate(boxes):
            for other_idx, other_box in enumerate(boxes[idx+1:]):
                iou = min_iou(box, other_box)
                if iou == 1:
                    boxes = remove_nested_boxes(boxes)
                    merged = True
                    break
                if iou > iou_threshold:
                    new_box = [
                        min(box[0], other_box[0]),
                        min(box[1], other_box[1]),
                        max(box[2], other_box[2]),
                        max(box[3], other_box[3])
                    ]
                    boxes[idx] = new_box
                    del boxes[idx + 1 + other_idx]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return boxes


def init_geojson() -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": []
    }


def make_feature(x0, y0, x1, y1, properties: Dict = {}) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [x0, y0],  # Top left
                    [x1, y0],  # Top right
                    [x1, y1],  # Bottom right
                    [x0, y1],  # Bottom left
                    [x0, y0]  # back to top left
                ]
            ]
        },
    }


def process_tif(tif_path):
    try:
        # Open TIFF using rasterio
        with rasterio.open(tif_path, driver="Gtiff") as dataset:
            # Read the image as an array
            img_array = dataset.read()
            crs, transform = dataset.crs, dataset.transform
            bounds = dataset.bounds

            # Reshape to (height, width, channels)
            img_array = reshape_as_image(img_array)

            # Convert to RGB (if more than 3 channels, take the first 3)
            if img_array.shape[-1] > 3:
                img_array = img_array[:, :, :3]  # Keep only the first 3 channels

            # Convert NumPy array to PIL Image
            img = Image.fromarray(np.uint8(img_array))

            return img, bounds, crs, transform  # Return the processed image and bounds

    except Exception as e:
        print(f"Error processing TIFF file: {e}")
        return None, None  # Explicitly return None for both values


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

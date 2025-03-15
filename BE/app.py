# import os
# import base64
# from io import BytesIO
# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# from PIL import Image, ImageDraw
# from ultralytics import YOLO
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# app.config['UPLOAD_FOLDER'] = 'uploads/'  # Define an upload directory
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', '.tif'}

# def increment_path(path):
#     """Ensure unique inference directory by appending a number if needed."""
#     base_path = path
#     counter = 1
#     while os.path.exists(path):
#         path = f"{base_path}_{counter}"
#         counter += 1
#     return path

# def remove_nested_boxes(boxes):
#     new_boxes = []
#     for box in boxes:
#         is_nested = False
#         for other_box in boxes:
#             if box != other_box:
#                 if (box[0] >= other_box[0] and box[1] >= other_box[1] and
#                         box[2] <= other_box[2] and box[3] <= other_box[3]):
#                     is_nested = True
#                     break
#         if not is_nested:
#             new_boxes.append(box)
#     return new_boxes

# def min_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)

#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     iou = inter_area / float(min(box1_area, box2_area))
    
#     return iou

# def merge_overlapping_boxes(boxes, iou_threshold=0.4):
#     if len(boxes) == 1:
#         return boxes

#     while True:
#         merged = False
#         for idx, box in enumerate(boxes):
#             for other_idx, other_box in enumerate(boxes[idx+1:]):
#                 iou = min_iou(box, other_box)
#                 if iou == 1:
#                     boxes = remove_nested_boxes(boxes)
#                     merged = True
#                     break
#                 if iou > iou_threshold:
#                     new_box = [
#                         min(box[0], other_box[0]),
#                         min(box[1], other_box[1]),
#                         max(box[2], other_box[2]),
#                         max(box[3], other_box[3])
#                     ]
#                     boxes[idx] = new_box
#                     del boxes[idx + 1 + other_idx]
#                     merged = True
#                     break
#             if merged:
#                 break
#         if not merged:
#             break
#     return boxes

# def detect_marine_debris(image_path):
#     model_path = r"ai_models/PLANET.pt"  # Using PLANETSCOPE model
#     inference_dir = r"detection_outputs/inference"
#     inference_dir = increment_path(inference_dir)
#     os.makedirs(inference_dir, exist_ok=True)

#     # Load the YOLO model
#     model = YOLO(model_path)

#     # Run inference
#     results = model(image_path, iou=0.6)
#     result = results[0]  # Assume a single result

#     # Open the original image
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)

#     # Draw bounding boxes
#     detections = []
#     for box in result.boxes.xywh.tolist():
#         x_center, y_center, width, height = box
#         x0 = int(x_center - width / 2)
#         y0 = int(y_center - height / 2)
#         x1 = int(x_center + width / 2)
#         y1 = int(y_center + height / 2)
#         detections.append({
#             "bbox": [x0, y0, x1, y1],
#         })
    
#     # Merge overlapping boxes
#     # Store bounding boxes in a separate list
#     bboxes = [d["bbox"] for d in detections]

#     # Merge overlapping boxes
#     merged_boxes = merge_overlapping_boxes(bboxes)

#     # Draw merged bounding boxes
#     for box in merged_boxes:
#         draw.rectangle(box, outline="red", width=3)


#     # Save the annotated image to a buffer
#     img_data = BytesIO()
#     img.save(img_data, format="JPEG")
#     img_data.seek(0)

#     # Encode image as Base64
#     img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")

#     return img_base64, merged_boxes

# @app.route('/marinedebris/detect', methods=['POST'])
# def detect():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400
    
#     file = request.files['file']
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         image_base64, detections = detect_marine_debris(file_path)
#         os.remove(file_path)  # Cleanup

#         return jsonify({
#             "image_base64": image_base64,
#             "detections": detections
#         })

#     return jsonify({"error": "Invalid file type"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)

import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, TiffImagePlugin
from ultralytics import YOLO
from flask_cors import CORS
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'uploads/'  # Define an upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_path = r"ai_models/PLANET.pt" 
# Load the YOLO model
model = YOLO(model_path)

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

from PIL import Image

def process_tif(tif_path):
    try:
        # Open TIFF using rasterio
        with rasterio.open(tif_path, driver="Gtiff") as dataset:
            print(f"TIFF metadata: {dataset.meta}")  # Debugging info
            
            # Read the image as an array
            img_array = dataset.read()
            print(dataset.meta)
            import rasterio

            bounds = dataset.bounds

            # Reshape to (height, width, channels)
            img_array = reshape_as_image(img_array)
            print(f"Image shape after reshape: {img_array.shape}")

            # Convert to RGB (if more than 3 channels, take the first 3)
            if img_array.shape[-1] > 3:
                img_array = img_array[:, :, :3]  # Keep only the first 3 channels
            
            # Convert NumPy array to PIL Image
            img = Image.fromarray(np.uint8(img_array))

            # Save as JPEG
            # temp_jpeg_path = tif_path.replace(".tif", ".jpg")
            # img.save(temp_jpeg_path, format="JPEG")

            return img, bounds # Return path for further processing

    except Exception as e:
        print(f"Error processing TIFF file: {e}")
        return None


def detect_marine_debris(image_path):
     # Using PLANETSCOPE model
    inference_dir = r"detection_outputs/inference"
    inference_dir = increment_path(inference_dir)
    os.makedirs(inference_dir, exist_ok=True)


    # Run inference
    results = model(image_path, iou=0.8)
    result = results[0]  # Assume a single result

    # Open the original image
    if image_path.lower().endswith(".jpg"):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    else:
        img, bounds = process_tif(image_path)
        draw = ImageDraw.Draw(img)
        
    # Draw bounding boxes
    detections = []
    for box in result.boxes.xywh.tolist():
        x_center, y_center, width, height = box
        x0 = int(x_center - width / 2)
        y0 = int(y_center - height / 2)
        x1 = int(x_center + width / 2)
        y1 = int(y_center + height / 2)
        detections.append({
            "bbox": [x0, y0, x1, y1],
        })
    
    # Merge overlapping boxes
    bboxes = [d["bbox"] for d in detections]
    merged_boxes = merge_overlapping_boxes(bboxes)

    # Draw merged bounding boxes
    for box in merged_boxes:
        draw.rectangle(box, outline="red", width=3)

    # Save the annotated image to a buffer
    img_data = BytesIO()
    img.save(img_data, format="JPEG")
    img_data.seek(0)

    # Encode image as Base64
    img_base64 = base64.b64encode(img_data.getvalue()).decode("utf-8")

    return img_base64, merged_boxes

@app.route('/marinedebris/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert TIFF if necessary
        # if filename.lower().endswith('.tif'):
        #     processed_image = process_tif(file_path)
        #     if not processed_image:
        #         return jsonify({"error": "Failed to process TIFF"}), 500
        #     file_path = processed_image # Use converted JPEG for detection

        image_base64, detections = detect_marine_debris(file_path)
        os.remove(file_path)  # Cleanup

        return jsonify({
            "image_base64": image_base64,
            "detections": detections
        })

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == '__main__':
    app.run(debug=True)

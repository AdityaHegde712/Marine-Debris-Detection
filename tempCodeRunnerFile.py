'''
As the name implies, a test file to run scripts. Has no importance in the application.
'''
import os
import shutil
from glob import glob
import rasterio
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from typing import List

# Define input and output folders
INPUT_FOLDER = "Test_small/**/*.tif"
VISUALS = [i for i in glob(INPUT_FOLDER, recursive=True) if 'conf' not in i and 'cl' not in i]
OUTPUT_COASTAL = "./masks/coastal"
OUTPUT_SEA = "./masks/sea"
OUTPUT_BASE = "./masks/base"
OUTPUT_FINAL = "./masks/output"
os.makedirs(OUTPUT_COASTAL, exist_ok=True)
os.makedirs(OUTPUT_SEA, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(OUTPUT_FINAL, exist_ok=True)
output_dir = "masks/boxes"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Sentinel-2 Band Wavelengths (in nm)
WAVELENGTHS = {
    "B4": 665,    # Red
    "B6": 740.5,  # Red Edge 2
    "B8": 832.8,  # NIR
    "B11": 1613.7  # SWIR1
}


def make_255(image: np.ndarray) -> np.ndarray:
    return np.where(image > 0, 255, 0).astype(np.uint8)


def compute_indices(dataset):
    """Compute NDVI, NDWI, and FDI from Sentinel-2 bands."""
    # Read required bands
    green = dataset.read(3).astype(np.float32)  # Band 3 (Green)
    red = dataset.read(4).astype(np.float32)  # Band 4 (Red)
    red_edge2 = dataset.read(6).astype(np.float32)  # Band 6 (Red Edge 2)
    nir = dataset.read(8).astype(np.float32)  # Band 8 (NIR)
    swir1 = dataset.read(10).astype(np.float32)  # Band 11 (SWIR1)

    # NDVI Calculation
    ndvi = (nir - red) / (nir + red + 1e-10)

    # NDWI Calculation
    ndwi = (green - nir) / (green + nir + 1e-10)

    # FDI Calculation
    I8_prime = red_edge2 + (swir1 - red_edge2) * ((WAVELENGTHS["B8"] - WAVELENGTHS["B4"]) /
                                                  (WAVELENGTHS["B8"] + WAVELENGTHS["B4"])) * 10
    fdi = nir - I8_prime

    return ndvi, ndwi, fdi


# TO BE DEFINED @Ananya
def decide_mask(coastal_mask: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    """
    Decide which mask to use based on pixel coverage.
    If a mask covers between 20% and 60% of the image, return that mask.
    If both meet the criteria, return the one with the higher coverage.
    If neither meets the criteria, return an empty mask.
    """
    def coverage(mask: np.ndarray) -> float:
        return np.sum(mask > 0) / mask.size  # Ratio of nonzero pixels to total pixels
    
    coastal_coverage = coverage(coastal_mask)
    print(coastal_coverage)
    sea_coverage = coverage(sea_mask)
    print(sea_coverage)
    
    valid_coastal = 0 < coastal_coverage <= 0.6
    valid_sea = 0 < sea_coverage <= 0.6
    
    if valid_coastal and valid_sea:
        return coastal_mask if coastal_coverage < sea_coverage else sea_mask
    elif valid_coastal:
        return coastal_mask
    elif valid_sea:
        return sea_mask
    
    return np.zeros_like(coastal_mask)


def get_bboxes(mask: np.ndarray) -> np.ndarray:
    # Ensure mask is a numpy array if it's a torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # If the mask has more than 2 dimensions (e.g., a color image), we need to reduce it to 2D
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask)
    elif len(mask.shape) == 3 and mask.shape[0] > 1:
        raise ValueError("Mask has more than one channel. Please provide a single-channel mask. We haven't built this for multichannel masks yet.")

    # Label connected components in the mask
    labeled_mask = label(mask == 255)

    # Get properties of labeled regions (connected components)
    props = regionprops(labeled_mask)

    # Extract bounding boxes
    bounding_boxes = []
    for i, prop in enumerate(props):
        ymin, xmin, ymax, xmax = prop.bbox
        area = prop.area
        bounding_boxes.append([chr(i + 97), area, xmin, ymin, xmax, ymax])

    return bounding_boxes


def process_tif_file(file_path):
    """Process a single .tif file and save the stacked output in FDI, NDWI, NDVI order."""
    coastal_output = os.path.join(OUTPUT_COASTAL, os.path.basename(file_path).replace(".tif", "_0coastal.tif"))
    sea_debris_output = os.path.join(OUTPUT_SEA, os.path.basename(file_path).replace(".tif", "_1sea_debris.tif"))
    merged_output = os.path.join(OUTPUT_FINAL, os.path.basename(file_path).replace(".tif", "_2merged.tif"))

    with rasterio.open(file_path) as dataset:
        # Compute indices
        ndvi, ndwi, fdi = compute_indices(dataset)

        # Make the mask
        coastal_mask = make_255(((fdi > 0.1) & (ndwi > 0) & (ndvi < 0))).astype(np.uint8)
        sea_debris_mask = make_255(
            remove_small_objects(
                (
                    (fdi > 0.009) & (fdi < 0.1) &
                    (ndwi < 0.35) &
                    (ndvi > -0.25)
                ),
                min_size=2
            )
        ).astype(np.uint8)

        # Update metadata (preserve georeferencing)
        profile = dataset.profile
        profile.update(dtype=np.uint8, count=1)

        # Save the new image
        with rasterio.open(coastal_output, "w", **profile) as dst:
            dst.write(coastal_mask, 1)
        with rasterio.open(sea_debris_output, "w", **profile) as dst:
            dst.write(sea_debris_mask, 1)

        # Decide mask to use
        final_mask = decide_mask(coastal_mask, sea_debris_mask)
        with rasterio.open(merged_output, "w", **profile) as dst:
            dst.write(final_mask, 1)
        # Get bounding boxes
        bboxes = get_bboxes(final_mask)
        if not bboxes:
            return None, final_mask
        # print(*bboxes, sep="\n", end=f"\n{len(bboxes)}\n")

    shutil.copy(file_path, OUTPUT_BASE)
    return bboxes, final_mask

def draw_yolo_boxes(image: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
    """Function to draw YOLO boxes on the image.

    Args:
        image (np.ndarray): Image array.
        bboxes (List[List[int]]): List of lists containing bboxes and class name.

    Returns:
        np.ndarray: Image array with boxes drawn.
    """
    if image.shape[0] == 1:
        image = np.stack((image[0],) * 3, axis=-1)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    for bbox in bboxes:
        _, _, x1, y1, x2, y2 = bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
    return image


# for image, bboxes in zip(new_images, new_bbox_set):
#     if image.shape[0] == 1:
#         image = np.stack((image[0],) * 3, axis=-1)
#     if len(image.shape) == 2:
#         image = np.stack((image,) * 3, axis=-1)
#     plt.figure(figsize=(10, 10))
#     print(f"Clumpy area: {bboxes[-1][1]}")
#     plt.imshow(draw_yolo_boxes(image, bboxes[:-1]))
#     plt.axis("off")
#     plt.show()

def side_by_side(image_set_1, image_set_2):
    for image1, image2 in zip(image_set_1, image_set_2):
        plt.figure(figsize=(6, 3))

        # Original mask
        plt.subplot(1, 2, 1)
        plt.imshow(image1, cmap="gray")
        plt.title("Original Mask")
        plt.axis("off")

        # Processed mask
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="gray")
        plt.title("Processed Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    # Iterate through all .tif files in the input folder
    for filename in tqdm(VISUALS[:100]):
        if filename.endswith(".tif"):
            bboxes, mask = process_tif_file(filename)

            output_image = draw_yolo_boxes(mask, bboxes)
        output_image = np.moveaxis(output_image, 2, 0)
        count, height, width = output_image.shape
        print(output_image.shape)
        with rasterio.open(os.path.join(output_dir, os.path.basename(filename)), mode='w', driver="Gtiff", count=count, width=width, height=height, dtype=np.uint8) as dst:
            dst.write(output_image )

    print("🎉 All files processed successfully!")


if __name__ == "__main__":
    main()

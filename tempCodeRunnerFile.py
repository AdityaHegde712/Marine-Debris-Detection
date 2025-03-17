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

# Define input and output folders
INPUT_FOLDER = "../datasets/MARIDA_sentinel-2/patches/**/*.tif"
VISUALS = [i for i in glob(INPUT_FOLDER, recursive=True) if 'conf' not in i and 'cl' not in i]
OUTPUT_COASTAL = "./masks/coastal"
OUTPUT_SEA = "./masks/sea"
OUTPUT_BASE = "./masks/base"
os.makedirs(OUTPUT_COASTAL, exist_ok=True)
os.makedirs(OUTPUT_SEA, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

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


def process_tif_file(file_path):
    """Process a single .tif file and save the stacked output in FDI, NDWI, NDVI order."""
    coastal_output = os.path.join(OUTPUT_COASTAL, os.path.basename(file_path).replace(".tif", "_0coastal.tif"))
    sea_debris_output = os.path.join(OUTPUT_SEA, os.path.basename(file_path).replace(".tif", "_1sea_debris.tif"))
    # merged_output = os.path.join(output_folder, os.path.basename(file_path).replace(".tif", "_2merged.tif"))

    with rasterio.open(file_path) as dataset:
        # Compute indices
        ndvi, ndwi, fdi = compute_indices(dataset)

        # Make the mask
        coastal_mask = ((fdi > 0.1) & (ndwi > 0) & (ndvi < 0)).astype(np.float32)
        sea_debris_mask = remove_small_objects(((fdi > 0.009) & (fdi < 0.1) &
                                                (ndwi < 0.35) &
                                                (ndvi > -0.25)), min_size=2).astype(np.float32)
        # merged_mask = ((coastal_mask == 1) | (sea_debris_mask == 1)).astype(np.float32)

        # Update metadata (preserve georeferencing)
        profile = dataset.profile
        profile.update(dtype=np.float32, count=1)

        # Save the new image
        with rasterio.open(coastal_output, "w", **profile) as dst:
            dst.write(coastal_mask, 1)
        with rasterio.open(sea_debris_output, "w", **profile) as dst:
            dst.write(sea_debris_mask, 1)
        # with rasterio.open(merged_output, "w", **profile) as dst:
        #     dst.write(merged_mask, 1)

    shutil.copy(file_path, OUTPUT_BASE)


def process_mask(mask: np.ndarray, bands: int = 1) -> np.ndarray:
    intermediary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    # If single band, convert to 3 band
    if bands == 3 and (len(intermediary_mask.shape) == 2 or 1 in intermediary_mask.shape):
        return np.stack((intermediary_mask,) * 3, axis=-1)
    return intermediary_mask


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
    for prop in props:
        ymin, xmin, ymax, xmax = prop.bbox
        bounding_boxes.append((xmin, ymin, xmax, ymax))

    return bounding_boxes


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
            process_tif_file(filename)

    print("🎉 All files processed successfully!")


if __name__ == "__main__":
    main()

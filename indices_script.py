import os
import numpy as np
import rasterio

# Define input and output folders
input_folder = "sentinel"  # Change this to the folder containing .tif files
output_folder = "indiced_images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Sentinel-2 Band Wavelengths (in nm)
wavelengths = {
    "B4": 665,   # Red
    "B6": 740.5, # Red Edge 2
    "B8": 832.8, # NIR
    "B11": 1613.7 # SWIR1
}

def compute_indices(dataset):
    """Compute NDVI, NDWI, and FDI."""
    # Read required bands
    green = dataset.read(3).astype(np.float32)  # Band 3 (Green)
    red = dataset.read(4).astype(np.float32)    # Band 4 (Red)
    red_edge2 = dataset.read(6).astype(np.float32)  # Band 6 (Red Edge 2)
    nir = dataset.read(8).astype(np.float32)    # Band 8 (NIR)
    swir1 = dataset.read(10).astype(np.float32) # Band 11 (SWIR1)

    # NDVI Calculation
    ndvi = (nir - red) / (nir + red + 1e-10)

    # NDWI Calculation
    ndwi = (green - nir) / (green + nir + 1e-10)

    # FDI Calculation
    I8_prime = red_edge2 + (swir1 - red_edge2) * ((wavelengths["B8"] - wavelengths["B4"]) /
                                                  (wavelengths["B8"] + wavelengths["B4"])) * 10
    fdi = nir - I8_prime

    return ndvi, ndwi, fdi

def process_tif_file(file_path):
    """Process a single .tif file and save the stacked output."""
    output_path = os.path.join(output_folder, os.path.basename(file_path))

    with rasterio.open(file_path) as dataset:
        # Compute indices
        ndvi, ndwi, fdi = compute_indices(dataset)

        # Stack the computed indices
        stacked_image = np.stack([ndvi, ndwi, fdi])

        # Rescale to 0-255 and convert to uint8
        stacked_rescaled = ((stacked_image - np.min(stacked_image)) / 
                            (np.max(stacked_image) - np.min(stacked_image)) * 255).astype(np.uint8)

        # Update metadata
        profile = dataset.profile
        profile.update(dtype=rasterio.uint8, count=3)

        # Save the new image
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(3):
                dst.write(stacked_rescaled[i], i + 1)

    print(f"✅ Processed and saved: {output_path}")

# Iterate through all .tif files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        process_tif_file(os.path.join(input_folder, filename))

print("🎉 All files processed successfully!")

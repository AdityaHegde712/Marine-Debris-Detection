# import os
# import cv2
# import numpy as np

# # Define input and output directories
# INPUT_DIR = "sample_planet_testing"
# OUTPUT_DIR = "planet_filtered"

# # Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def apply_filters(image_path, output_dir):
#     """Applies noise reduction, spectral index computation, edge enhancement, and morphological filtering."""
    
#     filename = os.path.basename(image_path).split('.')[0]  # Extract filename without extension
    
#     # Open Image
#     image = cv2.imread(image_path)  # Read as BGR
#     if image is None:
#         print(f"Error loading {image_path}")
#         return
    
#     # Convert to Grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 1. Noise Reduction
#     gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
#     median_filtered = cv2.medianBlur(gray, 5)

#     # 3. Edge Enhancement
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#     laplacian = np.uint8(np.absolute(laplacian))
#     edges = cv2.Canny(gray, 50, 150)

#     # 4. Morphological Operations
#     kernel = np.ones((3, 3), np.uint8)
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

#     # Save Processed Images
#     save_image(gaussian_filtered, output_dir, f"{filename}_gaussian.jpg")
#     save_image(median_filtered, output_dir, f"{filename}_median.jpg")
#     save_image(laplacian, output_dir, f"{filename}_laplacian.jpg")
#     save_image(edges, output_dir, f"{filename}_edges.jpg")
#     save_image(closed, output_dir, f"{filename}_closed.jpg")
#     save_image(opened, output_dir, f"{filename}_opened.jpg")

#     print(f"Processed: {filename}")

# def save_image(image, output_dir, filename):
#     """Saves an image to the specified output directory."""
#     output_path = os.path.join(output_dir, filename)
#     cv2.imwrite(output_path, image)

# def process_directory(input_dir, output_dir):
#     """Loops through the input directory and processes each image."""
#     for file in os.listdir(input_dir):
#         if file.lower().endswith((".jpg", ".jpeg", ".png")):
#             image_path = os.path.join(input_dir, file)
#             apply_filters(image_path, output_dir)

# # Run the script
# if __name__ == "__main__":
#     process_directory(INPUT_DIR, OUTPUT_DIR)
#     print(f"Processing complete. Output saved in: {OUTPUT_DIR}")

import os
import cv2
import numpy as np

# Define input and output directories
INPUT_DIR = "sample_planet_testing"
OUTPUT_DIR = "planet_filtered"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def weighted_avg(image, kernel, kernel_weights):        
    # Pad the image to handle borders
    pad_size = kernel // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    # Traverse the image, multiply the pixel values with the kernel weights and sum them
    for i in range(image.shape[0]):  # Going through the rows
        for j in range(image.shape[1]):  # Iterate over columns
            # Extract the neighborhood
            neighborhood = padded_image[i : i + kernel, j : j + kernel]  # Extracts window around the current (i,j) pixel
            # Multiply with the kernel weights and sum them
            output[i, j] = np.sum(neighborhood * kernel_weights)
    
    return output.astype(np.uint8)

def apply_filters(image_path, output_dir):
    """Applies noise reduction, spectral index computation, edge enhancement, and morphological filtering."""
    
    filename = os.path.basename(image_path).split('.')[0]  # Extract filename without extension
    
    # Open Image
    image = cv2.imread(image_path)  # Read as BGR
    if image is None:
        print(f"Error loading {image_path}")
        return
    
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Noise Reduction using Weighted Gaussian Filter
    kernel = 3
    kernel_weights = np.array([1, 1, 1, 
                                1, 10, 1, 
                                1, 1, 1]).reshape((3, 3)) / 16
    gaussian_filtered = weighted_avg(gray, kernel, kernel_weights)

    median_filtered = cv2.medianBlur(gray, 5)

    # 3. Edge Enhancement
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    edges = cv2.Canny(gray, 50, 150)

    # 4. Morphological Operations (Commented Out)
    # kernel = np.ones((3, 3), np.uint8)
    # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # Save Processed Images
    save_image(gaussian_filtered, output_dir, f"{filename}_gaussian.jpg")
    save_image(median_filtered, output_dir, f"{filename}_median.jpg")
    save_image(laplacian, output_dir, f"{filename}_laplacian.jpg")
    save_image(edges, output_dir, f"{filename}_edges.jpg")
    # save_image(closed, output_dir, f"{filename}_closed.jpg")
    # save_image(opened, output_dir, f"{filename}_opened.jpg")

    print(f"Processed: {filename}")

def save_image(image, output_dir, filename):
    """Saves an image to the specified output directory."""
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

def process_directory(input_dir, output_dir):
    """Loops through the input directory and processes each image."""
    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, file)
            apply_filters(image_path, output_dir)

# Run the script
if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
    print(f"Processing complete. Output saved in: {OUTPUT_DIR}")

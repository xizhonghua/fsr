import os
import cv2
import argparse
from tqdm import tqdm

def denoise_image(img, method='fastnl', h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Denoises a single image using OpenCV's denoising algorithms.

    Args:
        img: The input image (numpy array).
        method: Denoising method ('fastnl', 'tvl1', or 'gaussian').
                'fastnl': Fast Non-Local Means Denoising (for color images).
                'tvl1':  Total Variation Denoising (good for cartoon-like images).
                'gaussian' : Apply a Gaussian filter.
        h: Parameter controlling filter strength (higher = stronger denoising).
           Only used for 'fastnl'.
        templateWindowSize:  Size of the template patch (odd integer). Only used for 'fastnl'.
        searchWindowSize: Size of the search window (odd integer). Only used for 'fastnl'.

    Returns:
        The denoised image.
    """

    if method == 'fastnl':
        # For color images
        if len(img.shape) == 3:  # Check if it's a color image
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)
        else: # Grayscale image
            denoised_img = cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
    elif method == 'tvl1':
        # Convert to float32 for TVL1
        img_float = img.astype(np.float32) / 255.0
        denoised_img = cv2.denoise_TVL1(img_float, h) # h is lambda here
        denoised_img = (denoised_img * 255).astype(np.uint8) # Scale back

    elif method == 'gaussian':
        # Use Gaussian Blur.  Kernel size must be odd.
        kernel_size = (searchWindowSize, searchWindowSize)
        denoised_img = cv2.GaussianBlur(img, kernel_size, 0) # 0 calculates sigma automatically

    else:
        raise ValueError(f"Invalid denoising method: {method}")

    return denoised_img


def process_images(input_folder, output_folder, method='fastnl', h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Processes a folder of images, denoises them, and saves the results.

    Args: (Same as denoise_image, but applied to a folder)
        input_folder: Path to the input folder.
        output_folder: Path to the output folder.
        method, h, templateWindowSize, searchWindowSize:  Denoising parameters.
    """
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc=f"Processing {root}", unit="image"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                try:
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)

                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_subfolder, file)

                    img = cv2.imread(input_path)
                    if img is None:
                        print(f"Warning: Could not read image '{input_path}'. Skipping.")
                        continue

                    denoised_img = denoise_image(img, method, h, templateWindowSize, searchWindowSize)
                    cv2.imwrite(output_path, denoised_img)

                except Exception as e:
                    print(f"Error processing image '{input_path}': {e}")
                    continue

import numpy as np  # Import numpy

def main():
    parser = argparse.ArgumentParser(description="Denoise images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder.")
    parser.add_argument("output_folder", help="Path to the output folder.")
    parser.add_argument("-m", "--method", choices=['fastnl', 'tvl1', 'gaussian'], default='fastnl',
                        help="Denoising method (default: fastnl).")
    parser.add_argument("--h_value", type=float, default=5,
                        help="Filter strength parameter (default: 5).  Higher values = stronger denoising.")
    parser.add_argument("-t", "--template_window_size", type=int, default=7,
                        help="Template window size (odd integer, default: 7). For fastnl only.")
    parser.add_argument("-s", "--search_window_size", type=int, default=21,
                        help="Search window size (odd integer, default: 21). For fastnl and gaussian.")


    args = parser.parse_args()

    # Additional validation for window sizes (must be odd)
    if args.method in ('fastnl', 'gaussian') and (args.template_window_size % 2 == 0 or args.search_window_size % 2 == 0):
        parser.error("template_window_size and search_window_size must be odd integers.")


    process_images(args.input_folder, args.output_folder, args.method, args.h_value,
                   args.template_window_size, args.search_window_size)
    print("Image denoising complete.")

if __name__ == "__main__":
    main()
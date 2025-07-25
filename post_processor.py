#
# Script with post-processing functions, such as applying motion blur and creating binary masks
#

import cv2 as cv
import numpy as np
import random
import os
import argparse
import pandas as pd
import shutil


def get_motion_blur_kernel(x, y, thickness=1, ksize=21):
    """ Obtains Motion Blur Kernel
        Inputs:
            x - horizontal direction of blur
            y - vertical direction of blur
            thickness - thickness of blur kernel line
            ksize - size of blur kernel
        Outputs:
            blur_kernel
        """
    c = int(ksize/2)

    blur_kernel = np.zeros((ksize, ksize), dtype=np.uint8)
    blur_kernel = cv.line(blur_kernel, (c, c), (c + x, c + y), (255,), thickness)

    # Convert to float and normalize
    blur_kernel = blur_kernel.astype(np.float32)
    blur_kernel /= np.sum(blur_kernel)  # Normalize to sum to 1

    return blur_kernel


def add_motion_blur(dataset_folder_path, csv_file, prob=0.2):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert the 'Motion Blur Kernel Parameters' column to object/string type if it exists
    if 'Motion Blur Kernel Parameters [x, y, size]' in df.columns:
        df['Motion Blur Kernel Parameters [x, y, size]'] = df['Motion Blur Kernel Parameters [x, y, size]'].astype(str)

    for filename in os.listdir(dataset_folder_path):
        if random.random() < prob:
            img_path = os.path.join(dataset_folder_path, filename)
            img = cv.imread(img_path)

            if img is None:
                print(f"Skipping {filename}: Unable to read image.")
                continue

            # Specify the kernel size
            # The greater the size, the more the motion
            kernel_size = random.randint(10, 30)
            x = random.randint(-kernel_size // 2, kernel_size // 2)  # Random direction
            y = random.randint(-kernel_size // 2, kernel_size // 2)

            kernel = get_motion_blur_kernel(x, y, thickness=1, ksize=kernel_size)

            img_blurred = cv.filter2D(img, -1, kernel) # Apply the random kernel

            # Overwrite the original image
            cv.imwrite(img_path, img_blurred)

            # Extract the image name without the extension
            image_name = os.path.splitext(filename)[0]
            
            # Find the row with the matching image name in the first column
            mask = df.iloc[:, 0] == image_name
            if any(mask):
                # Convert parameters to string format
                params_str = f"[{x}, {y}, {kernel_size}]"
                # Update the 'Motion Blur Kernel Parameters' column for the matching row
                df.loc[mask, 'Motion Blur Kernel Parameters [x, y, size]'] = params_str
                print(f"Updated CSV for image {image_name} with parameters {params_str}")
            else:
                print(f"Warning: Image {image_name} not found in CSV file.")
    
    # Save the updated CSV
    df.to_csv(csv_file, index=False)


def force_motion_blur(dataset_folder_path, dictionary, img_name):

    if type(dictionary['motion_blur']) == bool:
        return False

    filename = f"{img_name}.png"
    
    img_path = os.path.join(dataset_folder_path, filename)
    img = cv.imread(img_path)

    if img is None:
        print(f"Skipping {filename}: Unable to read image.")
        return False

    # Get the kernel size
    # The greater the size, the greater the motion
    x = dictionary['motion_blur'][0]
    y = dictionary['motion_blur'][1]
    kernel_size = dictionary['motion_blur'][2]   


    kernel = get_motion_blur_kernel(x, y, thickness=1, ksize=kernel_size)

    img_blurred = cv.filter2D(img, -1, kernel) # Apply the random kernel

    # Overwrite the original image
    cv.imwrite(img_path, img_blurred)
    

def generate_masks_from_depth(rgb_folder, depth_folder, output_folder, threshold=0.95):
    """
    Generate binary masks using depth images.
    
    Args:
        rgb_folder (str): Folder containing RGB images (used for naming only).
        depth_folder (str): Folder containing corresponding depth images.
        output_folder (str): Folder to save the generated binary masks.
        threshold (float): Depth threshold (0.0 to 1.0) below which pixels are considered foreground.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(rgb_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        base_name = os.path.splitext(filename)[0]
        depth_filename = f"{base_name}_depth.png"
        depth_path = os.path.join(depth_folder, depth_filename)
        
        if not os.path.exists(depth_path):
            print(f"⚠️ Depth image not found for: {filename}")
            continue
        
        # Load depth image (grayscale)
        depth_img = cv.imread(depth_path, cv.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"❌ Failed to load: {depth_path}")
            continue

        # Normalize and threshold
        depth_normalized = depth_img.astype(np.float32) / 255.0
        mask = (depth_normalized < threshold).astype(np.uint8) * 255

        # Save binary mask
        mask_output_path = os.path.join(output_folder, f"{base_name}_mask.png")
        cv.imwrite(mask_output_path, mask)


def force_masks_from_depth(rgb_image_path, depth_image_path, output_folder, threshold=0.95):
    """
    Generate a binary mask using a single depth image.

    Args:
        rgb_image_path (str): Path to the RGB image (used for naming only).
        depth_image_path (str): Path to the corresponding depth image.
        output_path (str): Path to save the generated binary mask.
        threshold (float): Depth threshold (0.0 to 1.0) below which pixels are considered foreground.
    """
    if not os.path.exists(depth_image_path):
        print(f"⚠️ Depth image not found: {depth_image_path}")
        return

    # Load depth image (grayscale)
    depth_img = cv.imread(depth_image_path, cv.IMREAD_GRAYSCALE)
    if depth_img is None:
        print(f"❌ Failed to load: {depth_image_path}")
        return

    # Normalize and threshold
    depth_normalized = depth_img.astype(np.float32) / 255.0
    mask = (depth_normalized < threshold).astype(np.uint8) * 255

    # Use the base name of the RGB image for naming
    base_name = os.path.splitext(os.path.basename(rgb_image_path))[0]
    mask_output_path = os.path.join(output_folder, f"{base_name}_mask.png")

    os.makedirs(output_folder, exist_ok=True)
    cv.imwrite(mask_output_path, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies motion blur to a percentage of dataset images')
    parser.add_argument('folder_path', type=str, help='Add the path to the dataset.') 
    parser.add_argument('-p', '-prob', type=float, help='Add the probability.', dest='prob', default=None)
    args = parser.parse_args()
    folder_path = args.folder_path
    prob = args.prob

    # Validate the folder path
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")

    if prob == None:
        add_motion_blur(folder_path)
    else:
        add_motion_blur(folder_path, prob)
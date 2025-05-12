#
# Script with helper functions for acessing and manipulating json, csv, tsv files and folders or
# any other functions that might be used across other scripts, excluding the labels functions.
#

import os
from typing import Optional, List, Dict, Any
import json
import math
import csv
import random
import pandas as pd
import shutil
import numpy as np
import ast

# ------------- Image Functions ----------------------------------------

def determine_mask_colors(mask):
        """
        Given a binary mask (0 and 255), determine which value is background and which is foreground.
        
        Args:
            mask (np.ndarray): Binary image with pixel values 0 and 255.
            
        Returns:
            background (int): Background pixel value
            foreground (int): Foreground pixel value 
        """
        # Flatten the image and count 0s and 255s
        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        if 0 not in pixel_counts or 255 not in pixel_counts:
            raise ValueError("Mask must contain both 0 and 255 pixel values.")

        # Background is assumed to be the majority class
        if pixel_counts[0] > pixel_counts[255]:
            background, foreground = 0, 255
            return background, foreground
        else:
            background, foreground = 255, 0
            return background, foreground

# -------------- Folder and Subfolder Functions -------------------------------------

def find_image_path(folder_path: str, device: str, mode: str, dictionary):

    """
    Searches for the image name with the given mode in the dictionary and returns its full path.
    
    Args:
        folder_path (str): Path to the folder containing images
        device (str)
        mode (str)
        dictionary (dict)
    
    Returns:
        Optional[str]: Full path to the image if found, None otherwise
    """
    
    try:
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist")
            return None
        
        for img_dict in dictionary[device]:
            if img_dict["mode"] == mode:
                img_file = img_dict["image"]
                return os.path.join(folder_path, img_file), img_file
        
        '''# List all files in the directory
        files = os.listdir(folder_path)
        
        # Look for files that match the pattern: image_name + any image extension
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            
            # Check if the filename matches and the extension is an image extension
            if file_name == image_name and file_ext.lower() in image_extensions:
                # Return the full path to the image
                return os.path.join(folder_path, file)'''
        
        # If we get here, no matching image was found
        print(f"No image found with mode '{mode}' in folder '{folder_path}'")
        return None
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def count_folder_images(folder_path):
    """
    Returns the number of images inside a folder.
    """

    # Get the absolute path
    folder_path = os.path.join(os.path.abspath(os.path.dirname( __file__ )), folder_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")

    # Get list of image files (assuming common image extensions)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]

    # Count images
    total_images = len(image_files)

    return total_images   


def get_random_image(folder_path):
    """
    Given a folder path, returns a random image file from the folder.
    """

    # Get the absolute path
    #folder_path = os.path.join(os.path.abspath(os.path.dirname( __file__ )), folder_path)

    # Get list of image files (assuming common image extensions)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]
    
    # Count images
    total_images = len(image_files)
    
    # Return a random image if available, else None
    random_image = os.path.join(folder_path, random.choice(image_files)) if total_images > 0 else None

    # Extract measurement from filename
    filename = os.path.splitext(random_image)[0]  # Remove extension
    parts = filename.split('_')

    index = 0
    for i in parts:
        try:
            if float(i):
                break
        except:
            index +=1
        
    mode = parts[index-1] if len(parts) >= 3 else None
    #measurements = [float(m) for m in parts[index:]] if len(parts) >= 3 else None
    measurements = [str(m) for m in parts[index:]] if len(parts) >= 3 else None

    #print(parts, mode, measurements)
    
    return random_image, mode, measurements


def rename_images_in_folder(input_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Find the first occurrence of '0' and remove everything after, including the extension
            name, ext = os.path.splitext(file)
            name = name[:-4]
            
            # Ensure new name is not empty and doesn't already exist
            if name and name != file:
                new_name = name + ext  # Keep the original file extension
                new_path = os.path.join(root, new_name)
                if not os.path.exists(new_path):
                    shutil.move(file_path, new_path)
                    #print(f"Renamed: {file} -> {new_name}")


def rename_single_image(folder_path, img_name):
    image_name = f"{img_name}0001.png"
    file_path = os.path.join(folder_path, image_name)

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # Split filename and extension
    name, ext = os.path.splitext(image_name)

    # Trim last 4 characters from name
    new_name = name[:-4] + ext

    # Ensure new name is not empty and not the same as current
    if new_name != image_name and new_name:
        new_path = os.path.join(folder_path, new_name)
        shutil.move(file_path, new_path)
        print(f"Renamed: {image_name} -> {new_name}")
        #else:
        #    print(f"Target file already exists: {new_path}")
    else:
        print("New name is either empty or identical to the original.")

# ------------- Json File Functions ---------------------------------------------

def format_dict(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Formats the input Dictionary to the desired format
    
    Args:
        data (Dict[str, Any]): Dictionary containing the raw data to format and write to the JSON file
        
    Returns:
        Dict: Formatted dictionary
    """

    neg_z_rotation = 0 
    if data['Negative Case Rotation Z'].strip().lower() == "false":
        neg_z_rotation = False
    else:
        neg_z_rotation = math.degrees(float(data['Negative Case Rotation Z']))

    motion_blur = 0
    if data['Motion Blur Kernel Parameters [x, y, size]'].strip().lower() == "false":
        motion_blur = False
    else:
        motion_blur = np.fromstring(data['Motion Blur Kernel Parameters [x, y, size]'][1:-1], dtype=int, sep=',')
    
    # Format the data into the required structure
    formatted_data = {
        "display_color": np.fromstring(data['Display Color'][1:-1], dtype=float, sep=','),
        "settings": {
            "resolution": [1920, 1080],
            "file_format": "PNG"
        },
        "camera": {
            "distance": data['Camera Distance'],
            "shift_x": data['Camera Shift X'],
            "shift_y": data['Camera Shift Y'],
            "focal_length": data['Camera Focal Length']
        },
        "object": {
            "x_rotation": math.degrees(data['Object Rotation X']),
            "y_rotation": math.degrees(data['Object Rotation Y']),
            "z_rotation": math.degrees(data['Object Rotation Z']),
            "negative_case": {
                "z_rotation": neg_z_rotation,
            }
        },
        "light": {
            "color": np.fromstring(data["Light Color"][1:-1], dtype=float, sep=','),
            "energy": data['Light Energy'],
            "falloff": data['Light Falloff'],
            "radius": data['Light Radius'],
            "x_distance": data['Light Distance X'],
            "y_distance": data['Light Distance Y'],
            "z_distance": data['Light Distance Z']
        },
        "motion_blur": motion_blur
    }
            
    print("Dictionary formatted")

    return formatted_data


def load_json(json_file: str, initializer = {}, delete=False):
    """Loads the JSON file into a dictionary. Creates an empty json file in case in doesn't already exist"""
    
    # Remove file if it exists and you want to delete it
    #if (delete==True and os.path.exists(json_file)):
    #    os.remove(json_file)

    if not os.path.exists(json_file):
        with open(json_file, 'w') as file:
            json.dump(initializer, file)

    with open(json_file, 'r') as file:
        return json.load(file)
    

# ------------- CSV File Functions --------------------------------------

def write_to_csv(filename, data, column_titles):
    """
    Writes or updates a dictionary of data to a CSV file.
    
    If the first value in the data already exists in the first column of the file,
    it replaces that row with the new data.
    
    Args:
        filename (str): The name of the CSV file.
        data (dict): A dictionary where keys match the column_titles.
        column_titles (list): A list of column names for the CSV file.
    """
    updated = False
    rows = []

    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_titles)
    
        if not os.path.isfile(filename):
            writer.writeheader()

    if os.path.isfile(filename):
        with open(filename, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # If the first column matches the first value in data, replace it
                if row[column_titles[0]] == str(data[column_titles[0]]):
                    rows.append(data)
                    updated = True
                else:
                    rows.append(row)

    # If no match was found, add the new row
    if not updated:
        rows.append(data)

    # Write all rows back
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_titles)
        writer.writeheader()
        writer.writerows(rows)

def get_image_data(csv_path: str, image_name: str) -> Optional[Dict[str, Any]]:
    """
    Searches for a specific image name in the CSV file and returns all values from that row.
    
    Args:
        csv_path (str): Path to the CSV file
        image_name (str): Name of the image to search for
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary with column names as keys and row values as values,
                                 or None if image name not found
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 'Image' column exists
        if 'Image' not in df.columns:
            print("Error: CSV file does not contain an 'Image' column")
            return None
        
        # Find the row with the matching image name
        matching_row = df[df['Image'] == image_name]
        
        # If no matching row is found, return None
        if matching_row.empty:
            print(f"No image found with name: {image_name}")
            return None
        
        # Convert the first matching row to a dictionary
        row_dict = matching_row.iloc[0].to_dict()
        
        return row_dict
        
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None








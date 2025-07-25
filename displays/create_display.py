import os
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import ast
from itertools import product
from math import prod
import argparse
import sys

# Add the parent directory to the path
current_file = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_folder)

import helper_functions


class DisplayGenerator:
    def __init__(self, input_folder,  dict_folder, font_folder, output_folder):
        script_dir = os.path.abspath(os.path.dirname( __file__ ))
        self.input_path = f"{script_dir}/{input_folder}" 
        self.dict_path = f"{script_dir}/{dict_folder}"
        self.font_path = f"{script_dir}/{font_folder}"
        self.output_path = f"{script_dir}/{output_folder}"
    
    def get_child_folders(self):
        """
        Returns a list of names of each child folder in the input folder.
        """
        try:
            # List all entries in the directory and filter only directories
            return [name for name in os.listdir(self.input_path) 
                    if os.path.isdir(os.path.join(self.input_path, name))]
        except FileNotFoundError:
            print("Error: The specified folder does not exist.")
            return []
        except PermissionError:
            print("Error: Permission denied to access the folder.")
            return []

    def get_measurement_from_file(self, filename):
        with open(f"{self.dict_path}/{filename}", "r") as file:
            numbers = [line.strip() for line in file]
        return random.choice(numbers), numbers
    
    def get_random_font_file(self):
        ttf_files = [f for f in os.listdir(self.font_path) if f.endswith(".ttf")]
        if not ttf_files:
            return None
        return os.path.join(self.font_path, random.choice(ttf_files))
    
    def get_random_image(self, folder_path):
        """
        Selects a random image from the specified folder.
        """
        # List of image file extensions to consider
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        
        # Retrieves a list of files in the folder with valid image extensions
        try:
            files = [f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in image_extensions]
        except FileNotFoundError:
            print(f"The folder '{folder_path}' does not exist.")
            return None

        if not files:
            print("No image files found in the folder.")
            return None

        # Randomly choose an image file
        random_image = random.choice(files)

        return os.path.join(folder_path, random_image), random_image


    def remove_background(self, device, mode, dictionary, csv_mode=False):

        real_display_path = os.path.join(self.input_path, device)


        if csv_mode == False:
            # Load image
            img_path, img_name = helper_functions.find_image_path(real_display_path, device, mode, dictionary)
        else:
            mode = csv_mode
            img_path, img_name = helper_functions.find_image_path(real_display_path, device, mode, dictionary)

        img = cv.imread(img_path)

        # Convert to gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # threshold input image as mask 
        mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
     
        # Cleanup
        cv.destroyAllWindows()

        list_index = -1
        for image in dictionary[device]:
            list_index +=1
            if image["image"] == img_name:
                mode = image["mode"]
                break

        return mask, mode, list_index

    def change_measurement(self, img, device, dictionary, json_path, index, combo, csv_measurement=False):
        # Ensure JSON file exists
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({}, f)

        # Load existing JSON data into both file_data and dictionary
        with open(json_path, "r") as f:
            file_data = json.load(f)

        if "rois" not in dictionary[device][index]:
            dictionary[device][index]["rois"] = []
            
        if device in dictionary and len(dictionary[device][index]["rois"]) != dictionary[device][index]["#roi"]:
            for i in range(dictionary[device][index]["#roi"]):
                x, y, w, h = cv.selectROI("Select ROI", img, fromCenter=False, showCrosshair=False)
                cv.destroyWindow("Select ROI")

                # Create a dictionary with the ROI
                dictionary[device][index]["rois"].append([int(x), int(y), int(w), int(h)])

                file_data[device] = dictionary[device]
        elif device not in dictionary:
            raise Exception('PLEASE ADD DEVICE INFORMATION TO ROI_MAPPINGS.JSON FILE')

        # Save updated data to JSON
        with open(json_path, "w") as f:
            json.dump(file_data, f, indent=4)

        if dictionary[device][index]["#roi"] > 1:
            measurements = []

        if csv_measurement != False: #Measurements from csv come in string format, so they are converted into a list
            csv_measurement = ast.literal_eval(csv_measurement)

        # Get background and foreground pixel values
        background_value, foreground_value = helper_functions.determine_mask_colors(img)

        
        mode = combo[0]
        print('Combo: ', combo)

        for i in range(dictionary[device][index]["#roi"]):
            x, y, w, h = dictionary[device][index]["rois"][i]
            img[y:y+h, x:x+w] = cv.threshold(img[y:y+h, x:x+w], -1, background_value, cv.THRESH_BINARY)[1] #Applies a threshold to the ROI, making its color
                                                                                                           #equal to the background pixel value

            _, all_measurements = self.get_measurement_from_file(filename=dictionary[device][index]["dictionaries"][i] + ".txt")

            measurement = combo[i+1]
            
            if csv_measurement != False:
                if str(csv_measurement[i]) not in all_measurements:
                    print('Chosen measurement not possible in current device.')
                else:
                    measurement = str(csv_measurement[i])

            # Convert openCV image to PIL Image
            img = Image.fromarray(img)

            # Choose a font.
            font_path = self.get_random_font_file()

            if not os.path.exists(font_path):
                print("Error: Font file not found!")

            # Dynamically Adjust Font Size
            font_size = 1  # Start with a small font size
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = font.getbbox(measurement)[2:]  # Get text size

            # Increase font size until text fits inside ROI
            while text_width < w * 0.99 and text_height < h * 0.99:  # Keep a margin
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
                text_width, text_height = font.getbbox(measurement)[2:]

            # Center the text in the ROI
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2

            # Draw the measurement on the image
            draw = ImageDraw.Draw(img)
            draw.text((text_x, text_y), measurement, font=font, fill=foreground_value)  # Fill the text with the foreground value

            # Convert back to OpenCV format
            img = np.array(img)

            if dictionary[device][index]["#roi"] > 1:
                measurements.append(measurement)
            else:
                measurements = [measurement]


        measurements_name = "_".join(m for m in measurements)
        
        if csv_measurement != False:
            measurements = [float(m) for m in measurements]

        # Ensure the directory exists before saving the image
        device_folder = f"{self.output_path}/{device}"
        os.makedirs(device_folder, exist_ok=True)

        display_image_path = f"{device_folder}/{device}_{mode}_{measurements_name}.png"

        cv.imwrite(display_image_path, img)

        cv.waitKey(0)

        return display_image_path, measurements

    def maximum_display_images(self, device, roi_data, display_number):
        """
        Returns the maximum number of unique display images that can be generated for a device, in case it is smaller than the display_number.
        If larger, it simply outputs the display number
        A unique image is defined by its mode and measurement values (no repetition).
        """
        
        if device not in roi_data:
            raise ValueError(f"No data found for device: {device}")

        seen_combinations = set()
        max_count = 0

        for mode_info in roi_data[device]:
            mode = mode_info["mode"]
            dict_names = mode_info["dictionaries"]

            # Load all values from each dictionary
            dict_values_list = []
            for dict_name in dict_names:
                dict_file = os.path.join(self.dict_path, f"{dict_name}.txt")
                with open(dict_file, "r") as f:
                    values = [line.strip() for line in f if line.strip()]
                    dict_values_list.append(values)

            # Generate all combinations of measurements for this mode
            for combination in product(*dict_values_list):
                # Tuple of mode and measurements ensures uniqueness
                unique_id = (mode, tuple(combination))
                if unique_id not in seen_combinations:
                    seen_combinations.add(unique_id)
                    max_count += 1

                    if max_count >= display_number:
                        return display_number, False
                    else:
                        continue

        return max_count, True
            

    def generate_random_unique_image(self, device, display_number, actual_max, roi_json_path, used_path="used_combinations.json"):
    
        roi_data = helper_functions.load_json(roi_json_path)

        if device not in roi_data:
            raise ValueError(f"No ROI data found for device: {device}")
        
        used_combos = helper_functions.load_json(used_path)
        device_used = set(tuple(t) for t in used_combos.get(device, []))

        if actual_max==True: # All combinations for current device are to be generated

            for mode_info in roi_data[device]:
                mode = mode_info["mode"]
                dict_names = mode_info["dictionaries"]

                dict_values_list = []
                for dict_name in dict_names:
                    with open(os.path.join(self.dict_path, f"{dict_name}.txt"), "r") as f:
                        values = [line.strip() for line in f if line.strip()]
                        dict_values_list.append(values)

                # Shuffle for random sampling
                all_combos = list(product(*dict_values_list))
                random.shuffle(all_combos)

                for combo in all_combos:
                    combo_key = (mode, *combo)
                    if combo_key not in device_used:
                        # --- Image generation logic  ---
                        img, _, list_index = self.remove_background(device=device, mode=mode, dictionary=roi_data)
                        self.change_measurement(img, device=device, dictionary=roi_data, json_path=roi_json_path, 
                                                index=list_index, combo=combo_key)
            

                        # Save the combination
                        used_combos.setdefault(device, []).append(combo_key)

                        with open(used_path, "w") as f:
                            json.dump(used_combos, f, indent=4)

                        return True
                    
        else:
            generated_combos = set(device_used)
            attempts = 0
            max_attempts = display_number * 10  # Prevent infinite loop if not enough unique combos

            while len(generated_combos) < display_number and attempts < max_attempts:
                mode = random.choice(roi_data[device])["mode"]
                dict_names = [info["dictionaries"] for info in roi_data[device] if info["mode"] == mode][0]

                combo = [random.choice(open(os.path.join(self.dict_path, f"{name}.txt")).read().splitlines())
                        for name in dict_names]

                combo_key = (mode, *combo)

                if combo_key not in generated_combos:
                    generated_combos.add(combo_key)

                    # Proceed with image generation
                    img, _, list_index = self.remove_background(device=device, mode=mode, dictionary=roi_data)
                    self.change_measurement(img, device=device, dictionary=roi_data, json_path=roi_json_path,
                                            index=list_index, combo=combo_key)

                    # Save combination
                    used_combos.setdefault(device, []).append(combo_key)
                    with open(used_path, "w") as f:
                        json.dump(used_combos, f, indent=4)

                    return True

                attempts += 1

        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates OCR Ready Dataset')
    parser.add_argument('-display_number', type=int, help='Define number of display images per device.', dest='display_number', default=10)
    args = parser.parse_args()
        
    display_generator = DisplayGenerator(input_folder='images/real', dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    device_list = display_generator.get_child_folders()

    roi_filepath = os.path.abspath("displays/roi_mappings.json")
    roi_dictionary = helper_functions.load_json(roi_filepath)
    used_combinations_path = 'displays/images/generated/used_combinations.json'

    if os.path.exists(used_combinations_path):
        os.remove(used_combinations_path)

    for device_name in device_list:
        image_number = helper_functions.count_folder_images(f'displays/images/generated/{device_name}')
        max_image_number, is_actual_max = display_generator.maximum_display_images(device=device_name, roi_data=roi_dictionary, display_number=args.display_number)
        if image_number >= args.display_number or image_number >= max_image_number:
            print(f'The {device_name} display images are already generated. Advancing to next device...')
        else:
            print(f'Generating {device_name} display images:')
            
            generated = image_number
            while generated < max_image_number:
                success = display_generator.generate_random_unique_image(device=device_name, display_number=max_image_number, actual_max=is_actual_max, 
                                                                         roi_json_path=roi_filepath, used_path=used_combinations_path)
                
                if success:
                    generated += 1
                else:
                    break

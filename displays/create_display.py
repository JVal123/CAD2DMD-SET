import os
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import ast
import helper_functions
from itertools import product
from math import prod


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
        #image_name = random_image.split(".")[0]

        return os.path.join(folder_path, random_image), random_image


    def remove_background(self, device, mode, dictionary, csv_mode=False):

        real_display_path = os.path.join(self.input_path, device)


        if csv_mode == False:
            # Load image
            #img_path, img_name = self.get_random_image(real_display_path)
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

        #print(dictionary[device][index])

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
        #print(img)
        background_value, foreground_value = helper_functions.determine_mask_colors(img)

        
        mode = combo[0]
        print('Combo: ', combo)
        #print('Mode: ', mode)

        for i in range(dictionary[device][index]["#roi"]):
            x, y, w, h = dictionary[device][index]["rois"][i]
            img[y:y+h, x:x+w] = cv.threshold(img[y:y+h, x:x+w], -1, background_value, cv.THRESH_BINARY)[1] #Applies a threshold to the ROI, making its color
                                                                                                           #equal to the background pixel value

            _, all_measurements = self.get_measurement_from_file(filename=dictionary[device][index]["dictionaries"][i] + ".txt")

            measurement = combo[i+1]
            #print(measurement)

            #print("Current dictionary: ", dictionary[device][index]["dictionaries"][i] + ".txt")
            
            if csv_measurement != False:
                if str(csv_measurement[i]) not in all_measurements:
                    print('Chosen measurement not possible in current device.')
                else:
                    measurement = str(csv_measurement[i])
            
            #print('Measurement: ', measurement)

            # Convert openCV image to PIL Image
            img = Image.fromarray(img)

            # Choose a font.
            font_path = self.get_random_font_file()
            #print("Checking font path:", font_path)

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



        # Show the image
        #cv.imshow("Final Image:", opencv_img)

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

                #print('Before getting the combos')

                # Shuffle for random sampling
                all_combos = list(product(*dict_values_list))
                #print('All combos: ', all_combos)
                #print('Device Used: ', device_used, len(device_used))
                random.shuffle(all_combos)

                #print('Got the combos', len(all_combos))

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



            '''modes_amount = len(roi_data[device])
            mode_number = random.randint(0, modes_amount-1)
            mode = roi_data[device][mode_number]["mode"]
            dict_names = roi_data[device][mode_number]["dictionaries"]

            #print('Mode: ', mode)
            #print('Dict_names: ', dict_names)


            dict_values_list = []
            for dict_name in dict_names:
                with open(os.path.join(self.dict_path, f"{dict_name}.txt"), "r") as f:
                    values = [line.strip() for line in f if line.strip()]
                    dict_values_list.append(values)

            print('Before getting the combos')

            # Shuffle for random sampling
            #all_combos = list(product(*dict_values_list))
            #print('All combos: ', all_combos)
            #print('Device Used: ', device_used, len(device_used))
            #random.shuffle(all_combos)

            # Step 1: Compute sizes and total number of combinations
            sizes = [len(lst) for lst in dict_values_list]
            total_combinations = prod(sizes)

            # Step 2: Limit the number of combos
            num_samples = min(display_number, total_combinations)
            indices = np.random.choice(total_combinations, size=num_samples, replace=False)

            print('Got the combos')

            # Step 3: Map indices to combos
            for idx in indices:
                multi_idx = np.unravel_index(idx, sizes)
                combo = tuple(dict_values_list[i][j] for i, j in enumerate(multi_idx))
                combo_key = (mode, *combo)

            

            #for combo in all_combos:
            #    combo_key = (mode, *combo)
                if combo_key not in device_used:
                    # --- Your image generation logic goes here ---
                    img, _, list_index = self.remove_background(device=device, mode=mode, dictionary=roi_data)
                    self.change_measurement(img, device=device, dictionary=roi_data, json_path=roi_json_path, 
                                            index=list_index, combo=combo_key)
        

                    # Save the combination
                    used_combos.setdefault(device, []).append(combo_key)

                    with open(used_path, "w") as f:
                        json.dump(used_combos, f, indent=4)

                    return True'''
                    
        
        return False


if __name__ == "__main__":
    number = 10
    thresh_value, img_height, img_width = 0, 0, 0
 
    generator = DisplayGenerator(input_folder='images/real', dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    device_list = generator.get_child_folders()

    for device_name in device_list:
        print(f'Generating {device_name} display images:')

        #for i in range(0, number):
          #  img, _, _, _, mode = generator.remove_background(thresh_value, img_height, img_width, device=device_name, select_threshold=False)
           # generator.change_measurement(img, device=device_name, mode=mode, dictionary=roi_json, json_path=)

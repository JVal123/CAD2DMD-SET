import os
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import ast
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
        #image_name = random_image.split(".")[0]

        return os.path.join(folder_path, random_image), random_image




    def remove_background(self, device, dictionary, csv_mode=False):

        real_display_path = os.path.join(self.input_path, device)


        if csv_mode == False:
            # Load image
            img_path, img_name = self.get_random_image(real_display_path)
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

    def change_measurement(self, img, device, mode, dictionary, json_path, index, csv_measurement=False):
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

        for i in range(dictionary[device][index]["#roi"]):
            x, y, w, h = dictionary[device][index]["rois"][i]
            img[y:y+h, x:x+w] = cv.threshold(img[y:y+h, x:x+w], -1, background_value, cv.THRESH_BINARY)[1] #Applies a threshold to the ROI, making its color
                                                                                                           #equal to the background pixel value

            measurement, all_measurements = self.get_measurement_from_file(filename=dictionary[device][index]["dictionaries"][i] + ".txt")

            #print("Current dictionary: ", dictionary[device][index]["dictionaries"][i] + ".txt")
            
            if csv_measurement != False:
                if str(csv_measurement[i]) not in all_measurements:
                    print('Chosen measurement not possible in current device.')
                else:
                    measurement = str(csv_measurement[i])
            
            print('Measurement: ', measurement)

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

        #time.sleep(0.5)

        cv.waitKey(0)

        return display_image_path, measurements



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

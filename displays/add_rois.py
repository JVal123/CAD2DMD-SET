import os
import cv2 as cv
import json
import create_display
import sys

# Add the parent directory to the path
current_file = os.path.abspath(__file__)
parent_folder = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_folder)

import helper_functions

def add_rois(img, device, dictionary, json_path, index):

    new_key = "rois"

    # Ensure JSON file exists
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump({}, f)

    # Load existing JSON data into both file_data and dictionary
    with open(json_path, "r") as f:
        file_data = json.load(f)

    try:
        ocr_field = dictionary[device][index]["ocr"]
    except:
        print(f"The {device} doesn't have additional rois. Skipping this device...")
        return


    if new_key not in ocr_field:
        ocr_field[new_key] = []
    
    ocr_rois = ocr_field[new_key]
    ocr_labels = ocr_field["labels"]
        
    if device in dictionary and len(ocr_rois) != len(ocr_labels):

        for i in range(len(ocr_labels)):
            x, y, w, h = cv.selectROI("Select Extra ROI", img, fromCenter=False, showCrosshair=False)
            cv.destroyWindow("Select Extra ROI")

            # Create a dictionary with the ROI
            ocr_rois.append([int(x), int(y), int(w), int(h)])

            file_data[device] = dictionary[device]

    elif device not in dictionary:
        raise Exception('PLEASE ADD DEVICE INFORMATION TO ROI_MAPPINGS.JSON FILE')

    # Save updated data to JSON
    with open(json_path, "w") as f:
        json.dump(file_data, f, indent=4)



if __name__ == "__main__":

    real_images_path = 'images/real'
    display_generator = create_display.DisplayGenerator(input_folder=real_images_path, dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    device_list = display_generator.get_child_folders()

    roi_filepath = os.path.abspath("roi_mappings.json")
    roi_dictionary = helper_functions.load_json(roi_filepath)
    used_combinations_path = 'images/generated/used_combinations.json'

    if os.path.exists(used_combinations_path):
        os.remove(used_combinations_path)

    for device_name in device_list:
        indexes = len(roi_dictionary[device_name])
        for idx in range(0, indexes):
            image_name = roi_dictionary[device_name][idx]["image"]
            image_path = os.path.join(real_images_path, device_name, image_name)
            img = cv.imread(image_path)
            add_rois(img, device=device_name, dictionary=roi_dictionary, json_path=roi_filepath, index=idx)

    #remove_zero_rois(roi_filepath)

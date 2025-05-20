#
# Script with label generation functions
#

import helper_functions
import csv
import ast
import json
import pandas as pd
import os
import sys
import numpy as np
import re

# Add the parent directory to the path
vlmevalkit_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLMEvalKit/vlmeval/smp/'))

sys.path.append(vlmevalkit_folder)

from vlm import encode_image_file_to_base64



def get_qa_pair(device_type, mode, value, json_dict, random_generator):

    device_dicts = json_dict[device_type]
    #print(len(device_dicts))
    for index in range(len(device_dicts)):
        if device_dicts[index]["mode"] == mode:
            break

    labels = device_dicts[index]["labels"]
    measurement_type = labels["measurement_type"]
    unit = labels["unit"]

    #print('Device type: ', device_type)
    #print('Measurement Type: ', measurement_type, len(measurement_type), type(measurement_type))
    #print('Labels: ', labels)
    #print('Value: ', value, len(value), type(value))
    #print('Unit: ', unit, len(unit), type(unit))

    if len(measurement_type)==1 and len(unit)==1 and len(value)==1: #1 measurement type and 1 unit qa pairs

        measurement_type = measurement_type[0]
        value = value[0]
        unit = unit[0]
        full_value = f"{value} {unit}"

        mode_parts = mode.rsplit(' ', 1) #Get the mode, without its last word

        if len(mode_parts) == 2:
            first_part, _ = mode_parts
        else:
            first_part = mode_parts[0]
        

        if first_part != measurement_type: #Ambiguious cases (metronome)

            #print('Ambiguous Case...')

            qa_pairs = [
                {
                    "question": f"Identify the device and explain the relationship between the display mode and the measurement shown.",
                    "answer": f"The device is a digital {device_type}. Although the mode is set to '{mode}', the screen displays a {measurement_type} of {full_value}."
                },
                {
                    "question": f"What is the mode shown on the device, and what is the actual value being measured or displayed?",
                    "answer": f"The mode shown is '{mode}', but the displayed value is {full_value}, which corresponds to {measurement_type}."
                },
                {
                    "question": f"Describe the digital display including any mode label and the numerical reading.",
                    "answer": f"The screen shows the mode label '{mode}', and a measurement of {full_value} is displayed."
                },
                {
                    "question": f"What value is shown on the device, and how does it relate to the mode displayed?",
                    "answer": f"The value shown is {full_value}, which measures {measurement_type}, despite the mode being '{mode}'."
                },
                {
                    "question": f"What information is being measured, and what unrelated mode label is also present on the display?",
                    "answer": f"The device measures {measurement_type} and displays {full_value}, but the mode label reads '{mode}', which is unrelated to the measurement."
                },
                {
                    "question": f"Is the mode label on the display related to the actual measurement? Explain.",
                    "answer": f"No, the mode label '{mode}' is unrelated to the measurement, which is {full_value} ({measurement_type})."
                },
                {
                    "question": f"What does the display show, and what does the mode label indicate?",
                    "answer": f"The display shows a value of {full_value} for {measurement_type}, while the mode label indicates '{mode}'."
                },
                {
                    "question": f"Does the label on the device indicate what is being measured?",
                    "answer": f"No, the label '{mode}' does not reflect the measurement, which is {measurement_type} at {full_value}."
                }
            ]
        
        else:

            qa_pairs = [
                {
                    "question": "Identify the digital measurement device and report its reading with units.",
                    "answer": f"The image shows a digital {device_type} displaying a reading of {full_value}."
                },
                {
                    "question": "What does the device measure, and what is the current value?",
                    "answer": f"The device measures {measurement_type} and currently shows {full_value}."
                },
                {
                    "question": "What is the measurement shown on the device in this image?",
                    "answer": f"The display shows a measurement of {full_value}."
                },
                {
                    "question": "What is the function of the device and what is it currently displaying?",
                    "answer": f"The device functions as a {device_type} and is displaying {full_value}."
                },
                {
                    "question": "Describe the device, its current mode, and the reading it displays.",
                    "answer": f"It is a digital {device_type} in standard mode, displaying a {measurement_type} of {full_value}."
                },
                {
                    "question": "What value is shown on the digital display, and what does it represent?",
                    "answer": f"The digital display shows {full_value}, which represents a {measurement_type} measurement."
                },
                {
                    "question": "Which digital device is visible, and what measurement is it currently showing?",
                    "answer": f"A digital {device_type} is visible, showing a reading of {full_value}."
                },
                {
                    "question": "What kind of digital device is shown, and what is its reading?",
                    "answer": f"The image shows a digital {device_type} with a reading of {full_value}."
                },
                {
                    "question": "What does the screen of the digital device display?",
                    "answer": f"The screen displays a {measurement_type} reading of {full_value}."
                },
                {
                    "question": "Identify the device and summarize what it is displaying.",
                    "answer": f"The device is a digital {device_type}, and it is displaying {full_value}."
                },
                {
                    "question": "What reading is displayed on the screen, and what is being measured?",
                    "answer": f"The screen shows a value of {full_value}, which is a {measurement_type} measurement."
                },
                {
                    "question": "What is the device measuring, and what value does it report?",
                    "answer": f"It is measuring {measurement_type} and reports a value of {full_value}."
                },
                {
                    "question": "Describe what is being measured and the result shown on the device display.",
                    "answer": f"The device is measuring {measurement_type}, and the display shows {full_value}."
                },
                {
                    "question": "What measurement is visible on the device, and what might it indicate?",
                    "answer": f"The device shows a {measurement_type} of {full_value}."
                },
                {
                    "question": "What information is being conveyed by the digital display in this image?",
                    "answer": f"The digital display conveys a {measurement_type} reading of {full_value}."
                }
            ]

    elif len(measurement_type)==2 and len(unit)==2 and len(value)==2: #2 measurement types with 2 different units qa pairs
        
        m1_type, m1_val, m1_unit = measurement_type[0], value[0], unit[0]
        m2_type, m2_val, m2_unit = measurement_type[1], value[1], unit[1]

        qa_pairs = [
            {
                "question": f"Identify the digital measurement device and report both readings with their units.",
                "answer": f"The image shows a digital {device_type} displaying {m1_val} {m1_unit} for {m1_type} and {m2_val} {m2_unit} for {m2_type}."
            },
            {
                "question": f"What are the measurements shown on the device, and what do they represent?",
                "answer": f"The device shows {m1_val} {m1_unit} ({m1_type}) and {m2_val} {m2_unit} ({m2_type})."
            },
            {
                "question": f"What two measurement types are displayed, and what are their values and units?",
                "answer": f"{m1_type.capitalize()}: {m1_val} {m1_unit}, {m2_type.capitalize()}: {m2_val} {m2_unit}."
            },
            {
                "question": f"What values are shown on the device, and what do they each correspond to?",
                "answer": f"{m1_val} {m1_unit} corresponds to {m1_type}, and {m2_val} {m2_unit} corresponds to {m2_type}."
            },
            {
                "question": f"Describe the device and the two different measurements it shows.",
                "answer": f"The digital {device_type} shows two readings: {m1_val} {m1_unit} ({m1_type}) and {m2_val} {m2_unit} ({m2_type})."
            },
            {
                "question": f"What does the device measure, and what are the current values?",
                "answer": f"It measures {m1_type} and {m2_type}, showing {m1_val} {m1_unit} and {m2_val} {m2_unit}, respectively."
            },
            {
                "question": f"What information is presented on the display across different measurement types?",
                "answer": f"The display shows two types of measurements: {m1_val} {m1_unit} for {m1_type}, and {m2_val} {m2_unit} for {m2_type}."
            },
            {
                "question": f"What are the displayed measurement types and corresponding readings?",
                "answer": f"The device displays {m1_type} as {m1_val} {m1_unit} and {m2_type} as {m2_val} {m2_unit}."
            }
        ]

    elif len(measurement_type)==3 and len(unit)==2 and len(value)==3: #3 measurement types with 3 different units qa pairs (bllod pressure case)
            
            m1_type, m1_value, m1_unit = measurement_type[0], value[0], unit[0]
            m1_full = f"{m1_value} {m1_unit}"

            m2_type, m2_value, m2_unit = measurement_type[1], value[1], unit[1]
            m2_full = f"{m2_value} {m2_unit}"

            m3_type, m3_value, m3_unit = measurement_type[2], value[2], unit[1]
            m3_full = f"{m3_value} {m3_unit}"


            qa_pairs = [
                {
                    "question": "Identify the device and report all measurement values with their types and units.",
                    "answer": f"The image shows a {device_type} displaying {m1_full} ({m1_type}), {m2_full} ({m2_type}), and {m3_full} ({m3_type})."
                },
                {
                    "question": "What three measurements are displayed on the device, and what units are used?",
                    "answer": f"The device shows {m1_full} for {m1_type}, and {m2_full} and {m3_full} for {m2_type} and {m3_type}, respectively."
                },
                {
                    "question": "List all measurement types, values, and units shown on the display.",
                    "answer": f"{m1_type.capitalize()}: {m1_full}, {m2_type.capitalize()}: {m2_full}, {m3_type.capitalize()}: {m3_full}."
                },
                {
                    "question": "What value is shown for each measurement, and which units are used?",
                    "answer": f"{m1_type.capitalize()} is {m1_full}; {m2_type} is {m2_full}; and {m3_type} is {m3_full}."
                },
                {
                    "question": "Describe the measurements shown, including the one with a different unit.",
                    "answer": f"The device displays {m1_type} as {m1_full}, using a different unit, and both {m2_type} and {m3_type} in {unit[1]}."
                },
                {
                    "question": "Which measurement uses a different unit from the others, and what are the values?",
                    "answer": f"{m1_type.capitalize()} uses a different unit ({m1_unit}): {m1_value}. The other measurements are {m2_full} and {m3_full}."
                },
                {
                    "question": "What information is shown on the device across three measurement types?",
                    "answer": f"The device shows {m1_type}: {m1_full}, {m2_type}: {m2_full}, and {m3_type}: {m3_full}."
                },
                {
                    "question": "Are there multiple measurement types displayed with different units? Provide details.",
                    "answer": f"Yes, the device shows {m1_type} in {m1_unit} ({m1_value}), and {m2_type} and {m3_type} in {m2_unit} ({m2_value} and {m3_value})."
                }
            ]


    elif len(measurement_type)==1 and len(unit)==2 and len(value) == 2: #1 measurement type and 2 different units qa pairs
        measurement_type = measurement_type[0]
        values_formatted = []
        for i in range(2):
            values_formatted.append(f"{value[i]} {unit[i]}") 

        values_joined = " and ".join(values_formatted)
        units_joined = " and ".join(unit)


        qa_pairs = [
            {
                "question": f"Identify the digital measurement device and report the {measurement_type} it shows in both units.",
                "answer": f"The image shows a digital {device_type} displaying {values_joined}."
            },
            {
                "question": f"What are the {measurement_type} readings displayed on the device?",
                "answer": f"The device shows {values_joined}."
            },
            {
                "question": f"What values are displayed on the device, and what do they represent?",
                "answer": f"The display shows {values_joined}, both representing the same {measurement_type} in different units."
            },
            {
                "question": f"Describe the readings shown by the {device_type}, including all units.",
                "answer": f"The {device_type} shows a {measurement_type} of {values_joined}."
            },
            {
                "question": f"What measurement type is shown in multiple units, and what are the values?",
                "answer": f"{measurement_type.capitalize()} is shown in two units: {values_joined}."
            },
            {
                "question": f"What does the digital {device_type} display, and how are the units presented?",
                "answer": f"The {device_type} displays a {measurement_type} of {values_joined}."
            },
            {
                "question": f"What are the two {measurement_type} values shown on the device, and which units do they use?",
                "answer": f"The device shows two {measurement_type} values: {values_joined}."
            },
            {
                "question": f"Identify the {measurement_type} measurement and list all units present.",
                "answer": f"The device measures {measurement_type} and shows values in {units_joined}."
            }
        ]

    #print(qa_pairs)
    random_index = random_generator.integers(low=0, high=len(qa_pairs))
    #print(random_index)
    chosen_qa_pair = qa_pairs[random_index]
    #print(chosen_qa_pair)

    return chosen_qa_pair


def generate_labels(csv_filepath, mappings_json, labels_json, random_generator):

    # Load json files
    mappings_dict = helper_functions.load_json(mappings_json)
    labels_list = helper_functions.load_json(labels_json, initializer=[])

    if not isinstance(labels_list, list):
        labels_list = []

    columns_to_extract = ["Image", "Device", "Mode", "Measurement"]

    with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            extracted_data = {col: row[col] for col in columns_to_extract if col in row}

            image = extracted_data["Image"]
            device_type = extracted_data["Device"]
            mode = extracted_data["Mode"]

            try:
                value = ast.literal_eval(extracted_data["Measurement"])
            except:
                value = extracted_data["Measurement"]
                cleaned_value = value.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"') # Replace curly quotes with straight ones
                value = ast.literal_eval(cleaned_value)

            qa_pair = get_qa_pair(device_type, mode, value, mappings_dict, random_generator)

            label_entry = {
                "image": image,
                "question": qa_pair['question'],
                "answer": qa_pair['answer']
            }

            labels_list.append(label_entry)

    # Save all labels after the loop
    with open(labels_json, 'w') as file:
        json.dump(labels_list, file, indent=4)


def generate_training_labels(training_csv, foreground_labels_list, training_labels_path):
    
    # Load the training labels JSON file (delete if it already exists)
    training_labels_list = helper_functions.load_json(training_labels_path, initializer=[])

    if not isinstance(training_labels_list, list):
        training_labels_list = []

    columns_to_extract = ["Composite", "Foreground"]

    with open(training_csv, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            extracted_data = {col: row[col] for col in columns_to_extract if col in row}

            comp_img_name = os.path.splitext(extracted_data["Composite"])[0]
            foreground_name = os.path.splitext(extracted_data["Foreground"])[0]

            match = re.search(r'\d+', foreground_name)
            foreground_number = int(match.group())
        
            label = foreground_labels_list[foreground_number-1]

            if label['image'] == foreground_name:
                new_entry = {
                    "image": comp_img_name,
                    "question": label['question'],
                    "answer": label['answer']
                }
                training_labels_list.append(new_entry)
            else:
                raise Exception("Foreground labels wrongly built")
    # Extract the base name of the foreground image (e.g., "img1" from "path/to/img1.png")
    #foreground_basename = os.path.splitext(os.path.basename(pair['foreground']))[0]

    # Find and append the matching QA pair
    '''for label in foreground_labels_list:
        if label['image'] == foreground_basename:
            new_entry = {
                "image": comp_img_name,
                "question": label['question'],
                "answer": label['answer']
            }
            training_labels_list.append(new_entry)
            break'''

    # Save the updated training labels back to the JSON file
    with open(training_labels_path, 'w') as f:
        json.dump(training_labels_list, f, indent=4)


def labels_in_tsv(json_path, image_dir, tsv_path):    
    """
    Converts a JSON label file to a TSV format with base64-encoded images.
    
    Args:
        json_path (str): Path to the input labels.json
        image_dir (str): Path to the directory containing image files
        tsv_path (str): Path where the output TSV will be saved
    """
   
   # Load the JSON data
    data = helper_functions.load_json(json_path)
    
    rows = []
    for idx, item in enumerate(data, start=1):
        image_filename = os.path.join(image_dir, item["image"] + ".png")  # Modify extension if needed
        try:
            image_base64 = encode_image_file_to_base64(image_filename)
        except Exception as e:
            print(f"Error encoding image {image_filename}: {e}")
            image_base64 = ""

        rows.append({
            "index": idx,
            "image": image_base64,
            "question": item["question"],
            "answer": item["answer"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(tsv_path, sep='\t', index=False)
    

if __name__=="__main__":

    # --------- Creating TSV file for VLMEvalKit -----------------------------

    '''labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    json_path = os.path.join(labels_path, 'labels.json')
    tsv_path = os.path.join(labels_path, 'test_labels.tsv')
    image_dir = os.path.join(labels_path, 'foreground')

    print('Creating tsv labels version...')
 
    labels_in_tsv(json_path, image_dir, tsv_path, is_test_split=True)

    print('Stage Complete!')'''

    # --------- Creating test set labels -----------------------------

    csv_file = "test.csv" 
    roi_filepath = os.path.abspath("displays/roi_mappings.json")

    rng = np.random.default_rng(seed=42)

    print("Creating test set labels json file... ")
    generate_labels(csv_file, mappings_json=roi_filepath, labels_json='test.json', random_generator=rng)
    print("Test json file created! ")

    print("Creating test set labels tsv file... ")
    labels_in_tsv(json_path='test.json', image_dir='test_set', tsv_path='test.tsv')

    print("Label Generation Stage Complete! ✅")
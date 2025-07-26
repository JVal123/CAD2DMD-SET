import json
import uuid
from pathlib import Path
import argparse
import os


def convert_dataset_for_llava(input_path, output_path, image_folder):
    """
    Converts a custom dataset with fields `image`, `question`, and `answer`
    to the LLaVA fine-tuning format with `conversations`, assuming .png images.
    Renames the image to match the generated example ID.
    
    Args:
        input_path (str): Path to input JSON file.
        output_path (str): Path to save the converted JSON file.
        image_folder (str): Relative path to the folder where images are stored.
    """

    if not output_path.lower().endswith('.json'):
        raise ValueError("The output path must be a JSON file (with a .json extension).")

    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    converted_data = []
    for entry in original_data:
        example_id = str(uuid.uuid4())
        old_image_path = os.path.join(image_folder, entry['image'] + ".png")
        new_image_filename = example_id + ".png"
        new_image_path = os.path.join(image_folder, new_image_filename)

        # Rename the image file to match the example ID
        os.rename(old_image_path, new_image_path)

        conversation = {
            "id": example_id,
            "image": new_image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + entry["question"]
                },
                {
                    "from": "gpt",
                    "value": entry["answer"]
                }
            ]
        }
        converted_data.append(conversation)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Create Json File for LLaVA Fine-tuning')
    parser.add_argument('-input_labels', type=str, help='Add the path to the training labels json file.', dest="input_path", default=None)
    parser.add_argument('-output_labels', type=str, help='Add the path to the output folder.', dest="output_path", default=None)
    parser.add_argument('-image_dir', type=str, help='Add the path to the image_folder.', dest="image_folder", default=None)
    args = parser.parse_args()
    input_path = args.input_labels
    output_path = args.output_labels
    image_folder = args.image_dir


    convert_dataset_for_llava(input_path, output_path, image_folder)

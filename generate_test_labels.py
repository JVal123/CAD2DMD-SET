import os
import json
import pandas as pd
import argparse
import sys
import labeler
import numpy as np

# Add the parent directory to the path
vlmevalkit_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLMEvalKit/vlmeval/smp/'))

sys.path.append(vlmevalkit_folder)

from vlm import encode_image_file_to_base64


def labels_in_tsv(json_path, image_dir, tsv_path):    
    """
    Converts a JSON label file to a TSV format with base64-encoded images.
    
    Args:
        json_path (str): Path to the input labels.json
        image_dir (str): Path to the directory containing image files
        tsv_path (str): Path where the output TSV will be saved
    """
   
   # Load the JSON data
    if not os.path.exists(json_path):
        with open(json_path, 'w') as file:
            json.dump({}, file)

    with open(json_path, 'r') as file:
        data= json.load(file)

    
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test set labels in tsv format, ready for VLMEvalKit.")
    parser.add_argument('--mode', choices=['full_answers', 'short_answers', 'one_word'], required=True)
    parser.add_argument("--input_csv", help="Path to test set csv file")
    parser.add_argument("--image_dir", help="Path to test set image directory")
    parser.add_argument("--test_labels_json", help="Path to output test labels json file")
    parser.add_argument("--tsv_path", help="Path to output test set labels in tsv format")
    args = parser.parse_args()

    #labels_in_tsv(args.input_labels, args.image_dir, args.tsv_path)

    # --------- Creating test set labels ----------------------------- 
    roi_filepath = os.path.abspath("displays/roi_mappings.json")
    rng = np.random.default_rng(seed=42)

    print("Creating test set labels json file... ")
    if args.mode == "one_word":
        labeler.generate_labels(args.input_csv, mappings_json=roi_filepath, labels_json=args.test_labels_json, random_generator=rng, one_word_answers=True)
    elif args.mode == "short_answers":
        labeler.generate_labels(args.input_csv, mappings_json=roi_filepath, labels_json=args.test_labels_json, random_generator=rng, short_answers=True)
    else:
        labeler.generate_labels(args.input_csv, mappings_json=roi_filepath, labels_json=args.test_labels_json, random_generator=rng)

    labeler.remove_underscores_from_device_names_in_file(input_file=args.test_labels_json)
    print("Test json file created! ")

    print("Creating test set labels tsv file... ")
    labels_in_tsv(json_path=args.test_labels_json, image_dir=args.image_dir, tsv_path=args.tsv_path)

    print("Label Generation Stage Complete! ✅")

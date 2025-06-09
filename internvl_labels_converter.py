import os
import json
from PIL import Image
import argparse

def convert_to_internvl_single_image_format(input_json_path, output_jsonl_path, image_dir="dataset/results/training_set"):
    # Load the original JSON data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(data):
            image_filename = f"{item['image']}.png"
            image_path = os.path.join(image_dir, image_filename)

            # Default width and height if image not found
            width, height = 800, 600

            # Try to read actual image size
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Warning: Could not read image {image_path}. Using default size. Error: {e}")
            else:
                print(f"Warning: Image not found at {image_path}. Using default size.")

            entry = {
                "id": idx,
                "image": f"{image_dir}/{image_filename}",
                "width": width,
                "height": height,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": item['answer']
                    }
                ]
            }

            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Converted {len(data)} entries to {output_jsonl_path} with real image dimensions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate InternVL ready labels from CAD2DMD-SET label format.")
    parser.add_argument("--input_labels", help="Path to input training labels")
    parser.add_argument("--output_labels", help="Path to output labels in InternVL ready format (JSONL)")
    parser.add_argument("--image_dir", help="Path to training set image directory")
    args = parser.parse_args()

    convert_to_internvl_single_image_format(args.input_labels, args.output_labels, args.image_dir)
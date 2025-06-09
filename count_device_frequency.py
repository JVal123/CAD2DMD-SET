import csv
import os
from collections import Counter
import argparse

def count_device_frequencies(training_csv, foreground_csv):
    """
    Counts the frequency of each device type in the training CSV based on entries in the foreground CSV.

    Args:
        training_csv (str): Path to the training CSV with 'Foreground' column.
        foreground_csv (str): Path to the foreground CSV with 'Image' and 'Device' columns.

    Returns:
        dict: A dictionary with device types as keys and their frequency as values.
    """
    # Load foreground metadata into a dictionary: {image_name (without extension): device}
    foreground_device_map = {}
    with open(foreground_csv, mode='r', encoding='utf-8') as fg_file:
        reader = csv.DictReader(fg_file)
        for row in reader:
            image_name = os.path.splitext(row['Image'])[0]
            device = row['Device']
            foreground_device_map[image_name] = device

    # Initialize the counter
    device_counter = Counter()

    # Read training CSV and count device occurrences
    with open(training_csv, mode='r', encoding='utf-8') as train_file:
        reader = csv.DictReader(train_file)
        for row in reader:
            foreground_name = os.path.splitext(row['Foreground'])[0]
            device = foreground_device_map.get(foreground_name)
            if device:
                device_counter[device] += 1

    print(dict(device_counter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count device frequency on the generated training set.")
    parser.add_argument("--foreground_csv", help="Path to foreground csv file")
    parser.add_argument("--training_csv", help="Path to training csv file")
    args = parser.parse_args()

    count_device_frequencies(args.training_csv, args.foreground_csv)
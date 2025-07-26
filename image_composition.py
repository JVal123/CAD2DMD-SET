import os
import time
import cv2
import shutil
import random
import json
import helper_functions
import sys
import labeler
from PIL import Image
import multiprocessing
from multiprocessing import Process, Queue, Pool
import uuid
from filelock import FileLock
import argparse


# Add the parent directory to the path
libcom_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libcom/libcom'))
fopa_folder = os.path.join(libcom_folder, "fopa_heat_map")
naive_composite_folder = os.path.join(libcom_folder, "naive_composition")
color_transfer_folder = os.path.join(libcom_folder, "color_transfer")

sys.path.append(fopa_folder)
sys.path.append(naive_composite_folder)
sys.path.append(color_transfer_folder)

# Now you can import the module
from fopa_heat_map import FOPAHeatMapModel
from generate_composite_image import get_composite_image
from reinhard import color_transfer

# Global model variable
net = None


def init_worker(logical_gpu_id):
    """
    Initialize the FOPA model on the GPU specified by its logical index in CUDA_VISIBLE_DEVICES.
    """
    global net

    # Get list of physical GPU IDs from the CUDA_VISIBLE_DEVICES environment variable
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")

    if logical_gpu_id >= len(visible_devices):
        raise ValueError(f"GPU index {logical_gpu_id} exceeds available devices: {visible_devices}")

    # Set the CUDA_VISIBLE_DEVICES for this worker to the intended physical GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[logical_gpu_id].strip()

    # FOPAHeatMapModel(device=0) will now use the correct physical GPU
    net = FOPAHeatMapModel(device=0)


def get_next_unique_number(counter_file='dataset/unique_id.txt'):
    lock = FileLock(counter_file + '.lock')
    with lock:
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                number = int(f.read().strip())
        else:
            number = 1000  # Start from a reasonable baseline

        next_number = number + 1

        with open(counter_file, 'w') as f:
            f.write(str(next_number))

    return str(next_number)


def rescale_image_to_fixed_size(image_path, max_size=256):
    """
    Resizes an image so that its largest side equals `max_size`, preserving aspect ratio.

    Args:
        image_path (str): Path to the image.
        max_size (int): Target size for the largest dimension.

    Returns:
        resized_img (PIL.Image): Resized image.
        original_size (tuple): Original (width, height).
        scale_factors (tuple): (scale_x, scale_y) used for resizing.
    """
    img = Image.open(image_path)
    original_size = img.size  # (width, height)

    orig_w, orig_h = original_size
    if orig_w >= orig_h:
        scale = max_size / orig_w
    else:
        scale = max_size / orig_h

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized_img, original_size, (scale, scale)


def rescale_bbox_to_original(bbox, scale_factors):
    """
    Rescales a bounding box from resized image back to original image size using scale factors.
    Input bbox must be in [x, y, w, h] format.

    Args:
        bbox (list): [x, y, w, h] from resized image.
        scale_factors (tuple): (scale_x, scale_y)

    Returns:
        list: Rescaled [x, y, w, h]
    """
    scale_x, scale_y = scale_factors
    x, y, w, h = bbox

    x = int(x / scale_x)
    y = int(y / scale_y)
    w = int(w / scale_x)
    h = int(h / scale_y)

    return [x, y, w, h]


def save_temp_rescaled_image(rescaled_img, original_path):
    """
    Saves the rescaled image to a temporary file for FOPA inference.

    Args:
        rescaled_img (PIL.Image): Rescaled image object.
        original_path (str): Path of the original image, to derive temp path.

    Returns:
        str: Path to the temporary saved rescaled image.
    """

    temp_folder = os.path.join(os.path.dirname(original_path), 'resized_backgrounds')
    os.makedirs(temp_folder, exist_ok=True)
    image_name = os.path.basename(original_path)
    
    temp_path = os.path.join(temp_folder, image_name)
    rescaled_img.save(temp_path)
    return temp_path


def rename_pair_images_to_number(file_paths_dict, new_number=1674):
    """
    Renames three image files (from a dict) to the same filename (a number) with their original extensions,
    without overwriting existing files. Returns a similarly structured dict with rename info.

    Args:
        file_paths_dict (dict): Dict with keys 'background', 'foreground_mask', 'foreground'.
        new_number (int): The number to use for renaming.

    Returns:
        dict: Same keys as input, each mapping to the new file path.

    Raises:
        FileExistsError: If a destination file already exists.
        FileNotFoundError: If any of the input paths are invalid.
        ValueError: If required keys are missing.
    """
    required_keys = ('background', 'foreground_mask', 'foreground')
    if not all(k in file_paths_dict for k in required_keys):
        raise ValueError(f"Dictionary must contain keys: {required_keys}")

    renamed_paths = {}

    for key, path in file_paths_dict.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found for {key}: {path}")

        directory = os.path.dirname(path)
        ext = os.path.splitext(path)[1]
        new_filename = f"{new_number}{ext}"
        new_path = os.path.join(directory, new_filename)

        if os.path.exists(new_path):
            raise FileExistsError(f"Target file already exists: {new_path}")

        shutil.move(path, new_path)

        renamed_paths[key] = new_path

    return renamed_paths


def restore_pair_images_name(originals, renamed):
    """
    Renames image files back to their original names using two dictionaries:
    one with original paths, and one with renamed paths.

    Args:
        originals (dict): Original file paths with keys 'background', 'foreground_mask', 'foreground'.
        renamed (dict): Current renamed file paths with the same keys.

    Returns:
        dict: A dictionary with the same keys, each mapping to the restored full path.

    Raises:
        FileExistsError: If the original file path already exists.
        FileNotFoundError: If the renamed file does not exist.
        ValueError: If the keys don't match.
    """
    if originals.keys() != renamed.keys():
        raise ValueError("Original and renamed dictionaries must have the same keys.")

    restored_paths = {}

    for key in originals:
        original_path = originals[key]
        renamed_path = renamed[key]

        if not os.path.isfile(renamed_path):
            raise FileNotFoundError(f"Renamed file not found for {key}: {renamed_path}")

        if os.path.exists(original_path):
            raise FileExistsError(f"Cannot restore {key}, file already exists at original path: {original_path}")

        shutil.move(renamed_path, original_path)
        restored_paths[key] = original_path

    return restored_paths


def get_unique_image_triplet(background_folder, foreground_folder, mask_folder, used_combinations_path):

    if os.path.exists(used_combinations_path):
        with open(used_combinations_path, "r") as f:
            try:
                used_combinations = set(tuple(x) for x in json.load(f))
            except json.JSONDecodeError:
                used_combinations = set()
    else:
        used_combinations = set()

    valid_image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    backgrounds = [f for f in os.listdir(background_folder)
                if os.path.isfile(os.path.join(background_folder, f)) and os.path.splitext(f)[1].lower() in valid_image_exts]

    foregrounds = [f for f in os.listdir(foreground_folder)
                if os.path.isfile(os.path.join(foreground_folder, f)) and os.path.splitext(f)[1].lower() in valid_image_exts]

    max_combinations = len(backgrounds) * len(foregrounds)

    if len(used_combinations) >= max_combinations:
        raise Exception("All unique combinations have been used.")

    # Try to find a new, unused combination
    while True:
        bg = random.choice(backgrounds)
        fg = random.choice(foregrounds)
        combo_key = (bg, fg)

        if combo_key not in used_combinations:
            used_combinations.add(combo_key)
            break

    # Save updated combinations
    with open(used_combinations_path, "w") as f:
        json.dump(list(used_combinations), f)

    # Get mask (similar filename as foreground)
    name, _ = os.path.splitext(fg) #We extract the name of the foreground image
    mask = f"{name}_mask.png"

    return {
        'background': os.path.join(background_folder, bg),
        'foreground_mask': os.path.join(mask_folder, mask),
        'foreground': os.path.join(foreground_folder, fg)
    }


def get_pair_fopa(used_combinations_path):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    background_folder = os.path.join(data_dir, 'background')
    foreground_folder = os.path.join(data_dir, 'foreground')
    mask_folder = os.path.join(data_dir, 'foreground_mask')

    pair = get_unique_image_triplet(background_folder, foreground_folder, mask_folder, used_combinations_path)
   
    return pair


def get_custom_bbox(net, combinations_path, area_coverage, parent_folder):
    '''
    Uses the FOPA Model to extract possible location bounding boxes and then calculates their center. It then uses this coordinate,
    and the aspect ratio of the bounding box of the corresponding foreground object to get tailored bounding boxes, but still centered
    in a location given by FOPA. When no center is viable the bounding box center is chosen at random. 

    Returns the custom bbox in (x1, y1, x2, y2) format, ready to be used in the naive composition function.
    '''
    task_name = 'fopa_heat_map'

    # Collects a pairwise sample that has not been used before
    pair = get_pair_fopa(used_combinations_path=combinations_path)

    base_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', task_name)

    unique_id = get_next_unique_number(counter_file=os.path.join(parent_folder, "unique_id.txt"))
    result_dir = os.path.join(base_result_dir, unique_id)
    os.makedirs(os.path.join(result_dir, 'cache'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'heatmap'), exist_ok=True)

    unique_number = int(uuid.uuid4().int % 1e4)  # 4-digit unique number
    renamed_pair = rename_pair_images_to_number(file_paths_dict=pair, new_number=unique_number)

    # Rescale background image
    rescaled_bg_img, _, background_scale_factors = rescale_image_to_fixed_size(renamed_pair['background'])
    rescaled_bg_path = save_temp_rescaled_image(rescaled_bg_img, renamed_pair['background'])

    #fg_img, fg_mask, bg_img = renamed_pair['foreground'], renamed_pair['foreground_mask'], renamed_pair['background']
    fg_img, fg_mask = renamed_pair['foreground'], renamed_pair['foreground_mask']
    bboxes, heatmaps = net(fg_img, fg_mask, rescaled_bg_path, 
                cache_dir=os.path.join(result_dir, 'cache'), 
                heatmap_dir=os.path.join(result_dir, 'heatmap'))
 
    restore_pair_images_name(originals=pair, renamed=renamed_pair) #Rename filenames so that all combinations are possible
  
    # Clean up temporary rescaled background
    if os.path.exists(rescaled_bg_path):
        os.remove(rescaled_bg_path)

    _, _, w, h = helper_functions.get_bounding_box_xywh(pair['foreground_mask']) #Foreground object bounding box information
    aspect_ratio = w/h

    # Load background image and get its size
    bg_image = Image.open(pair['background'])
    bg_width, bg_height = bg_image.size
    bg_area = bg_width * bg_height
    min_bbox_area = area_coverage * bg_area  # 10% of the background area (minimum size for foreground object

    valid_bboxes = []

    for bbox in bboxes:
        rescaled_bbox = rescale_bbox_to_original(bbox, background_scale_factors)
        height = rescaled_bbox[3]
        width = rescaled_bbox[2]
        center_x = rescaled_bbox[0] + (width)/2
        center_y = rescaled_bbox[1] + (height)/2

        custom_width = aspect_ratio*height #Custom width, in order for new bounding box to preserve foreground object aspect ratio
        
        # Scaling
        area = custom_width * height

        if area < min_bbox_area:
            scale_factor = int((min_bbox_area / area) ** 0.5)  # Scale both width and height equally
            new_height = height * scale_factor
            new_width = custom_width * scale_factor
        else:
            new_height = height
            new_width = custom_width

        x1 = int(center_x - new_width/2)
        y1 = int(center_y - new_height/2)
        x2, y2 = int(x1 + new_width), int(y1 + new_height)
        custom_bbox = [x1, y1, x2, y2]

        # Validation
        if x1 >= 0 and y1 >= 0 and x2 <= bg_width and y2 <= bg_height:
            valid_bboxes.append(custom_bbox)
            print("✅ Custom bbox fits in background.")
            break

    if not valid_bboxes: # If no bbox is valid, because they are out of bounds
        return None
    else:
        pair['bbox'] = custom_bbox #You choose only one of the bboxes
        return pair


def get_random_bbox(combinations_path, area_coverage):
    '''
    Randomly generates a valid bounding box based on area coverage. The aspect ratio of the bounding box 
    corresponds to the one of the foreground object. 
    Returns the custom bbox in (x1, y1, x2, y2) format, ready to be used in the naive composition function.
    '''
    pair = get_pair_fopa(used_combinations_path=combinations_path)

    # Load images
    bg_image = Image.open(pair['background'])
    bg_width, bg_height = bg_image.size
    bg_area = bg_width * bg_height

    _, _, obj_w, obj_h = helper_functions.get_bounding_box_xywh(pair['foreground_mask'])
    aspect_ratio = obj_w / obj_h

    min_area = area_coverage * bg_area

    while True:
        height = random.randint(1, int(bg_height * 0.5)) # We limit the height to half the background's image height to avoid large, unrealistic foregroound objects
        width = int(height * aspect_ratio)

        # Ensure minimum area
        if width * height < min_area:
            scale = (min_area / (width * height)) ** 0.5
            width = int(width * scale)
            height = int(height * scale)

        x1 = random.randint(0, max(0, bg_width - width))
        y1 = random.randint(0, max(0, bg_height - height))
        x2 = x1 + width
        y2 = y1 + height

        if x2 <= bg_width and y2 <= bg_height:
            print("✅ Random bbox fits in background.")
            pair['bbox'] = [x1, y1, x2, y2]
            return pair


def naive_composition(pair):

    bg_img = pair['background']
    fg_img = pair['foreground']
    fg_mask = pair['foreground_mask']
    custom_bbox = pair['bbox']
    
    # Generate composite images by naive methods
    comp_img, comp_mask = get_composite_image(fg_img, fg_mask, bg_img, custom_bbox, 'none')

    return comp_img, comp_mask


def worker_task(args):
    """Worker function using globally initialized model."""
    worker_id, combinations_json, area_coverage, method, parent_folder= args
    global net
    try:
        pair = None
        while pair is None:
            if method == 'fopa':
                pair = get_custom_bbox(net, combinations_json, area_coverage, parent_folder)
            elif method == 'random':
                pair = get_random_bbox(combinations_json, area_coverage)
            else:
                raise ValueError(f"Unknown composition method: {method}")
            
        comp_img, comp_mask = naive_composition(pair)
        return {
            "comp_img": comp_img,
            "Foreground": os.path.basename(pair["foreground"]),
            "Foreground Mask": os.path.basename(pair["foreground_mask"]),
            "Background": os.path.basename(pair["background"]),
            "Bbox": pair["bbox"]
        }
    except Exception as e:
        print(f"[Worker {worker_id}] Failed: {e}")
        return None
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image composition pipeline")
    parser.add_argument('--method', type=str, default='fopa', choices=['fopa', 'random'],
                        help='Method to use for object placement: "fopa" or "random"')
    parser.add_argument('--result_dir', type=str, default='dataset/results/training_set',
                        help='Directory to save composite images and labels')
    parser.add_argument('--total_images', type=int, default=10,
                        help='Total number of images to generate')

    args = parser.parse_args()

    method = args.method
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    total_images = args.total_images

    multiprocessing.set_start_method('spawn')

    parent_folder = os.path.abspath(os.path.dirname(result_dir))
    combinations_json = os.path.join(parent_folder, 'used_combinations.json')
    foreground_labels = helper_functions.load_json(json_file='dataset/foreground/foreground_labels.json')
    training_labels_json = os.path.join(result_dir, 'training_labels.json')

    if os.path.exists(combinations_json):
        os.remove(combinations_json)
    if os.path.exists(training_labels_json):
        os.remove(training_labels_json)

    csv_file = os.path.join(result_dir, "training.csv")
    columns = ["Composite", "Foreground", "Foreground Mask", "Background", "Bbox"]
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Configuration
    workers_per_gpu = 2

    # Parse logical GPU IDs from CUDA_VISIBLE_DEVICES (e.g., "2,3")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip().isdigit()]
    num_visible_gpus = len(visible_gpu_ids)
    max_concurrent = num_visible_gpus * workers_per_gpu

    image_index = 1
    successful_count = 0

    # One pool per logical GPU index
    pools = []
    for logical_gpu_id in range(num_visible_gpus):
        pool = Pool(
            processes=workers_per_gpu,
            initializer=init_worker,
            initargs=(logical_gpu_id,)
        )
        pools.append(pool)

    try:
        while successful_count < total_images:
            remaining = total_images - successful_count
            batch_size = min(max_concurrent, remaining)
            batch_args = [(i, combinations_json, 0.10, method, parent_folder) for i in range(batch_size)]

            task_queue = []
            for i, args in enumerate(batch_args):
                pool = pools[i % num_visible_gpus]
                result = pool.apply_async(worker_task, (args,))
                task_queue.append((pool, result))

            for pool, result_obj in task_queue:
                result = result_obj.get()
                if result:
                    img_name = f"img{image_index}.png"
                    img_path = os.path.join(result_dir, img_name)
                    cv2.imwrite(img_path, result["comp_img"])

                    row = {
                        "Composite": img_name,
                        "Foreground": result["Foreground"],
                        "Foreground Mask": result["Foreground Mask"],
                        "Background": result["Background"],
                        "Bbox": result["Bbox"]
                    }
                    helper_functions.write_to_csv(filename=csv_file, data=row, column_titles=columns)

                    image_index += 1
                    successful_count += 1

                if successful_count >= total_images:
                    break

    finally:
        for pool in pools:
            pool.close()
            pool.join()

    labeler.generate_training_labels(
        training_csv=csv_file,
        foreground_labels_list=foreground_labels,
        training_labels_path=training_labels_json
    )     

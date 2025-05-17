#from common import *
#from libcom import FOPAHeatMapModel, get_composite_image, color_transfer
import os
import cv2
import shutil
import random
import json
import helper_functions
import sys
from PIL import Image

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
    # Load existing used combinations
    if os.path.exists(used_combinations_path):
        with open(used_combinations_path, "r") as f:
            used_combinations = set(tuple(x) for x in json.load(f))
    else:
        used_combinations = set()

    backgrounds = os.listdir(background_folder)
    foregrounds = os.listdir(foreground_folder)

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


def get_pair_fopa(used_combinations_filename):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    background_folder = os.path.join(data_dir, 'background')
    foreground_folder = os.path.join(data_dir, 'foreground')
    mask_folder = os.path.join(data_dir, 'foreground_mask')
    used_combinations_path = os.path.join(data_dir, used_combinations_filename)
    #samples  = []
    #for i in range(number):
    pair = get_unique_image_triplet(background_folder, foreground_folder, mask_folder, used_combinations_path)
    #samples.append(pair)

    #print(pair)
    return pair


def get_custom_bbox(combinations_json, area_coverage):
    '''
    Uses the FOPA Model to extract possible location bounding boxes and then calculates their center. It then uses this coordinate,
    and the aspect ratio of the bounding box of the corresponding foreground object to get tailored bounding boxes, but still centered
    in a location given by FOPA. When no center is viable the bounding box center is chosen at random. 

    Returns the custom bbox in (x1, y1, x2, y2) format, ready to be used in the naive composition function.
    '''
    task_name = 'fopa_heat_map'

    # Collects a pairwise sample that has not been used before
    pair = get_pair_fopa(used_combinations_filename=combinations_json)

    #print('Pairs List: ', pairs_list)
    #print('Pairs List[:1] ', pairs_list[:1])

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'grid'), exist_ok=True)
    
    net = FOPAHeatMapModel(device=0)
    #for pair in pairs_list:

    #print('Pair: ', pair)
    renamed_pair = rename_pair_images_to_number(file_paths_dict=pair)
    #print('Renamed Pair: ', renamed_pair)
    fg_img, fg_mask, bg_img = renamed_pair['foreground'], renamed_pair['foreground_mask'], renamed_pair['background']
    bboxes, heatmaps = net(fg_img, fg_mask, bg_img, 
                cache_dir=os.path.join(result_dir, 'cache'), 
                heatmap_dir=os.path.join(result_dir, 'heatmap'))
    #print('Bbox: ', bboxes)
    restore_pair_images_name(originals=pair, renamed=renamed_pair) #Rename filenames so that all combinations are possible
    #print('Restored Pair: ', restored_pair)

    _, _, w, h = helper_functions.get_bounding_box_xywh(pair['foreground_mask']) #Foreground object bounding box information
    aspect_ratio = w/h

    # Load background image and get its size
    bg_image = Image.open(pair['background'])
    bg_width, bg_height = bg_image.size
    bg_area = bg_width * bg_height
    min_bbox_area = area_coverage * bg_area  # 10% of the background area (minimum size for foreground object

    valid_bboxes = []

    #print('FOPA bboxes: ', bboxes)

    for bbox in bboxes:
        #print('FOPA Bbox: ', bbox)
        height = bbox[3]
        width = bbox[2]
        center_x = bbox[0] + (width)/2
        center_y = bbox[1] + (height)/2

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
        #print('Custom bbox: ', custom_bbox)

        # Validation
        if x1 >= 0 and y1 >= 0 and x2 <= bg_width and y2 <= bg_height:
            valid_bboxes.append(custom_bbox)
            print("✅ Custom bbox fits in background.")
            break
        else:
            print("❌ Custom bbox goes outside background bounds. Trying other FOPA center coordinates.")

    if not valid_bboxes: # If no bbox is valid, because they are out of bounds
        print('NO BOUNDING BOXES FIT, FETCHING ANOTHER PAIR...')
        #get_custom_bbox(combinations_json)
        return None
    else:
        pair['bbox'] = custom_bbox #You choose only one of the bboxes
        return pair

        

def naive_composition(pair):
    #task_name = 'naive_composition'

    #print('Pair: ', pair)

    bg_img = pair['background']
    fg_img = pair['foreground']
    fg_mask = pair['foreground_mask']
    custom_bbox = pair['bbox']

    #Change bbox format from [x, y, w, h] to [x1, y1, x2, y2]
    #x, y, w, h = fopa_bbox[0], fopa_bbox[1], fopa_bbox[2], fopa_bbox[3]
    #bbox = [x, y, x+w, y+h]

    #print('Custom bbox: ', custom_bbox)

    #result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', task_name)
    #if os.path.exists(result_dir):
    #    shutil.rmtree(result_dir)
    #os.makedirs(result_dir, exist_ok=True)
    
    
    #fg_mask = test_dir + 'foreground_mask/' + img_name.replace('.jpg', '.png')
    
    # generate composite images by naive methods
    comp_img, comp_mask = get_composite_image(fg_img, fg_mask, bg_img, custom_bbox, 'none')

    #comp_img_name = f'img{iteration}.png'
    #comp_mask_name = f'img{iteration}_mask.png'
    
    #comp_img_dir = os.path.join(result_dir, comp_img_name)
    #comp_mask_dir = os.path.join(result_dir, comp_mask_name)

    #Saves images
    #cv2.imwrite(comp_img_dir, comp_img)
    #cv2.imwrite(comp_mask_dir, comp_mask)

    return comp_img, comp_mask




if __name__ == '__main__':
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', 'training_set')
    comp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', 'naive_composition')
    #if os.path.exists(result_dir):
    #    shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    
    json_filename='used_combinations.json'


    render_number = 10
    pair = None

    for i in range(render_number):

        # -------------- Get Bounding Box ----------------------------------------------------

        while pair == None: #While there isn't an appropriate bounding box 
            pair = get_custom_bbox(combinations_json='used_combinations.json', area_coverage=0.10)

        bbox = pair['bbox'] # In (x1, y1, x2, y2) format
        # -------------- Naive Composition ----------------------------------------------------
        
        comp_img, comp_mask = naive_composition(pair)
        comp_img_name = f'img{i}.png'
        comp_mask_name = f'img{i}_mask.png'
        comp_img_dir = os.path.join(comp_dir, comp_img_name)
        comp_mask_dir = os.path.join(comp_dir, comp_mask_name)
        cv2.imwrite(comp_img_dir, comp_img)
        cv2.imwrite(comp_mask_dir, comp_mask)

        # -------------- Color Transfer ----------------------------------------------------

        transf_img = color_transfer(comp_img, comp_mask)
        pair = None

        transf_img_name = comp_img_name
        transf_img_dir = os.path.join(result_dir, transf_img_name)
        cv2.imwrite(transf_img_dir, transf_img)
        
        #comp_img_name = f'img{i}.png'
        #cv2.imwrite(os.path.join(result_dir, comp_img_name), comp_img)

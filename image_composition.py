#from common import *
from libcom import FOPAHeatMapModel, get_composite_image, color_transfer
import os
import cv2
import shutil
import random
import json
import helper_functions



'''def create_combinations(background_folder, foreground_folder, json_filename='composition_combinations.json'):
    # Create combinations .json file

    combinations = helper_functions.load_json(json_filename, initializer=[])

    backgrounds = os.listdir(background_folder)
    foregrounds = os.listdir(foreground_folder)

    max_combinations = len(backgrounds) * len(foregrounds)

    if len(combinations) >= max_combinations:
        print("All unique combinations have been used.")

    # Try to find all combination
    while (len(combinations) < max_combinations):
        bg = random.choice(backgrounds)
        fg = random.choice(foregrounds)
        combo_key = (bg, fg)

        if combo_key not in combinations:
            combinations.append(combo_key)

    # Save updated combinations
    with open(json_filename, "w") as f:
        json.dump(list(combinations), f)

    print('Created combinations json file...')'''


'''def return_combination(background_folder, foreground_folder, mask_folder, json_file, index):
    background, foreground = json_file[index]

    # Get mask (same filename as foreground)
    mask = f"{foreground}_mask.png"

    return {
        'background': os.path.join(background_folder, background),
        'foreground_mask': os.path.join(mask_folder, mask),
        'foreground': os.path.join(foreground_folder, foreground)
    }'''


'''def get_list_fopa(number, combinations_filename):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    background_folder = os.path.join(data_dir, 'background')
    foreground_folder = os.path.join(data_dir, 'foreground')
    mask_folder = os.path.join(data_dir, 'foreground_mask')

    combinations_path = os.path.join(data_dir, combinations_filename)
    if not os.path.exists(combinations_path): #Combination json has not yet been created
        create_combinations(background_folder, foreground_folder)

    combinations = helper_functions.load_json(combinations_path, initializer=[])
    samples  = []
    for i in range(number):
        pair = return_combination(background_folder, foreground_folder, mask_folder, json_file=combinations, index=i)
        samples.append(pair)
    return samples'''


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


def get_fopa_bbox(combinations_json):
    task_name = 'fopa_heat_map'

    # collect pairwise sample
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
    pair['bbox'] = bboxes
        

    #img_name  = os.path.basename(bg_img).replace('.png', '.jpg')
    #grid_img  = make_image_grid([bg_img, fg_img, heatmaps[0]])
    #res_path  = os.path.join(result_dir, 'grid', img_name)
    #cv2.imwrite(res_path, grid_img)
    #print('save result to ', res_path)

    #print('Pair with bbox: ', pair)
    return pair

        

def naive_composition(pair):
    task_name = 'naive_composition'

    print('Pair: ', pair)

    bg_img = pair['background']
    fg_img = pair['foreground']
    fg_mask = pair['foreground_mask']
    bbox = pair['bbox'][0]

    print('Bbox: ', bbox)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/results', task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    
    #fg_mask = test_dir + 'foreground_mask/' + img_name.replace('.jpg', '.png')
    
    # generate composite images by naive methods
    comp_img, comp_mask = get_composite_image(fg_img, fg_mask, bg_img, bbox, 'none')

    #Saves images
    cv2.imwrite(result_dir, comp_img)
    cv2.imwrite(result_dir, comp_mask)

    return comp_img, comp_mask


def naive_img_harmonization(comp_img, comp_mask):

    trans_img = color_transfer(comp_img, comp_mask)
    

    cv2.imwrite('../docs/_static/image/colortransfer_result1.jpg', grid_img)




if __name__ == '__main__':
    #background_folder = 'dataset/background'
    #foreground_folder = 'dataset/foreground'
    
    json_filename='used_combinations.json'

    #print(f'Begin extracting fopa bounding boxes...')

    for i in range(1):
        pair = get_fopa_bbox(combinations_json='used_combinations.json')
        comp_img, comp_mask = naive_composition(pair)

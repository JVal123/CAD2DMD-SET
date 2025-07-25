from displays import create_dictionary
from displays import create_display
import render_script
import post_processor
import bpy
import os
import copy
import argparse
import numpy as np
import helper_functions
import labeler



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates OCR Ready Dataset')
    parser.add_argument('-blender_path', type=str, help="Add the path to blender", dest="blender_path", default="/media/goncalo/3TBHDD/Joao/Thesis_Joao/blender-4.3.2-linux-x64/blender")
    parser.add_argument('-dataset_path', type=str, help='Add the path to the dataset folder.', dest="dataset_path", default='dataset')
    parser.add_argument('-foreground_path', type=str, help='Add the path to the foreground images.', dest="foreground_path", default='dataset/foreground')
    parser.add_argument('-background_path', type=str, help='Add the path to the background images.', dest="background_path", default='dataset/background')
    parser.add_argument('-mask_path', type=str, help='Add the path to the foreground mask images.', dest="mask_path", default='dataset/foreground_mask')
    parser.add_argument('-output_passes_path', type=str, help='Add the path to the output passes images.', dest="output_passes_path", default='dataset/output_passes')
    parser.add_argument('-displays_path', type=str, help='Add the path to the displays folder.', dest="displays_path", default='displays')
    parser.add_argument('-display_number', type=int, help='Define number of display images per device.', dest='display_number', default=10)
    parser.add_argument('-render_number', type=int, help='Define number of render images per device.', dest='render_number', default=10)
    parser.add_argument('-p', '-prob', type=float, help='Add the probability.', dest='prob', default=None)
    args = parser.parse_args()
    blender_path = args.blender_path
    dataset_path = args.dataset_path
    foreground_path = args.foreground_path
    background_path = args.background_path
    mask_path = args.mask_path
    output_passes_path = args.output_passes_path
    displays_path = args.displays_path
    display_number = args.display_number
    render_number = args.render_number
    prob = args.prob

    rng = np.random.default_rng(seed=42)

    # ---------------- Create Displays -------------------------------
 
    display_generator = create_display.DisplayGenerator(input_folder='images/real', dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    device_list = display_generator.get_child_folders()

    roi_filepath = os.path.abspath("displays/roi_mappings.json")
    roi_dictionary = helper_functions.load_json(roi_filepath)
    used_combinations_path = os.path.join(displays_path, 'images/generated/used_combinations.json')

    if os.path.exists(used_combinations_path):
        os.remove(used_combinations_path)

    for device_name in device_list:
        image_number = helper_functions.count_folder_images(f'displays/images/generated/{device_name}')
        max_image_number, is_actual_max = display_generator.maximum_display_images(device=device_name, roi_data=roi_dictionary, display_number=display_number)
        if image_number >= display_number or image_number >= max_image_number:
            print(f'The {device_name} display images are already generated. Advancing to next device...')
        else:
            print(f'Generating {device_name} display images:')

            generated = image_number
            while generated < max_image_number:
                success = display_generator.generate_random_unique_image(device=device_name, display_number=max_image_number, actual_max=is_actual_max, 
                                                                         roi_json_path=roi_filepath, used_path=used_combinations_path)
                
                if success:
                    generated += 1
                else:
                    break

    print('Display Generation Stage Complete! ✅')
    
    # ---------------- Renders -------------------------------------------

    render_generator = render_script.DataGenerator(models_folder='models', output_folder=foreground_path)

    # Json filepaths

    indices_filepath = os.path.abspath("models/face_indices.json")
    display_colors_filepath = os.path.abspath("display_colors.json")
    face_uv_rotation_filepath = os.path.abspath("uv_rotation.json")
    render_parameters = helper_functions.load_json("render_parameters.json")
    csv_file = os.path.join(foreground_path, "foreground.csv")
    columns = ["Image", "Device", "Mode", "Measurement", "Display Color", "Camera Distance", "Camera Shift X", "Camera Shift Y",
                "Camera Focal Length", "Object Rotation X", "Object Rotation Y", "Object Rotation Z", "Negative Case Rotation Z", "Light Color",
                "Light Energy", "Light Falloff", "Light Radius", "Light Distance X", "Light Distance Y", "Light Distance Z", 
                "Motion Blur Kernel Parameters [x, y, size]"]

    current_engine = bpy.context.scene.render.engine
    print("Current render engine:", current_engine)

    #Start image number based on current images in dataset folder

    dataset_count = helper_functions.count_folder_images(foreground_path) 

    if dataset_count > 0:
        counter = dataset_count + 1
    else:
        counter = 1



    for model in os.listdir(render_generator.models_folder):
        if model.endswith(".blend"):
            model_path = os.path.join(render_generator.models_folder, model)
            output_path = os.path.join(render_generator.output_folder, os.path.splitext(model)[0])
            model_name = os.path.splitext(model)[0]            

            face_index = render_generator.get_json_value(model_name, indices_filepath)
            face_uv_rotation = render_generator.get_json_value(model_name, face_uv_rotation_filepath)

            for i in range(counter, render_number + counter):

                # Open the .blend file
                bpy.ops.wm.open_mainfile(filepath=model_path)

                camera = bpy.data.objects.get("Camera")
                light = bpy.data.objects.get("Light")
                object = bpy.data.objects['3DModel']

                initial_rotation = copy.deepcopy(object.rotation_euler)
                
                render_generator.set_render_settings(render_parameters)

                full_object_visibility = False

                random_image, mode, measurement = helper_functions.get_random_image(f'displays/images/generated/{model_name}')
                display_color = render_generator.add_display_texture(object, face_index, face_uv_rotation, image_path=random_image, display_color_path=display_colors_filepath)

                while full_object_visibility == False:
                    camera_distance, shift_x, shift_y, focal_length= render_generator.translate_camera(camera, object, render_parameters)
                    x_rotation, y_rotation, z_rotation, neg_case_rotation = render_generator.rotate_object(object, initial_rotation, render_parameters)

                    full_object_visibility = render_generator.is_object_fully_visible(camera, object)

                color, energy, falloff, radius, x_distance, y_distance, z_distance = render_generator.translate_light(light, object, render_parameters)

                image_name = str(f'img{i}')

                render_generator.img_passes(image_name, output_dir=output_passes_path)

                counter = i + 1 

                
                row = {"Image": image_name, "Device": model_name, "Mode": mode, "Measurement": measurement, "Display Color": display_color, 
                       "Camera Distance": camera_distance, "Camera Shift X": shift_x, "Camera Shift Y": shift_y, "Camera Focal Length": focal_length, 
                       "Object Rotation X": x_rotation, "Object Rotation Y": y_rotation, "Object Rotation Z": z_rotation,  "Negative Case Rotation Z": neg_case_rotation,
                       "Light Color": color, "Light Energy": energy, "Light Falloff": falloff, "Light Radius": radius, "Light Distance X": x_distance,
                        "Light Distance Y": y_distance, "Light Distance Z": z_distance, "Motion Blur Kernel Parameters [x, y, size]": False}

                helper_functions.write_to_csv(csv_file, row, columns)

    # Blender cleanup
    render_generator.full_cleanup()
    
    helper_functions.rename_images_in_folder(foreground_path)
    helper_functions.rename_images_in_folder(output_passes_path)

    print("Dataset Generation Stage Complete! ✅")

    # ------------------ Label Generation -----------------------
                
    labeler.generate_labels(csv_file, mappings_json=roi_filepath, labels_json=f'{foreground_path}/foreground_labels.json', random_generator=rng)

    print("Label Generation Stage Complete! ✅")    

    # ------------------ Post Processing -------------------------

    # Validate the folder path
    if not os.path.exists(foreground_path):
        print(f"Error: The folder '{foreground_path}' does not exist.")

    if not os.path.isdir(foreground_path):
        print(f"Error: '{foreground_path}' is not a directory.")

    if prob == None:
        post_processor.add_motion_blur(foreground_path, csv_file)
    else:
        post_processor.add_motion_blur(foreground_path, csv_file, prob)
        
    # Generate binary masks
    
    post_processor.generate_masks_from_depth(rgb_folder=foreground_path, depth_folder=os.path.join(output_passes_path, 'depth'),
                                                         output_folder=mask_path)

    print("Post Processing Stage Complete! ✅")


    

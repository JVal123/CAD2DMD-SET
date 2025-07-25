from displays import create_dictionary
from displays import create_display
import render_script
import post_processor
import bpy
import os
import copy
import argparse
import csv
import pandas as pd
from typing import Dict, Any, Optional
import helper_functions



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Modifies Dataset Images of Choice')
    parser.add_argument('-blender_path', type=str, help='Add the path to blender.', dest="blender_path", default="/media/goncalo/3TBHDD/Joao/Thesis_Joao/blender-4.3.2-linux-x64/blender")
    parser.add_argument('-dataset_path', type=str, help='Add the path to the dataset folder.', dest="dataset_path", default='dataset')
    parser.add_argument('-foreground_path', type=str, help='Add the path to the foreground images.', dest="foreground_path", default='dataset/foreground')
    parser.add_argument('-background_path', type=str, help='Add the path to the background images.', dest="background_path", default='dataset/background')
    parser.add_argument('-mask_path', type=str, help='Add the path to the foreground mask images.', dest="mask_path", default='dataset/foreground_mask')
    parser.add_argument('-output_passes_path', type=str, help='Add the path to the output passes images.', dest="output_passes_path", default='dataset/output_passes')
    parser.add_argument('-csv_path', type=str, help='Add the csv filepath.', dest="csv_path", default='dataset.csv')
    parser.add_argument("-img_number", type=str, help='Select the image you want to modify.', dest="img_number", default='img1')
    args = parser.parse_args()
    blender_path=args.blender_path
    dataset_path = args.dataset_path
    foreground_path = args.foreground_path
    background_path = args.background_path
    mask_path = args.mask_path
    output_passes_path = args.output_passes_path
    csv_path = args.csv_path
    img_number = args.img_number


# ---------------- Get Csv Information ----------------------------------

    result = helper_functions.get_image_data(csv_path, img_number)
    
# ---------------- Change Display Information -------------------------------
 
    display_generator = create_display.DisplayGenerator(input_folder='images/real', dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    roi_filepath = os.path.abspath("displays/roi_mappings.json")
    roi_dictionary = helper_functions.load_json(roi_filepath)

   
    print(f"Adding new {result['Device']} display image...")
    img, mode, list_index = display_generator.remove_background(device=result['Device'], dictionary=roi_dictionary, csv_mode=result['Mode'])
    display_img_path, measurement = display_generator.change_measurement(img, device=result['Device'], mode=mode, dictionary=roi_dictionary, 
                                                            json_path=roi_filepath, index=list_index, csv_measurement=result['Measurement'])

    print('Done! ✅')
    
    # ---------------- Render -------------------------------------------

    render_generator = render_script.DataGenerator()

    # Json filepaths

    indices_filepath = os.path.abspath("models/face_indices.json")
    display_colors_filepath = os.path.abspath("display_colors.json")
    face_uv_rotation_filepath = os.path.abspath("uv_rotation.json")
    render_parameters = helper_functions.format_dict(result)

    csv_file = "dataset.csv"
    columns = ["Image", "Device", "Mode", "Measurement", "Display Color", "Camera Distance", "Camera Shift X", "Camera Shift Y",
                "Camera Focal Length", "Object Rotation X", "Object Rotation Y", "Object Rotation Z", "Negative Case Rotation Z", "Light Color",
                "Light Energy", "Light Falloff", "Light Radius", "Light Distance X", "Light Distance Y", "Light Distance Z", 
                "Motion Blur Kernel Parameters [x, y, size]"]

    current_engine = bpy.context.scene.render.engine
    print("Current render engine:", current_engine)


    for model in os.listdir(render_generator.models_folder):
        if model.endswith(".blend") and os.path.splitext(model)[0] == result['Device']:
            model_path = os.path.join(render_generator.models_folder, model)
            output_path = os.path.join(render_generator.output_folder, os.path.splitext(model)[0])
            model_name = os.path.splitext(model)[0]

            # Open the .blend file
            bpy.ops.wm.open_mainfile(filepath=model_path)

            camera = bpy.data.objects.get("Camera")
            light = bpy.data.objects.get("Light")
            object = bpy.data.objects['3DModel']

            render_generator.set_render_settings(render_parameters)

            initial_rotation = copy.deepcopy(object.rotation_euler)

            face_index = render_generator.get_json_value(model_name, indices_filepath)
            face_uv_rotation = render_generator.get_json_value(model_name, face_uv_rotation_filepath)

            # In this case, the used image is not random and corresponds to the previously generated display image
            display_color = render_generator.add_display_texture(object, face_index, face_uv_rotation, image_path=display_img_path, 
                                                 display_color_path=display_colors_filepath, display_color=tuple(render_parameters['display_color']))

            camera_distance, shift_x, shift_y, focal_length = render_generator.force_translate(camera, object, render_parameters)
            x_rotation, y_rotation, z_rotation, neg_case_rotation = render_generator.force_rotate(object, initial_rotation, render_parameters)
            color, energy, falloff, radius, x_distance, y_distance, z_distance = render_generator.force_light(light, object, render_parameters)

            image_name = img_number

            render_generator.img_passes(image_name)

            #Update the csv line in case any value wasn't possible and random ones had to be generated

            row = {"Image": image_name, "Device": model_name, "Mode": mode, "Measurement": measurement, "Display Color": display_color, 
                       "Camera Distance": camera_distance, "Camera Shift X": shift_x, "Camera Shift Y": shift_y, "Camera Focal Length": focal_length, 
                       "Object Rotation X": x_rotation, "Object Rotation Y": y_rotation, "Object Rotation Z": z_rotation,  "Negative Case Rotation Z": neg_case_rotation,
                       "Light Color": color, "Light Energy": energy, "Light Falloff": falloff, "Light Radius": radius, "Light Distance X": x_distance,
                        "Light Distance Y": y_distance, "Light Distance Z": z_distance, "Motion Blur Kernel Parameters [x, y, size]": False}

            helper_functions.write_to_csv(csv_file, row, columns)

    
    helper_functions.rename_single_image(folder_path='dataset', img_name=image_name)

    print("New image generated! ✅")

    # ------------------ Post Processing -------------------------

    # Validate the folder path
    if not os.path.exists(foreground_path):
        print(f"Error: The folder '{foreground_path}' does not exist.")

    if not os.path.isdir(foreground_path):
        print(f"Error: '{foreground_path}' is not a directory.")

    post_processor.force_motion_blur(foreground_path, dictionary=render_parameters, img_name=image_name)

    # Generate binary mask

    post_processor.force_masks_from_depth(rgb_image_path=f"{foreground_path}/{image_name}", depth_image_path=f"{output_passes_path}/depth/{image_name}",
                                          output_folder=mask_path)

    print("Post Processing Stage Complete! ✅")


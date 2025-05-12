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
    parser.add_argument('-dataset_path', type=str, help='Add the path to the dataset folder.', dest="dataset_path", default='dataset')
    parser.add_argument('-foreground_path', type=str, help='Add the path to the foreground images.', dest="foreground_path", default='dataset/foreground')
    parser.add_argument('-background_path', type=str, help='Add the path to the background images.', dest="background_path", default='dataset/background')
    parser.add_argument('-mask_path', type=str, help='Add the path to the foreground mask images.', dest="mask_path", default='dataset/foreground_mask')
    parser.add_argument('-output_passes_path', type=str, help='Add the path to the output passes images.', dest="output_passes_path", default='dataset/output_passes')
    parser.add_argument('-display_number', type=int, help='Define number of display images per device.', dest='display_number', default=10)
    parser.add_argument('-render_number', type=int, help='Define number of render images per device.', dest='render_number', default=10)
    parser.add_argument('-p', '-prob', type=float, help='Add the probability.', dest='prob', default=None)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    foreground_path = args.foreground_path
    background_path = args.background_path
    mask_path = args.mask_path
    output_passes_path = args.output_passes_path
    display_number = args.display_number
    render_number = args.render_number
    prob = args.prob

    rng = np.random.default_rng(seed=42)

    # ---------------- Create Dictionaries ---------------------------

    #dictionary = create_dictionary.DictionaryCreator(output_folder="dicts")
    #dictionary.create_dict(models_folder='models', device_name='metronome')

    #print('Dictionary Generation Stage Complete! ✅')

    # ---------------- Create Displays -------------------------------
 
    display_generator = create_display.DisplayGenerator(input_folder='images/real', dict_folder='dicts', font_folder='fonts', output_folder='images/generated')
    device_list = display_generator.get_child_folders()

    roi_filepath = os.path.abspath("displays/roi_mappings.json")
    roi_dictionary = helper_functions.load_json(roi_filepath)

    #Force display number to be much higher than render number to reduce chances of choosing the same measurement twice
    display_number = 5*render_number

    for device_name in device_list:
        image_number = helper_functions.count_folder_images(f'displays/images/generated/{device_name}')
        if image_number >= display_number:
            print(f'The {device_name} display images are already generated. Advancing to next device...')
        else:
            print(f'Generating {device_name} display images:')
            # Get the first image of the device and save the ROI coordinates  
            #img, thresh_value, img_height, img_width, mode = display_generator.remove_background(thresh_value, img_height, img_width, device=device_name, select_threshold=True)
            #x, y, width, height = display_generator.change_measurement(img, device=device_name, mode=mode, x=x, y=y, w=width, h=height, selectROI=True)

            for i in range(image_number, display_number):
                img, mode, list_index = display_generator.remove_background(device=device_name, dictionary=roi_dictionary)
                display_generator.change_measurement(img, device=device_name, mode=mode, dictionary=roi_dictionary, json_path=roi_filepath, index=list_index)

    print('Display Generation Stage Complete! ✅')
    
    # ---------------- Renders -------------------------------------------

    blender_path = "/media/goncalo/3TBHDD/Joao/Thesis_Joao/blender-4.3.2-linux-x64/blender" # Maybe add to the argument parser
    render_generator = render_script.DataGenerator(models_folder='models', output_folder=foreground_path)

    # Json filepaths

    indices_filepath = os.path.abspath("models/face_indices.json")
    display_colors_filepath = os.path.abspath("display_colors.json")
    face_uv_rotation_filepath = os.path.abspath("uv_rotation.json")
    #render_parameters_filepath = os.path.abspath("render_parameters.json") 
    render_parameters = helper_functions.load_json("render_parameters.json")
    csv_file = "dataset.csv"
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

            #print('Model path: ', model_path)

            # Open the .blend file
            bpy.ops.wm.open_mainfile(filepath=model_path)

            camera = bpy.data.objects.get("Camera")
            light = bpy.data.objects.get("Light")
            object = bpy.data.objects['3DModel']

            
            render_generator.set_render_settings(render_parameters)

            initial_rotation = copy.deepcopy(object.rotation_euler)

            for i in range(counter, render_number + counter):

                #display_visibility_ratio = 0
                full_object_visibility = False

                face_index = render_generator.get_json_value(model_name, indices_filepath)
                face_uv_rotation = render_generator.get_json_value(model_name, face_uv_rotation_filepath)
                #random_image = render_script.get_random_image(f'images/generated/{model_name}')
                random_image, mode, measurement = helper_functions.get_random_image(f'displays/images/generated/{model_name}')
                display_color = render_generator.add_display_texture(object, face_index, face_uv_rotation, image_path=random_image, display_color_path=display_colors_filepath)

                #while (display_visibility_ratio < 0.8 and full_object_visibility == False):
                while full_object_visibility == False:
                    camera_distance, shift_x, shift_y, focal_length= render_generator.translate_camera(camera, object, render_parameters)
                    x_rotation, y_rotation, z_rotation, neg_case_rotation = render_generator.rotate_object(object, initial_rotation, render_parameters)

                    #if neg_case_rotation != False: #Negative case scenarios don't matter for display visibility
                    #    break

                    #display_visibility_ratio = render_generator.get_display_visibility_ratio(camera, object, face_index)
                    full_object_visibility = render_generator.is_object_fully_visible(camera, object)

                    '''print(f"Object visibility ratio: {object_visibility_ratio*100:.2f}%")

                    if object_visibility_ratio >= 1:
                        print("Display is sufficiently visible!")
                    else:
                        print("Too much display is outside view. Re-randomizing.")'''

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
    
    helper_functions.rename_images_in_folder(foreground_path)
    helper_functions.rename_images_in_folder(output_passes_path)

    print("Dataset Generation Stage Complete! ✅")

    # ------------------ Label Generation -----------------------
                
    labeler.generate_labels(csv_file, mappings_json=roi_filepath, labels_json=f'{dataset_path}/labels.json', random_generator=rng)

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


    
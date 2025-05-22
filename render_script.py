import bpy
import bpy_extras
import os
import math
import mathutils
import random
import copy
import json
import shutil
import numpy as np
import cv2 as cv
import helper_functions
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint, box as shapely_box
from shapely.ops import unary_union



def get_random_image_from_subfolder(parent_folder):
    """
    Randomly selects a subfolder from the given parent folder,
    then randomly selects an image from that subfolder.
    
    Args:
        parent_folder (str): Path to the parent folder
        
    Returns:
        str: Path to the randomly selected image, or None if no image is found
    """
    # Check if the parent folder exists
    if not os.path.exists(parent_folder) or not os.path.isdir(parent_folder):
        print(f"Error: {parent_folder} does not exist or is not a directory")
        return None
    
    # List all subfolders in the parent folder
    subfolders = [f for f in os.listdir(parent_folder) 
                 if os.path.isdir(os.path.join(parent_folder, f))]
    
    # Check if there are any subfolders
    if not subfolders:
        print(f"Error: No subfolders found in {parent_folder}")
        return None
    
    # Randomly select a subfolder
    selected_subfolder = random.choice(subfolders)
    subfolder_path = os.path.join(parent_folder, selected_subfolder)
    
    print(f"Selected subfolder: {selected_subfolder}")
    
    # List all files in the selected subfolder
    files = os.listdir(subfolder_path)
    
    # Filter for image files (common image extensions)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]
    
    # Check if there are any image files
    if not image_files:
        print(f"Error: No image files found in {subfolder_path}")
        return None
    
    # Randomly select an image file
    selected_image = random.choice(image_files)
    image_path = os.path.join(subfolder_path, selected_image)
    
    print(f"Selected image: {selected_image}")
    
    return image_path


class DataGenerator:
    def __init__(self, models_folder, output_folder):
        # Absolute path of the script
        script_dir = os.path.dirname(os.path.realpath(__file__)) 

        # Join the script's directory with the relative paths.
        self.models_folder = os.path.join(script_dir, models_folder)
        self.output_folder = os.path.join(script_dir, output_folder)

        # Ensure that the output folder exists and eliminates existing one
        #if os.path.exists(self.output_folder):
        #    shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        
        
        #os.makedirs(self.output_folder, exist_ok=True)

    def set_render_settings(self, dictionary):
        """
        Sets the render settings
        """
        # Render settings
        bpy.context.scene.render.resolution_x = dictionary["settings"]["resolution"][0]  # Set resolution
        bpy.context.scene.render.resolution_y = dictionary["settings"]["resolution"][1]
        bpy.context.scene.render.image_settings.file_format = dictionary["settings"]["file_format"]  # Output format

    # ------ Blender Objects Functions --------------------

    def rotate_object(self, object, initial_rotation, dictionary):
        """
        Selects the object, applies its initial rotation and then rotates it randomly.
        The negative_case_prob variable defines how many negative case renders are purposefully created.
        """

        object.select_set(True) #Selects the object
        object.rotation_euler = initial_rotation #Reset the object's rotation

        negative_case_prob = dictionary["object"]["negative_case"]["prob"]
        #print(negative_case_prob)

        if random.random() < negative_case_prob:  
            # Apply a different rotation for 10% of cases
            #x_rotation = random.uniform(math.radians(-80), math.radians(80))
            #y_rotation = random.uniform(math.radians(-40), math.radians(40))
            z_rotation = random.uniform(math.radians(dictionary["object"]["negative_case"]["z_rotation"][0]), math.radians(dictionary["object"]["negative_case"]["z_rotation"][1]))

            #bpy.ops.transform.rotate(value=x_rotation, orient_axis='X', orient_type='GLOBAL')
            #bpy.ops.transform.rotate(value=y_rotation, orient_axis='Y', orient_type='GLOBAL')
            bpy.ops.transform.rotate(value=z_rotation, orient_axis='Z', orient_type='GLOBAL')

            print('NEGATIVE CASE CREATED')

            return 0, 0, 0, z_rotation

        else:
            rotation_axis = random.randint(0, 2) # Define the axis of rotation for the object

            match rotation_axis:
                case 0:
                    x_rotation = random.uniform(math.radians(dictionary["object"]["x_rotation"][0]), math.radians(dictionary["object"]["x_rotation"][1])) #Defines a random x-axis rotation for the object
                    bpy.ops.transform.rotate(value=x_rotation, orient_axis='X', orient_type='GLOBAL') #Rotates the object in the global x axis
                    return x_rotation, 0, 0, False
                case 1:
                    y_rotation = random.uniform(math.radians(dictionary["object"]["y_rotation"][0]), math.radians(dictionary["object"]["y_rotation"][1])) #Defines a random y-axis rotation for the object
                    bpy.ops.transform.rotate(value=y_rotation, orient_axis='Y', orient_type='GLOBAL') #Rotates the object in the global y axis
                    return 0, y_rotation, 0, False
                case 2:
                    z_rotation = random.uniform(math.radians(dictionary["object"]["z_rotation"][0]), math.radians(dictionary["object"]["z_rotation"][1])) #Defines a random z-axis rotation for the object
                    bpy.ops.transform.rotate(value=z_rotation, orient_axis='Z', orient_type='GLOBAL') #Rotates the object in the global z axis
                    return 0, 0, z_rotation, False

    def force_rotate(self, object, initial_rotation, dictionary):
        """
        Selects the object, applies its initial rotation and then rotates it as indicated by the dictionary. This function is to 
        be used when the dictionary corresponds to a row of the dataset csv file.
        """

        object.select_set(True) #Selects the object
        object.rotation_euler = initial_rotation #Reset the object's rotation

        if dictionary["object"]["negative_case"]["z_rotation"] == False:
            if dictionary["object"]["x_rotation"] != 0: #X rotation
                x_rotation = math.radians(dictionary["object"]["x_rotation"]) #Defines the x-axis rotation for the object
                bpy.ops.transform.rotate(value=x_rotation, orient_axis='X', orient_type='GLOBAL') #Rotates the object in the global x axis
                return x_rotation, 0, 0, False
            elif dictionary["object"]["y_rotation"] != 0: #Y rotation
                y_rotation = math.radians(dictionary["object"]["y_rotation"]) #Defines the y-axis rotation for the object
                bpy.ops.transform.rotate(value=y_rotation, orient_axis='Y', orient_type='GLOBAL') #Rotates the object in the global y axis
                return 0, y_rotation, 0, False
            else: #Z rotation
                z_rotation = math.radians(dictionary["object"]["z_rotation"]) #Defines the z-axis rotation for the object
                bpy.ops.transform.rotate(value=z_rotation, orient_axis='Z', orient_type='GLOBAL') #Rotates the object in the global z axis
                return 0, 0, z_rotation, False

        else:
            z_rotation = math.radians(dictionary["object"]["negative_case"]["z_rotation"])
            bpy.ops.transform.rotate(value=z_rotation, orient_axis='Z', orient_type='GLOBAL')

            print('NEGATIVE CASE CREATED')

            return 0, 0, 0, z_rotation

        

    def translate_camera(self, camera, object, dictionary):
        """
        Places the camera at a specific distance from the object, while focusing on it, applies random vertical and horizontal camera shifts and changes camera parameters.
        """
        # Ensure the camera type is set to Perspective
        if bpy.context.scene.camera.data.type != 'PERSP':
            bpy.context.scene.camera.data.type = 'PERSP'

        bpy.context.scene.unit_settings.system = 'METRIC'  #Ensure the blender units belong to the metric system (standard is in meters)
        bpy.context.scene.unit_settings.scale_length = 1.0  #Change the scale length to change from meters to another

        max_dimension = max(object.dimensions)
        #print('Object Dimensions: ', object.dimensions)
        #print('Max Dimension: ', max_dimension)
        distance = random.uniform(dictionary["camera"]["relative_distance"][0], dictionary["camera"]["relative_distance"][1])* max_dimension #Defines a random distance between the camera and object, based on its maximum dimension
        #shift_x = random.uniform(dictionary["camera"]["relative_shift_x"][0], dictionary["camera"]["relative_shift_x"][1]) * distance #Defines a random camera horizontal shift
        #shift_y = random.uniform(dictionary["camera"]["relative_shift_y"][0], dictionary["camera"]["relative_shift_y"][1]) * distance #Defines a random camera vertical shift
        camera_settings = bpy.types.Camera(camera.data)

        camera_settings.lens_unit = 'MILLIMETERS' #Set the focal length to millimeters
        camera_settings.lens = random.randint(dictionary["camera"]["focal_length"][0], dictionary["camera"]["focal_length"][1]) #Set a random focal length value 

        #camera_settings.shift_x = shift_x #Applies the camera horizontal shift
        #camera_settings.shift_y = shift_y #Applies the camera vertical shift


        camera.location =  mathutils.Vector((0.0, -1.0, 1.0))
        #print('Max dimension: ', max_dimension)
        #print('Camera location: ', camera.location)
        focus_direction = camera.location - object.location #Vector pointing from the object to the camera
        rot_quat = focus_direction.to_track_quat('Z', 'Y') #Aligns the Z axis of the camera to the focus direction (Y axis is as the secondary reference)

        camera.rotation_euler = rot_quat.to_euler() #Converts the camera's quaternion to euler representation
        camera.location = object.location + (rot_quat @ mathutils.Vector((0.0, 0.0, distance))) #Places the camera at the specified distance

        #print('Distance: ', distance)
        #rint('Camera shift-x: ', camera_settings.shift_x)
        #print('Camera shift-y: ', camera_settings.shift_y)

        #return distance, shift_x, shift_y, camera_settings.lens
        return distance, 0, 0, camera_settings.lens

    def force_translate(self, camera, object, dictionary):
        """
        Places the camera at a specific distance from the object, while focusing on it, applies vertical and horizontal camera shifts and changes camera parameters
        according to the dictionary. This function should onlu be used when the dictionary is a row from the dataset csv file.
        """
        # Ensure the camera type is set to Perspective
        if bpy.context.scene.camera.data.type != 'PERSP':
            bpy.context.scene.camera.data.type = 'PERSP'

        bpy.context.scene.unit_settings.system = 'METRIC'  #Ensure the blender units belong to the metric system (standard is in meters)
        bpy.context.scene.unit_settings.scale_length = 1.0  #Change the scale length to change from meters to another

        distance = dictionary["camera"]["distance"]
        #shift_x = random.uniform(dictionary["camera"]["relative_shift_x"][0], dictionary["camera"]["relative_shift_x"][1]) * distance #Defines a random camera horizontal shift
        #shift_y = random.uniform(dictionary["camera"]["relative_shift_y"][0], dictionary["camera"]["relative_shift_y"][1]) * distance #Defines a random camera vertical shift
        camera_settings = bpy.types.Camera(camera.data)

        camera_settings.lens_unit = 'MILLIMETERS' #Set the focal length to millimeters
        camera_settings.lens = dictionary["camera"]["focal_length"] #Set the focal length value 

        #camera_settings.shift_x = shift_x #Applies the camera horizontal shift
        #camera_settings.shift_y = shift_y #Applies the camera vertical shift


        camera.location =  mathutils.Vector((0.0, -1.0, 1.0))
        #print('Max dimension: ', max_dimension)
        #print('Camera location: ', camera.location)
        focus_direction = camera.location - object.location #Vector pointing from the object to the camera
        rot_quat = focus_direction.to_track_quat('Z', 'Y') #Aligns the Z axis of the camera to the focus direction (Y axis is as the secondary reference)

        camera.rotation_euler = rot_quat.to_euler() #Converts the camera's quaternion to euler representation
        camera.location = object.location + (rot_quat @ mathutils.Vector((0.0, 0.0, distance))) #Places the camera at the specified distance

        #print('Distance: ', distance)
        #rint('Camera shift-x: ', camera_settings.shift_x)
        #print('Camera shift-y: ', camera_settings.shift_y)

        return distance, 0, 0, camera_settings.lens



    def translate_light(self, light, object, dictionary):
        """
        Places the light at a specific distance from the object, while focusing on it and changes its parameters.
        """

        bpy.context.scene.unit_settings.system = 'METRIC'  #Ensure the blender units belong to the metric system (standard is in meters)
        bpy.context.scene.unit_settings.scale_length = 1.0  #Change the scale length to change from meters to another
        
        max_dimension = max(object.dimensions)

        #light_type = random.randint(0, 2) # Random light type is selected
        light_type = 0

        match light_type:
            case 0:
                light.data.type = 'POINT'
                color_vector = (1, random.uniform(dictionary["light"]["color"][0][0], dictionary["light"]["color"][0][1]), 
                                random.uniform(dictionary["light"]["color"][1][0], dictionary["light"]["color"][1][1])) #Defines the light color
                light.data.color = color_vector
                light.data.energy = random.uniform(dictionary["light"]["energy"][0], dictionary["light"]["energy"][1]) #Defines the light intensity
                light.data.use_soft_falloff = dictionary["light"]["falloff"]  #Apply falloff to avoid sharp edges when the light geometry intersects with other objects
                light.data.shadow_soft_size = random.uniform(dictionary["light"]["radius"][0], dictionary["light"]["radius"][1]) #Defines the light radius
                #print('Point Light Intensity: ', light.data.energy)
                #print('Point Light Radius: ', light.data.shadow_soft_size)
            case 1:
                light.data.type = 'SUN'
            case 2:
                light.data.type = 'SPOT'

        x_distance = random.uniform(dictionary["light"]["relative_x_distance"][0], dictionary["light"]["relative_x_distance"][1]) * max_dimension #Defines a random x-axis distance between the light and object
        y_distance = random.uniform(dictionary["light"]["relative_y_distance"][0], dictionary["light"]["relative_y_distance"][1]) * max_dimension #Defines a random y-axis distance between the light and object
        z_distance = random.uniform(dictionary["light"]["relative_z_distance"][0], dictionary["light"]["relative_z_distance"][1]) * max_dimension #Defines a random z-axis distance between the light and object

        light.location = object.location + mathutils.Vector((x_distance, y_distance, z_distance)) #Places the light at the specified distances
        #light.location = object.location + mathutils.Vector((0, -1, 1)) 
        #print('Distances: ', x_distance, y_distance, z_distance)

        return color_vector, light.data.energy, light.data.use_soft_falloff, light.data.shadow_soft_size, x_distance, y_distance, z_distance

    def force_light(self, light, object, dictionary):
        """
        Places the light at a specific distance from the object, while focusing on it and changes its parameters.
        """

        bpy.context.scene.unit_settings.system = 'METRIC'  #Ensure the blender units belong to the metric system (standard is in meters)
        bpy.context.scene.unit_settings.scale_length = 1.0  #Change the scale length to change from meters to another

        #light_type = random.randint(0, 2) # Random light type is selected
        light_type = 0

        match light_type:
            case 0:
                light.data.type = 'POINT'
                color_vector = (dictionary["light"]["color"][0], dictionary["light"]["color"][1], dictionary["light"]["color"][2]) #Defines the light color
                light.data.color = color_vector
                light.data.energy = dictionary["light"]["energy"] #Defines the light intensity
                light.data.use_soft_falloff = dictionary["light"]["falloff"]  #Apply falloff to avoid sharp edges when the light geometry intersects with other objects
                light.data.shadow_soft_size = dictionary["light"]["radius"] #Defines the light radius
            case 1:
                light.data.type = 'SUN'
            case 2:
                light.data.type = 'SPOT'

        x_distance = dictionary["light"]["x_distance"] #Defines the x-axis distance between the light and object
        y_distance = dictionary["light"]["y_distance"] #Defines the y-axis distance between the light and object
        z_distance = dictionary["light"]["z_distance"] #Defines the z-axis distance between the light and object

        light.location = object.location + mathutils.Vector((x_distance, y_distance, z_distance)) #Places the light at the specified distances
        #light.location = object.location + mathutils.Vector((0, -1, 1)) 
        #print('Distances: ', x_distance, y_distance, z_distance)

        return color_vector, light.data.energy, light.data.use_soft_falloff, light.data.shadow_soft_size, x_distance, y_distance, z_distance


        
    #------- Render Visibility Functions -----------------

    def get_display_visibility_ratio(self, camera, object, face_index):
        scene = bpy.context.scene
        mesh = object.data
        face = mesh.polygons[face_index]
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = object.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()

        coords_2d = []

        for vertex_idx in face.vertices:
            vertex = mesh_eval.vertices[vertex_idx]
            world_coord = object.matrix_world @ vertex.co

            co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_coord)

            # Keep points in NDC space, but still may be outside 0-1
            coords_2d.append((co_ndc.x, co_ndc.y))

        obj_eval.to_mesh_clear()

        if len(coords_2d) < 3:
            return 0.0  # Not enough points to form a polygon

        display_polygon = Polygon(coords_2d)

        # Define the camera view box [0,1] x [0,1]
        camera_view_box = shapely_box(0.0, 0.0, 1.0, 1.0)

        # Area before clipping
        original_area = display_polygon.area

        if not display_polygon.is_valid:
            # Reconstructs the geometry by buffering it outward by 0 units, in case the polygon is not valid
            #print('Polygon was not valid...')
            display_polygon = display_polygon.buffer(0) 

        if not camera_view_box.is_valid:
            # Reconstructs the geometry by buffering it outward by 0 units, in case the polygon is not valid
            #print('Polygon was not valid...')
            camera_view_box = camera_view_box.buffer(0)  

        # Clip display polygon with camera view
        visible_polygon = display_polygon.intersection(camera_view_box)

        visible_area = visible_polygon.area

        if original_area == 0:
            return 0.0

        visibility_ratio = visible_area / original_area
        return visibility_ratio


    def is_object_fully_visible(self, camera, obj):
        """
        Returns True if all vertices of the mesh object are within the camera view.
        """
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)

        mesh = obj_eval.to_mesh()
        
        if not mesh:
            return False

        for vert in mesh.vertices:
            world_coord = obj_eval.matrix_world @ vert.co
            co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_coord)

            # Check if vertex is outside camera view
            if not (0.0 <= co_ndc.x <= 1.0 and
                    0.0 <= co_ndc.y <= 1.0 and
                    co_ndc.z >= 0.0):  # in front of camera
                obj_eval.to_mesh_clear()
                return False

        obj_eval.to_mesh_clear()
        return True


    #------ UV Mappping Functions --------------------

    def get_json_value(self, object_name, json_filepath):
        # Load the JSON file
        if not os.path.exists(json_filepath):
            print(f"Error: JSON file not found at {json_filepath}")
            return None
        
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)

        value = json_data.get(object_name, None)  # Get the value of the corresponding object in the json file

        # Return both the UV coordinates and the face index
        return value

    def adjust_uv(self, object, face_index, uv_rotation, image):
        """Scales and rotates the UV map to match the image dimensions."""

        bpy.ops.object.mode_set(mode='OBJECT') # Change to Object Mode

        uv_layer = object.data.uv_layers.active.data
        face = object.data.polygons[face_index]

        # Get the UV coordinates of the face
        uv_coords = [uv_layer[loop_index].uv for loop_index in face.loop_indices]
        #print(uv_coords)

        # Calculate UV center
        uv_center = sum(uv_coords, mathutils.Vector((0, 0))) / len(uv_coords)
        #print(uv_center)

        # Offset to center the UV map on (0.5, 0.5)
        offset = mathutils.Vector((0.5, 0.5)) - uv_center
        uv_coords = [uv + offset for uv in uv_coords]
        #print('UV coords after offset: ', uv_coords)

        # Get image dimensions
        #image_width, image_height = image.size

        # Compute aspect ratios
        '''uv_aspect_ratio = abs((uv_coords[1].x - uv_coords[0].x)/ (uv_coords[2].y - uv_coords[1].y)) if uv_coords[1].x != uv_coords[0].x else None
        image_aspect_ratio = image_width / image_height
        print('UV aspect ratio: ', uv_aspect_ratio)
        print('Image aspect ratio: ', image_aspect_ratio)'''

        # Conditions to apply face uv map rotation 
        #if ((uv_aspect_ratio > 1 and image_aspect_ratio < 1) or (uv_aspect_ratio < 1 and image_aspect_ratio > 1)):
            #angle = math.radians(-90)
        angle = math.radians(uv_rotation)
        rot_matrix = mathutils.Vector((math.cos(angle), -math.sin(angle))), mathutils.Vector((math.sin(angle), math.cos(angle)))

        uv_coords = [mathutils.Vector((uv.x * rot_matrix[0].x + uv.y * rot_matrix[1].x,
                            uv.x * rot_matrix[0].y + uv.y * rot_matrix[1].y)) for uv in uv_coords]

        # Scale UV to fit image
        uv_min = mathutils.Vector((min(uv.x for uv in uv_coords), min(uv.y for uv in uv_coords)))
        uv_max = mathutils.Vector((max(uv.x for uv in uv_coords), max(uv.y for uv in uv_coords)))

        uv_size = uv_max - uv_min
        scale = mathutils.Vector((1.0 / uv_size.x, 1.0 / uv_size.y))
        
        uv_coords = [(uv - uv_min) * scale for uv in uv_coords]

        # Apply new UVs
        for loop_index, new_uv in zip(face.loop_indices, uv_coords):
            uv_layer[loop_index].uv = new_uv

    def add_display_texture(self, object, face_index, uv_rotation, image_path, display_color_path, display_color=False):
        # Ensure we have an active object and it's a mesh
        if object and object.type == 'MESH':
            # Switch to Edit Mode
            bpy.ops.object.mode_set(mode='EDIT')

            # Ensure we are in Face Select mode
            bpy.ops.mesh.select_mode(type="FACE")

            # Deselect all faces
            bpy.ops.mesh.select_all(action='DESELECT')

            bpy.ops.object.mode_set(mode='OBJECT') # Guarantee that the model is in object mode
            object.data.polygons[face_index].select = True # Select the face of the blender model

            # Create a new material
            mat = bpy.data.materials.get('Display')
            if not mat:
                mat = bpy.data.materials.new(name='Display')
                mat.use_nodes = True # Enable "Use Nodes" for the material

            # Assign the material to the object if not already assigned
            if mat.name not in [slot.name for slot in object.material_slots]:
                object.data.materials.append(mat)
                #print(f"Material added to object")
        else:
            print("Error: Selected object is not a mesh.")

        # Get the BSDF shader node
        bsdf = mat.node_tree.nodes["Principled BSDF"]

        # Create an Image Texture node
        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(image_path)

        # Create a ColorRamp node
        color_ramp = mat.node_tree.nodes.new('ShaderNodeValToRGB')

        # Create shader nodes
        mix_shader = mat.node_tree.nodes.new('ShaderNodeMixShader')  # Mix reflection and BSDF
        output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')  # Material output

        # Create a Glossy BSDF for reflections
        glossy = mat.node_tree.nodes.new('ShaderNodeBsdfGlossy')
        glossy.inputs['Roughness'].default_value = 0.1  # Adjust reflectivity

        # Create a Fresnel node to control reflections
        fresnel = mat.node_tree.nodes.new('ShaderNodeFresnel')
        fresnel.inputs['IOR'].default_value = 1.45  # Index of refraction for glass-like reflections

        # Adjust ColorRamp settings
        color_ramp.color_ramp.elements[0].position = 0.0  # Black (text)
        color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)  # Keep text black

        with open(display_color_path, 'r') as file:
            colors = json.load(file)
        # Combine all colors into a single list
        all_colors = list(colors["gray_colors"].values()) + list(colors["near_gray_colors"].values())

        color_ramp.color_ramp.elements[1].position = 1.0  # White (background)
        if type(display_color) == bool: #Means that we are not using the csv file for data generation
            display_color = tuple(random.choice(all_colors))  # Replace white with a random color from the json file

        # Only if the background of the image is white, is the display color applied 
        img = cv.imread(image_path)
        background_value, _ = helper_functions.determine_mask_colors(img)
        if background_value == 255: 
            color_ramp.color_ramp.elements[1].color = display_color
        #print('Display color: ', display_color)


        # Link nodes
        mat.node_tree.links.new(tex_image.outputs['Color'], color_ramp.inputs['Fac'])  # Image controls ColorRamp
        mat.node_tree.links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])  # Send result to BSDF

        # Link Fresnel to control reflection intensity
        mat.node_tree.links.new(fresnel.outputs['Fac'], mix_shader.inputs['Fac'])  # Fresnel as mix factor
        mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])  # BSDF to mix
        mat.node_tree.links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])  # Glossy reflection to mix

        # Final output
        mat.node_tree.links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

        # Assign the material to the selected face
        object.data.polygons[face_index].material_index = len(object.data.materials) - 1
        #print('Material assigned')

        # Ensure the object has a UV map
        if not object.data.uv_layers:
            print('The object does not have a UV map. Creating one instead...')
            object.data.uv_layers.new(name="UVMap")  # Create a UV map if none exists

        # Switch to Edit Mode for UV adjustments
        bpy.ops.object.mode_set(mode='EDIT')

        # Select the correct face for UV mapping
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        object.data.polygons[face_index].select = True
        bpy.ops.object.mode_set(mode='EDIT')

        # Reset the UVs so the texture covers the entire face
        bpy.ops.uv.unwrap(method='ANGLE_BASED')  # Use smart unwrapping
        bpy.ops.uv.select_all(action='SELECT')


        # Ensure the image matches the face’s UV map
        self.adjust_uv(object, face_index, uv_rotation, tex_image.image)

        # Ensure we are back in Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Update the scene
        bpy.context.view_layer.update()

        return display_color



    #------ Output Passes Functions --------------------

    def create_output_node(self, tree, render_layers, output_dir, render_name, node_output_name):
        '''
        Creates and links output nodes in blender
        '''
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = output_dir
        output_node.file_slots[0].path = render_name
        output_node.format.file_format = 'PNG'
        tree.links.new(render_layers.outputs[node_output_name], output_node.inputs[0])

    def img_passes(self, image_name, output_dir, extra_passes=False):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # Enable relevant render passes
        view_layer = bpy.context.scene.view_layers["ViewLayer"]
        view_layer.use_pass_combined = True  # Full Render

        if extra_passes:
            view_layer.use_pass_diffuse_color = True  # Albedo (Base Color)
            view_layer.use_pass_diffuse_direct = True  # Direct Diffuse Light (Shading approximation)
            view_layer.use_pass_normal = True  # Normal Map

        view_layer.use_pass_z = True  # Depth Pass 

        # Set output settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        # Enable Compositing Nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        tree.nodes.clear()

        # Create nodes
        render_layers = tree.nodes.new(type="CompositorNodeRLayers")

        # Create output nodes for different passes
        self.create_output_node(tree, render_layers, output_dir=self.output_folder, render_name=image_name, node_output_name="Image")
        
        if extra_passes:
            self.create_output_node(tree, render_layers, output_dir=os.path.join(output_dir, "albedo"), render_name=f"{image_name}_albedo", node_output_name="DiffCol")
            self.create_output_node(tree, render_layers, output_dir=os.path.join(output_dir, "shading"), render_name=f"{image_name}_shading", node_output_name="DiffDir")
            self.create_output_node(tree, render_layers, output_dir=os.path.join(output_dir, "normal"), render_name=f"{image_name}_normal", node_output_name="Normal")


        # ------ Normalizing Depth Pass ------
        depth_output_dir = os.path.join(output_dir, "depth")
        if not os.path.exists(depth_output_dir):
            os.makedirs(depth_output_dir)

        # Create a Map Range node to normalize depth
        map_range = tree.nodes.new(type="CompositorNodeMapRange")
        map_range.inputs[1].default_value = 0  # Min Depth (auto-detected)
        map_range.inputs[2].default_value = 10  # Max Depth (auto-detected, adjust if needed)
        map_range.inputs[3].default_value = 0  # Normalize to 0
        map_range.inputs[4].default_value = 1  # Normalize to 1
        map_range.use_clamp = True  # Clamp between 0 and 1

        # Create output node for normalized depth
        depth_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output_node.base_path = depth_output_dir
        depth_output_node.file_slots[0].path = f"{image_name}_depth"
        depth_output_node.format.file_format = 'PNG'

        # Link depth pass to Map Range, then to output
        tree.links.new(render_layers.outputs["Depth"], map_range.inputs[0])
        tree.links.new(map_range.outputs[0], depth_output_node.inputs[0])

        # Render & save outputs
        bpy.ops.render.render(write_still=False)

        print(f"Intrinsic images saved in: {output_dir}")



if __name__ == "__main__":
    blender_path = "/media/goncalo/3TBHDD/Joao/Thesis_Joao/blender-4.3.2-linux-x64/blender"
    generator = DataGenerator()
    render_number = 10

    current_engine = bpy.context.scene.render.engine
    print("Current render engine:", current_engine)

    # --- Individual Version -------------------------------------------------------------

    model_path = os.path.abspath("models/metronome.blend")
    output_path = generator.output_folder
    indices_filepath = os.path.abspath("models/face_indices.json")
    display_colors_filepath = os.path.abspath("display_colors.json")
    render_parameters = helper_functions.load_json("render_parameters.json")
    face_uv_rotation_filepath = os.path.abspath("uv_rotation.json")
    backgrounds_path = os.path.abspath("backgrounds/indoorCVPR_09")

    #print('Model path: ', model_path)

    # Open the .blend file
    bpy.ops.wm.open_mainfile(filepath=model_path)

    camera = bpy.data.objects.get("Camera")
    light = bpy.data.objects.get("Light")
    object = bpy.data.objects['3DModel']

    initial_rotation = copy.deepcopy(object.rotation_euler)

    face_index = generator.get_json_value('metronome', indices_filepath)
    face_uv_rotation = generator.get_json_value('metronome', face_uv_rotation_filepath)
    #random_image = render_script.get_random_image(f'images/generated/{model_name}')
    random_image, mode, measurement = helper_functions.get_random_image(f'displays/images/generated/metronome')
    generator.add_display_texture(object, face_index, face_uv_rotation, image_path=random_image, display_color_path=display_colors_filepath)

    rel_distance, shift_x, shift_y, focal_length= generator.translate_camera(camera, object, render_parameters)
    x_rotation, y_rotation, z_rotation, neg_case_rotation = generator.rotate_object(object, initial_rotation, render_parameters)
    color, energy, falloff, radius, x_distance, y_distance, z_distance = generator.translate_light(light, object, render_parameters)
    #render_generator.add_background(backgrounds_path)

    image_name = str(f'img')

    generator.img_decomposition(image_name, backgrounds_path)

    # --- Loop Version -----------------------------------------------------------

    # Loop through all .blend files
    '''for model in os.listdir(generator.models_folder):
        if model.endswith(".blend"):
            model_path = os.path.join(generator.models_folder, model)
            output_path = os.path.join(generator.output_folder, os.path.splitext(model)[0])

            #print('Model path: ', model_path)

            # Open the .blend file
            bpy.ops.wm.open_mainfile(filepath=model_path)

            camera = bpy.data.objects.get("Camera")
            light = bpy.data.objects.get("Light")
            object = bpy.data.objects['3DModel']

            # ----------- Render Process ---------------
            
            generator.set_render_settings()

            initial_rotation = copy.deepcopy(object.rotation_euler)

            for i in range(1, render_number + 1):
                generator.translate_camera(camera, object)
                generator.rotate_object(object, initial_rotation, negative_case_prob=0.05)
                #generator.render_depth(os.path.splitext(model)[0])
                generator.translate_light(light, object)

                # Set render output file path
                #bpy.context.scene.render.filepath = f"{output_path}_{i}.png"

                #bpy.ops.render.render(write_still=True)

                image_name = str(os.path.splitext(model)[0]) + f'_{i}'

                generator.img_decomposition(image_name)

    
    rename_images_in_folder('dataset')'''
            

    print("✅ Dataset Generation complete!")

import render_script
import bpy
import bpy_extras
import os
import copy
import argparse
import csv
import helper_functions
import ast
import cv2
import math
import numpy as np
from mathutils import Vector, geometry, Matrix
import json


def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        cleaned = s.replace("‘","'").replace("’","'").replace("“",'"').replace("”",'"')
        return ast.literal_eval(cleaned)
    

def find_roi_entry(roi_dict, device, mode):
    entries = roi_dict.get(device, [])
    for e in entries:
        if e.get("mode") == mode:
            return e
    # fallback: first entry with same mode prefix
    for e in entries:
        if str(e.get("mode","")).strip().lower() == str(mode).strip().lower():
            return e
    return None



def compute_homography(src_roi, dst_roi):
    """
    Compute the 3x3 homography H such that [x', y', 1]^T ~ H @ [x, y, 1]^T,
    mapping points in src_roi to the corresponding points in dst_roi.

    Parameters
    ----------
    src_roi : iterable of 4 (x, y)
        Source ROI points in order: TL, TR, BR, BL.
    dst_roi : iterable of 4 (x, y)
        Destination ROI points in the same order: TL, TR, BR, BL.

    Returns
    -------
    H : (3, 3) numpy.ndarray
        Homography matrix normalized so that H[2,2] == 1.

    Notes
    -----
    - Uses the DLT algorithm (SVD of the 8x9 design matrix built from 4 correspondences).
    - Requires at least 4 non-collinear correspondences; here we use exactly 4.
    """
    src = np.asarray(src_roi, dtype=float)
    dst = np.asarray(dst_roi, dtype=float)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src_roi and dst_roi must be arrays of shape (4, 2).")

    # Build the design matrix A (8 x 9) from point correspondences
    A = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([ 0,  0,  0, -x, -y, -1,  yp*x,  yp*y,  yp])
        A.append([ x,  y,  1,  0,  0,  0, -xp*x, -xp*y, -xp])
    A = np.asarray(A, dtype=float)

    # Solve Ah = 0 via SVD: h is the last column of V (or row of V^T) for smallest singular value
    _, _, VT = np.linalg.svd(A)
    h = VT[-1, :]  # (9,)
    H = h.reshape(3, 3)

    # Normalize so that bottom-right is 1 (avoid division by ~0)
    if np.isclose(H[2, 2], 0.0):
        # fallback: scale by Frobenius norm to avoid infs; caller can re-scale if desired
        H = H / (np.linalg.norm(H) + 1e-12)
    else:
        H = H / H[2, 2]

    return H


def apply_homography(H, points):
    """
    Apply homography H to a list of 2D points.

    Parameters
    ----------
    H : (3,3) array
        Homography matrix.
    points : iterable of (x, y)
        Points in source coordinates.

    Returns
    -------
    mapped_points : list of (x, y)
        Transformed points in destination coordinates.
    """
    print(points)
    pts = np.asarray(points, dtype=float)
    ones = np.ones((pts.shape[0], 1))
    print(pts)
    print(ones)
    homog_pts = np.hstack([pts, ones])       # shape (N,3)
    mapped = (H @ homog_pts.T).T             # shape (N,3)
    mapped /= mapped[:, [2]]                 # divide by w
    return mapped[:, :2].tolist()


def format_roi(roi):
    """
    Convert [x, y, w, h] (top-left pixel, width, height) to 4 corner pixels
    in clockwise order: TL -> TR -> BR -> BL.

    Notes
    -----
    - Assumes x,y are pixel indices (0-based). Width/height are in pixels.
    - Uses inclusive indexing for the far edge: (x+w-1, y+h-1).
      This pairs correctly with your mapper which samples pixel centers via +0.5.
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI must have positive width/height, got w={w}, h={h}")

    x0 = x
    y0 = y
    x1 = x + w - 1
    y1 = y + h - 1

    # Clockwise: top-left, top-right, bottom-right, bottom-left
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts Foreground ROIs.')
    parser.add_argument('--blender_path', type=str, help='Add the path to blender.', default="/media/goncalo/3TBHDD/Joao/Thesis_Joao/blender-4.3.2-linux-x64/blender")
    parser.add_argument('--foreground_dir', type=str, help='Add the path to the foreground folder.', default='dataset/foreground')
    parser.add_argument('--output_csv', type=str, help='Add the output csv filepath.', default='dataset/foreground/foreground_rois.csv')
    parser.add_argument('--face_index_json', type=str, help='Add the path to the face indices json file.', default='models/face_indices.json')
    parser.add_argument('--display_colors_json', type=str, help='Add the path to the display colors json file.', default='display_colors.json')
    parser.add_argument('--uv_rotation_json', type=str, help='Add the path to the uv rotation json file.', default='uv_rotation.json')
    parser.add_argument('--face_vertices_json', type=str, help='Add the path to the face vertices json file.', default='models/vertex_coords.json')
    parser.add_argument("--roi_json", type=str, default="displays/roi_mappings.json")
    args = parser.parse_args()

    render_generator = render_script.DataGenerator(models_folder="models", output_folder=args.foreground_dir)

    # Extract display corner vertices local coordinates
    if not os.path.exists(args.face_vertices_json):
        raise Exception("Please export local face vertices first.")


    # Json filepaths
    roi_dict = helper_functions.load_json(args.roi_json)
    face_idx_map = helper_functions.load_json(args.face_index_json)
    uvrot_map = helper_functions.load_json(args.uv_rotation_json)
    face_vertices = helper_functions.load_json(args.face_vertices_json)

    foreground_csv = os.path.join(args.foreground_dir, "foreground.csv")
    # Output CSV
    columns = ["Image", "Polygons", "Labels"]

    current_engine = bpy.context.scene.render.engine
    print("Current render engine:", current_engine)

    # Read foreground.csv
    with open(foreground_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["Image"]
            device = row["Device"]
            mode = row["Mode"]

            # measurements from csv (list)
            measurement_values = safe_literal_eval(row["Measurement"])
            if not isinstance(measurement_values, (list, tuple)):
                measurement_values = [measurement_values]

            # roi entry
            roi_entry = find_roi_entry(roi_dict, device, mode)
            if roi_entry is None:
                print(f"[WARN] No ROI entry for {device} / {mode}; skipping {image_name}")
                continue

            measurement_rois = roi_entry.get("rois", [])
            ocr_dict = roi_entry.get("ocr", {})
            ocr_rois = ocr_dict.get("rois", [])
            ocr_labels = ocr_dict.get("labels", [])
            all_rois = measurement_rois + ocr_rois

            display_img_name = roi_entry.get("image", None)
            img_path = os.path.join("displays/images/real", device, display_img_name)
            img = cv2.imread(img_path)
            img_h ,img_w = img.shape[:2]

    
            model_path = os.path.join(render_generator.models_folder, f"{device}.blend")

            # Open the .blend file
            bpy.ops.wm.open_mainfile(filepath=model_path)
            scene = bpy.context.scene
            render = scene.render
            camera = bpy.data.objects.get("Camera")
            object = bpy.data.objects['3DModel']
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = object.evaluated_get(depsgraph)

            row_info = helper_functions.get_image_data(foreground_csv, image_name)
            row_parameters =  helper_functions.format_dict(row_info)
            render_generator.set_render_settings(row_parameters)

            initial_rotation = copy.deepcopy(object.rotation_euler)

            face_index = render_generator.get_json_value(object_name=device, json_filepath=args.face_index_json)
            face_uv_rotation = render_generator.get_json_value(object_name=device, json_filepath=args.uv_rotation_json)

            camera_distance, shift_x, shift_y, focal_length = render_generator.force_translate(camera, object, row_parameters)
            x_rotation, y_rotation, z_rotation, neg_case_rotation = render_generator.force_rotate(object, initial_rotation, row_parameters)

            if neg_case_rotation:
                row = {"Image": image_name, "Polygons": [], "Labels": "."}
                helper_functions.write_to_csv(args.output_csv, row, columns)
                continue

            point_coords = []
            roi_coords = []
            roi_labels = []

            display_image_2d_points = [[0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]]
            print("2D image points: ", display_image_2d_points)

            # --------- Convert 3D local display points to 2D foreground image points -------------------

            for point_co in face_vertices[device]:
                
                #image_roi = format_roi(roi)
                
                point_world = obj_eval.matrix_world @ Vector(point_co)
    
                # Convert to normalized camera coordinates
                co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, point_world)

                # co_ndc is in [0, 1] range: (x, y, depth)
                # To get pixel coordinates:
                px = int(co_ndc.x * render.resolution_x)
                py = int((1 - co_ndc.y) * render.resolution_y)

                point_coords.append([px, py])

            foreground_homog_matrix = compute_homography(src_roi=display_image_2d_points, dst_roi=point_coords) # Calculate corresponding homography matrix

            # --------- Map 2D rois from display image to foreground image -------------------

            #point_coords = [] # Reset variable

            for image_roi in all_rois:
                formatted_roi = format_roi(image_roi)
                #for point_co in formatted_roi:
                    
                foreground_roi = apply_homography(foreground_homog_matrix, formatted_roi)

                #point_coords.append(foreground_point)
                
                roi_coords.append(foreground_roi)

            # ----------- Map 2D foreground image rois to composite image --------------------








            row = {"Image": image_name, "Polygons": roi_coords, "Labels": "."}

            helper_functions.write_to_csv(args.output_csv, row, columns)

            # -------- Visualize -------------------------------------------------

            '''# 1) Load your image
            #print(image_name)
            image = cv2.imread(f"dataset/foreground/{image_name}.png")
            if image is None:
                raise FileNotFoundError("Couldn't read image")
            H, W = image.shape[:2]


            #print(roi_coords)
            # 5) Draw points (circles) on the image
            for polygon in roi_coords:
                for (x, y) in polygon:
                    cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)  # red dot (BGR)

                # 6) (Optional) connect them with a line or polyline
                if len(polygon) > 1:
                    cv2.polylines(image, [np.array(roi_coords, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow("pts", image); cv2.waitKey(0); cv2.destroyAllWindows()'''

    

            




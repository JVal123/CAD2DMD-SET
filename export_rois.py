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

    pts = np.asarray(points, dtype=float)
    ones = np.ones((pts.shape[0], 1))
    homog_pts = np.hstack([pts, ones])       # shape (N,3)
    mapped = (H @ homog_pts.T).T             # shape (N,3)
    mapped /= mapped[:, [2]]                 # divide by w
    return mapped[:, :2].tolist()


def format_roi(roi, format="xywh"):
    """
    Convert an ROI to its 4 corner pixels in clockwise order:
    top-left -> top-right -> bottom-right -> bottom-left.

    Parameters
    ----------
    roi : sequence
        If format == "xywh": [x, y, w, h]
            - x, y are 0-based pixel indices (top-left).
            - w, h are in pixels; must be > 0.
            - Inclusive far edge is used: (x+w-1, y+h-1).

        If format == "tlbr":
            - Either [(x0, y0), (x1, y1)] or [x0, y0, x1, y1].
            - (x0, y0) is top-left, (x1, y1) is bottom-right.
            - Values are treated as inclusive pixel indices.
            - If the two points are not ordered, they will be normalized.

    format : {"xywh", "tlbr"}
        Input format key.

    Returns
    -------
    list[tuple[int, int]]
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    Raises
    ------
    ValueError
        On invalid shape, non-positive size (for xywh), or degenerate ROI.
        (A single pixel ROI is valid.)
    """
    if format not in {"xywh", "tlbr"}:
        raise ValueError(f"Unsupported format '{format}'. Use 'xywh' or 'tlbr'.")

    if format == "xywh":
        if len(roi) != 4:
            raise ValueError(f"'xywh' expects 4 values, got {len(roi)}")
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI must have positive width/height, got w={w}, h={h}")
        x0 = x
        y0 = y
        x1 = x + w - 1
        y1 = y + h - 1

    else:  # format == "tlbr"
        # Accept [(x0,y0),(x1,y1)] or [x0,y0,x1,y1]
        if len(roi) == 2 and hasattr(roi[0], "__iter__") and hasattr(roi[1], "__iter__"):
            (x0, y0), (x1, y1) = roi
        elif len(roi) == 4:
            x0, y0, x1, y1 = roi
        else:
            raise ValueError(
                "'tlbr' expects [(x0,y0),(x1,y1)] or [x0,y0,x1,y1]"
            )
        # Normalize in case points are not ordered
        x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
        y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)

        # Degenerate check: allow single-pixel (x0==x1 and y0==y1)
        # but disallow lines of zero height or width only if you want; here we allow them too.
        # No extra check needed unless you want to forbid zero-height/width lines.

    # Clockwise: top-left, top-right, bottom-right, bottom-left
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def get_object_bbox(scene, camera, obj_eval, clamp=True):
    """
    Returns the bounding box of an object in render pixel space
    as a list of four [x, y] lists: [TL, TR, BR, BL].
    
    Values are floats (sub-pixel precision).
    Returns None if object is not in front of camera or has no mesh.
    """

    r = scene.render
    W = r.resolution_x * r.resolution_percentage / 100.0
    H = r.resolution_y * r.resolution_percentage / 100.0

    me = obj_eval.to_mesh()
    if not me:
        return None

    try:
        xs, ys = [], []
        any_in_front = False

        mw = obj_eval.matrix_world
        for v in me.vertices:
            co_world = mw @ v.co
            co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, co_world)

            if co_ndc.z >= 0.0:
                any_in_front = True

            xs.append(co_ndc.x)
            ys.append(co_ndc.y)

        if not any_in_front:
            return None

        # Min/max in NDC
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        def ndc_to_px(x, y):
            px = x * W
            py = (1.0 - y) * H  # flip Y for image space
            return [px, py]

        # Corners: TL, TR, BR, BL
        TL = ndc_to_px(x_min, y_max)
        TR = ndc_to_px(x_max, y_max)
        BR = ndc_to_px(x_max, y_min)
        BL = ndc_to_px(x_min, y_min)

        if clamp:
            def clamp_pt(pt):
                return [max(0.0, min(W, pt[0])), max(0.0, min(H, pt[1]))]
            TL, TR, BR, BL = map(clamp_pt, [TL, TR, BR, BL])

        return [TL, TR, BR, BL]

    finally:
        obj_eval.to_mesh_clear()


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extracts Foreground ROIs.')
    parser.add_argument('--blender_path', type=str, help='Add the path to blender.', default="~/blender-4.3.2-linux-x64/blender")
    parser.add_argument('--foreground_dir', type=str, help='Add the path to the foreground image folder.', default='dataset/foreground')
    parser.add_argument('--composite_dir', type=str, help='Add the path to the composite image folder.', default='dataset/results/training_set')
    parser.add_argument('--face_vertices_json', type=str, help='Add the path to the face vertices json file.', default='models/vertex_coords.json')
    parser.add_argument("--roi_json", type=str, default="displays/roi_mappings.json")
    args = parser.parse_args()

    render_generator = render_script.DataGenerator(models_folder="models", output_folder=args.foreground_dir)

    # Extract display corner vertices local coordinates
    if not os.path.exists(args.face_vertices_json):
        raise Exception("Please export local face vertices first.")


    # Json filepaths
    roi_dict = helper_functions.load_json(args.roi_json)
    face_vertices = helper_functions.load_json(args.face_vertices_json)

    foreground_csv = os.path.join(args.foreground_dir, "foreground.csv")
    foreground_roi_csv = os.path.join(args.foreground_dir, "foreground_rois.csv")
    # Output CSV
    columns = ["Image", "Foreground Bbox", "Polygons", "Labels"]


    if not os.path.exists(foreground_roi_csv): # If the foreground file already exists, skip this step

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
                all_labels = measurement_values +  ocr_labels

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

                camera_distance, shift_x, shift_y, focal_length = render_generator.force_translate(camera, object, row_parameters)
                x_rotation, y_rotation, z_rotation, neg_case_rotation = render_generator.force_rotate(object, initial_rotation, row_parameters)

                foreground_bbox = get_object_bbox(scene, camera, obj_eval) #Bbox coordinates of the foreground object

                if neg_case_rotation:
                    row = {"Image": image_name, "Foreground Bbox": foreground_bbox, "Polygons": [], "Labels": ""}
                    helper_functions.write_to_csv(foreground_roi_csv, row, columns)
                    continue

                point_coords = []
                roi_coords = []
                roi_labels = []

                display_image_2d_points = [[0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]]

                # --------- Convert 3D local display points to 2D foreground image points -------------------

                for point_co in face_vertices[device]:
                    
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

                for image_roi in all_rois:
                    formatted_roi = format_roi(image_roi, format="xywh")
                    foreground_roi = apply_homography(foreground_homog_matrix, formatted_roi)
                    roi_coords.append(foreground_roi)

                # ------- Associate labels to obtained rois and save information to csv file --------------------------

                row = {"Image": image_name, "Foreground Bbox": foreground_bbox, "Polygons": roi_coords, "Labels": all_labels}

                helper_functions.write_to_csv(foreground_roi_csv, row, columns)

    
    
    # ----------- Map 2D foreground image rois to composite image --------------------

    composite_csv = os.path.join(args.composite_dir, "training.csv")
    composite_roi_csv = os.path.join(args.composite_dir, "composite_rois.csv")

    columns = ["Image", "Composite Bbox", "Polygons", "Labels"]

    # Build an index: Image -> row
    with open(foreground_roi_csv, "r", encoding="utf-8", newline="") as fgf:
        fg_reader = csv.DictReader(fgf)
        image_index = {row["Image"]: row for row in fg_reader}

    # Read foreground.csv
    with open(composite_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            comp_key = os.path.splitext(row["Composite"])[0]
            composite_bbox = ast.literal_eval(row["Bbox"]) # Convert csv bbox from string to list format
            composite_bbox = format_roi(composite_bbox, format="tlbr") # Convert bbox into appropriate 4 corner format

            # ----- Get the Foreground Bbox, Polygons and Labels from the foreground rois csv ----------------------

            fg_key = os.path.splitext(row["Foreground"])[0]
            match = image_index.get(fg_key)

            fg_bbox = ast.literal_eval(match.get("Foreground Bbox"))
            polygons = ast.literal_eval(match.get("Polygons")) if match else None
            labels = match.get("Labels") if match else None

            # ------ Calculate homography matrix and map rois -------------------------

            composite_homog_matrix = compute_homography(src_roi=fg_bbox, dst_roi=composite_bbox)

            comp_rois = []

            for fg_image_roi in polygons:
                comp_image_roi = apply_homography(composite_homog_matrix, fg_image_roi)
                comp_rois.append(comp_image_roi)

    
            row = {"Image": comp_key, "Composite Bbox": composite_bbox, "Polygons": comp_rois, "Labels": labels}

            helper_functions.write_to_csv(composite_roi_csv, row, columns)

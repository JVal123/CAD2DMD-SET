"""
Visualize foreground ROI polygons + labels on top of rendered images.

Notes:
- Assumes the CSV has columns: Image, Polygons, Labels
- Polygons is a JSON list of 4-point polygons in pixel coords:
    [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
- Labels is a JSON list (same length as Polygons)
- Colors cycle automatically.
"""

import os
import cv2
import json
import glob
import argparse
import numpy as np
import csv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/results/training_set/composite_rois.csv", help="Path to rois CSV")
    ap.add_argument("--images_root", default="dataset/results/training_set", help="Directory to search for images (recursive)")
    ap.add_argument("--out_dir", default="dataset/results/training_set/viz", help="Where to save visualized images")
    ap.add_argument("--show", action="store_true", help="Also open a preview window (Esc to close)")
    ap.add_argument("--line_thickness", type=int, default=1, help="Polygon outline thickness")
    ap.add_argument("--font_scale", type=float, default=0.4, help="Label font scale")
    ap.add_argument("--font", type=str, default="HERSHEY_SIMPLEX", help="OpenCV font (name)")
    return ap.parse_args()

def cv_font_by_name(name):
    # Map a few friendly names to cv2 fonts
    m = {
        "HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
        "HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
        "HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
        "HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
        "HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
        "HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        "HERSHEY_SIMPLEX_BOLD": cv2.FONT_HERSHEY_SIMPLEX,  # alias
    }
    return m.get(name, cv2.FONT_HERSHEY_SIMPLEX)

def find_image(images_root, image_name):
    # try direct join first
    direct = os.path.join(images_root, image_name)
    if os.path.exists(direct):
        return direct
    # else search recursively
    pattern = os.path.join(images_root, "**", image_name)
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None

def draw_label_with_bg(img, text, org, font, font_scale, txt_color, bg_color, thickness=1, pad=3):
    """
    Draw text with a filled rectangle background at org (top-left of text box).
    org: (x, y) in pixels
    """
    # text size
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    # background rect (slightly padded)
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + baseline + pad), bg_color, thickness=cv2.FILLED)
    # put text baseline at (x,y)
    cv2.putText(img, text, (x, y), font, font_scale, txt_color, thickness, lineType=cv2.LINE_AA)

def main():
    args = parse_args()
    font = cv_font_by_name(args.font)
    os.makedirs(args.out_dir, exist_ok=True)

    # simple color palette (BGR)
    palette = [
        (60,180,75), (230,25,75), (255,225,25),
        (0,130,200), (245,130,48), (145,30,180),
        (70,240,240), (240,50,230), (210,245,60),
        (250,190,190), (0,128,128), (230,190,255),
    ]

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            image_name = row["Image"].strip()
            try:
                polys = json.loads(row["Polygons"])
            except Exception as e:
                print(f"[WARN] {image_name}: couldn't parse Polygons JSON: {e}")
                continue
            try:
                labels = json.loads(row["Labels"].replace("'", '"'))
            except Exception:
                labels = []

            img_path = find_image(args.images_root, f"{image_name}.png")
            if not img_path:
                print(f"[WARN] {image_name}: image not found under {args.images_root}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] {image_name}: failed to load image.")
                continue

            H, W = img.shape[:2]
            thickness = max(1, int(round(args.line_thickness * (W + H) / 2000.0)))
            label_thickness = max(1, int(round(thickness)))
            font_scale = args.font_scale * (W + H) / 2000.0

            # Draw all polygons with labels
            for i, poly in enumerate(polys):
                color = palette[i % len(palette)]
                try:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                except Exception:
                    print(f"[WARN] {image_name}: polygon {i} has invalid shape; skipping.")
                    continue

                cv2.polylines(img, [pts], isClosed=True, color=color,
                              thickness=thickness, lineType=cv2.LINE_AA)

                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

                tl = tuple(pts[0,0].tolist())
                label_text = str(labels[i]) if i < len(labels) and labels[i] is not None else f"ROI {i}"
                #draw_label_with_bg(
                #    img, label_text, (tl[0], max(0, tl[1]-4)),
                #    font=font, font_scale=font_scale,
                #    txt_color=(255,255,255), bg_color=(0,0,0),
                #    thickness=label_thickness, pad=3
                #)

            out_path = os.path.join(args.out_dir,
                                    os.path.splitext(os.path.basename(image_name))[0] + "_viz.png")
            cv2.imwrite(out_path, img)
            print(f"[OK] Saved {out_path}")

            if args.show:
                cv2.imshow("ROI Viz", img)
                key = cv2.waitKey(0)
                if key == 27:  # Esc
                    cv2.destroyAllWindows()
                    return
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

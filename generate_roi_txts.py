"""
Read a CSV with columns: Image, Polygons, Labels
and write one .txt per image with each line:
x1,y1,x2,y2,...,xn,yn,<label>

- "Image" is used as the base filename (extension stripped if present).
- "Polygons" should be a Python/JSON-like list of polygons,
  each polygon a list of [x, y] points (floats or ints).
- "Labels" should be a list of strings, same order/length as Polygons.
  If a label is missing, "UNK" is used.
"""
import os
import ast
import argparse
import pandas as pd

def parse_points(obj):
    """Return list of (x, y) as integers, rounding if floats come in."""
    pts = []
    for pair in obj:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid point found: {pair}")
        x, y = pair
        # Round-to-nearest int
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        pts.append((xi, yi))
    return pts

def ensure_list(val):
    """Safely literal-eval strings like '[]' or '["a","b"]' to Python lists."""
    if pd.isna(val):
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
        except Exception:
            # Try to coerce CSV-style string lists into Python lists
            # e.g., "a,b,c" -> ["a","b","c"]
            out = [v.strip() for v in val.split(",") if v.strip() != ""]
        return out if isinstance(out, (list, tuple)) else [out]
    return [val]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", required=True, help="Directory to write .txt files")
    ap.add_argument("--image-col", default="Image", help="Column with composite image filename")
    ap.add_argument("--poly-col", default="Polygons", help="Column with polygons")
    ap.add_argument("--label-col", default="Labels", help="Column with labels")
    ap.add_argument("--missing-label", default="UNK", help="Fallback label when missing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    if args.image_col not in df.columns or args.poly_col not in df.columns:
        raise SystemExit(f"CSV must contain '{args.image_col}' and '{args.poly_col}' columns. Found: {list(df.columns)}")

    for idx, row in df.iterrows():
        image_name = str(row[args.image_col])
        base = os.path.splitext(os.path.basename(image_name))[0]

        polys_raw = ensure_list(row[args.poly_col])
        labels_raw = ensure_list(row[args.label_col]) if args.label_col in df.columns else []

        # Some CSVs store polygons as list-of-lists-of-points; ensure that.
        # If it's a single polygon dict/list, wrap it to a list.
        if len(polys_raw) > 0 and isinstance(polys_raw[0], str):
            # Many times polygons come in as strings like "[[x,y],[x,y],...]"
            # because of double-serialization.
            try:
                polys_raw = [ast.literal_eval(p) for p in polys_raw]
            except Exception:
                pass

        # If it's actually a single polygon (list of [x,y] points), wrap it
        if len(polys_raw) > 0 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polys_raw):
            polys_raw = [polys_raw]

        labels = [str(l) for l in labels_raw]
        out_lines = []

        for i, poly in enumerate(polys_raw):
            if not isinstance(poly, (list, tuple)) or len(poly) == 0:
                continue
            pts = parse_points(poly)
            flat = []
            for (x, y) in pts:
                flat.extend([x, y])
            label = labels[i] if i < len(labels) else args.missing_label
            # Compose "x1,y1,x2,y2,...,label"
            line = ",".join([str(v) for v in flat + [label]])
            out_lines.append(line)

        out_path = os.path.join(args.outdir, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))

        print(f"Wrote {out_path} with {len(out_lines)} polygons.")

if __name__ == "__main__":
    main()

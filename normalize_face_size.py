"""
Make all faces the same size, independent of hair.

Hair makes the cut-out silhouettes very different in size, so the size of
the visible (non-transparent) area says little about the size of the face.
Instead this script finds facial landmarks with MediaPipe Face Landmarker
(478 points) and measures the face with a landmark distance that hair does
not affect. MEASURE picks which one:
  "eye_chin" -- midpoint between the pupils down to the chin. Barely changes
                when the head is turned a little to the side. (default)
  "ipd"      -- distance between the pupils (inter-pupillary distance),
                common in face research, but shrinks when the head is turned
                to the side, so turned faces get scaled up too much.
  "combined" -- geometric mean of the two, balancing their errors.

Every image is then scaled so that this measure equals the
same target (by default the median over all images, so on average faces
change as little as possible), and shifted so that the midpoint between the
eyes lands on the same spot in every image. Transparency is kept.

Canvas (CANVAS_MODE):
  "fit"  -- one square canvas, shared by all images, just big enough that no
            face or hair is cut off after scaling (eyes centered
            horizontally). Recommended.
  "same" -- keep the input size (400x400); faces that were scaled up may get
            hair cut off at the edge (listed at the end of the run).

Outputs (OUTPUT_DIR):
  <name>.png                 the normalized faces
  face_size_report.csv       per image: measured size, scale factor, eyes
                             position, and whether anything got cut off
  face_size_check.png        contact sheet of all results with the
                             detected eyes and chin marked, for checking
  layout.json                canvas size and where the eyes, chin and face
                             center are on it (shared by all images), plus
                             the middle of the nose in each image
                             ("nose_px"). The experiment reads this to put
                             each face's nose at the screen center rather
                             than centering the whole image.

Run with venv311 (needs mediapipe and the model file in models/):
    venv311/bin/python normalize_face_size.py              # eye_chin
    venv311/bin/python normalize_face_size.py ipd          # or: combined
Each measure writes to its own folder (see OUTPUT_DIR).

To make another set (e.g. the practice faces) match an already-normalized
set exactly -- same face size, canvas, eye position and face center -- use
--match with that set's folder (its layout.json is used as the target):
    venv311/bin/python normalize_face_size.py combined \
        --input Stimuli/practice_faces_nobg \
        --output Stimuli/practice_faces_sizenorm_combined \
        --match Stimuli/experimental_faces_sizenorm_combined
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MEASURES = ("eye_chin", "ipd", "combined")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_parser = argparse.ArgumentParser(description="Normalize face size.")
_parser.add_argument("measure", nargs="?", default="eye_chin", choices=MEASURES)
_parser.add_argument("--input", help="folder of background-removed PNGs")
_parser.add_argument("--output", help="output folder")
_parser.add_argument("--match", help="normalized folder whose layout.json to match")
ARGS, _ = _parser.parse_known_args()
MEASURE = ARGS.measure

INPUT_DIR = ARGS.input or os.path.join(SCRIPT_DIR, "Stimuli", "experimental_faces_nobg")
OUTPUT_DIR = ARGS.output or os.path.join(
    SCRIPT_DIR, "Stimuli",
    "experimental_faces_sizenorm" + ("" if MEASURE == "eye_chin" else f"_{MEASURE}"))
MATCH_LAYOUT = None
if ARGS.match:
    with open(os.path.join(ARGS.match, "layout.json")) as _f:
        MATCH_LAYOUT = json.load(_f)
    if MATCH_LAYOUT.get("measure") != MEASURE or "target_px" not in MATCH_LAYOUT:
        sys.exit(f"{ARGS.match}/layout.json is for a different measure or "
                 "has no target_px; re-run that folder first.")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")

# Target value of the measure in pixels. None -> median over all images.
TARGET_PX = None

# Where the midpoint between the eyes goes, as a fraction of the canvas
# (x, y). None for y -> median eye height over all images (after scaling),
# so faces move as little as possible.
EYES_X_FRAC = 0.5
EYES_Y_FRAC = None

CANVAS_MODE = "fit"     # "fit" or "same" (see the top of the file)
CANVAS_MARGIN_PX = 4    # empty border kept around everything in "fit" mode

# Rotate each face so the eyes are level. Off by default: it changes the
# image more than scaling does.
LEVEL_EYES = False

# MediaPipe Face Mesh landmark indices
RIGHT_IRIS = 468   # center of the right iris (subject's right)
LEFT_IRIS = 473    # center of the left iris
CHIN = 152         # lowest point of the chin (menton)
NOSE_TOP = 168     # top of the nose bridge, between the eyes
NOSE_TIP = 1       # tip of the nose

IMAGE_EXTS = (".png",)


def detect_landmarks(landmarker, rgba):
    """Return (eye_mid, chin, right_eye, left_eye, nose) in pixels, or None.
    nose = middle of the nose, halfway between the bridge top and the tip."""
    # Detect on the face composited over white (the model expects RGB)
    rgb = Image.alpha_composite(
        Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    w, h = rgba.size

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    right_eye, left_eye, chin = pt(RIGHT_IRIS), pt(LEFT_IRIS), pt(CHIN)
    eye_mid = (right_eye + left_eye) / 2
    nose = (pt(NOSE_TOP) + pt(NOSE_TIP)) / 2
    return eye_mid, chin, right_eye, left_eye, nose


def transform(rgba, scale, angle_deg, eye_mid, target_eye):
    """Scale (and optionally rotate) around the eyes, then move the eyes to
    target_eye. Returns a new RGBA image of the same size."""
    a = np.radians(angle_deg)
    cos, sin = np.cos(a) * scale, np.sin(a) * scale
    # forward map: out = M @ (in - eye_mid) + target_eye ; PIL needs inverse
    m = np.array([[cos, -sin], [sin, cos]])
    inv = np.linalg.inv(m)
    off = eye_mid - inv @ target_eye
    coeffs = (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1])
    # Transform premultiplied colors so transparent edges don't turn dark
    arr = np.asarray(rgba).astype(np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    chans = [Image.fromarray(arr[..., c]).transform(
        rgba.size, Image.AFFINE, coeffs, resample=Image.BICUBIC) for c in range(4)]
    out = np.stack([np.asarray(c) for c in chans], axis=-1).clip(0, 1)
    alpha = out[..., 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[..., :3] = np.where(alpha > 0, out[..., :3] / alpha, 0)
    return Image.fromarray((out.clip(0, 1) * 255 + 0.5).astype(np.uint8))


def _pad(rgba, size):
    """Place rgba at the top-left of a transparent size x size canvas, so
    transform() can render onto a larger working canvas."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(rgba, (0, 0))
    return canvas


def touches_edge(rgba, thresh=10):
    """True if any visible pixel lies on the canvas border (= cut off)."""
    a = np.asarray(rgba)[..., 3]
    return bool((a[0] > thresh).any() or (a[-1] > thresh).any()
                or (a[:, 0] > thresh).any() or (a[:, -1] > thresh).any())


def contact_sheet(items, path, cols=10, thumb=160):
    """items: list of (name, rgba, eye_mid, chin) after normalization."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 14)), (128, 128, 128))
    draw = ImageDraw.Draw(sheet)
    for k, (name, rgba, eye_mid, chin) in enumerate(items):
        s = thumb / rgba.size[0]
        x0, y0 = (k % cols) * thumb, (k // cols) * (thumb + 14)
        tile = Image.alpha_composite(
            Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
        sheet.paste(tile.resize((thumb, thumb)).convert("RGB"), (x0, y0))
        ex, ey = x0 + eye_mid[0] * s, y0 + eye_mid[1] * s
        cx, cy = x0 + chin[0] * s, y0 + chin[1] * s
        draw.line([(x0, ey), (x0 + thumb, ey)], fill=(0, 160, 255))   # eyes line
        draw.line([(x0, cy), (x0 + thumb, cy)], fill=(255, 80, 0))    # chin line
        draw.text((x0 + 3, y0 + thumb), os.path.splitext(name)[0], fill=(255, 255, 255))
    sheet.save(path)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model file not found: {MODEL_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(IMAGE_EXTS) and not f.startswith("."))

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1)
    images, measures, failed = {}, {}, []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for name in names:
            rgba = Image.open(os.path.join(INPUT_DIR, name)).convert("RGBA")
            found = detect_landmarks(landmarker, rgba)
            if found is None:
                failed.append(name)
                continue
            images[name] = rgba
            measures[name] = found

    if failed:
        print("No face found (skipped): " + ", ".join(failed))
    if not measures:
        sys.exit("No faces detected.")

    eye_chin = {n: np.linalg.norm(m[1] - m[0]) for n, m in measures.items()}
    ipd = {n: np.linalg.norm(m[3] - m[2]) for n, m in measures.items()}
    size = {n: {"eye_chin": eye_chin[n], "ipd": ipd[n],
                "combined": np.sqrt(eye_chin[n] * ipd[n])}[MEASURE]
            for n in measures}
    target = TARGET_PX or float(np.median(list(size.values())))
    w, h = next(iter(images.values())).size
    if MATCH_LAYOUT:
        target = MATCH_LAYOUT["target_px"]
        eyes_y = 0.0  # replaced below by the matched layout
    elif EYES_Y_FRAC is None:
        # median eye height after scaling each face around its own eyes is
        # just the median of the original eye heights
        eyes_y = float(np.median([m[0][1] for m in measures.values()]))
    else:
        eyes_y = EYES_Y_FRAC * h
    target_eye = np.array([EYES_X_FRAC * w, eyes_y])

    params = {}
    for name in measures:
        eye_mid, chin, r_eye, l_eye, _ = measures[name]
        angle = 0.0
        if LEVEL_EYES:
            d = l_eye - r_eye          # subject's right -> left eye
            if d[0] < 0:
                d = -d
            angle = np.degrees(np.arctan2(d[1], d[0]))
        params[name] = (target / size[name], angle)

    if MATCH_LAYOUT:
        # Same canvas and eye position as the matched set; anything outside
        # that canvas is cut off (reported at the end).
        mw, mh = MATCH_LAYOUT["canvas_px"]
        target_eye = np.array(MATCH_LAYOUT["eyes_px"], dtype=float)
        outputs = {n: transform(_pad(images[n], max(mw, mh, w, h)), s, a,
                                measures[n][0], target_eye).crop((0, 0, mw, mh))
                   for n, (s, a) in params.items()}
    elif CANVAS_MODE == "fit":
        # Render everything on a big working canvas, find the area any image
        # uses, and crop all images to the same square around it.
        big = 3 * max(w, h)
        work_eye = np.array([big / 2, big / 2])
        rendered = {n: transform(_pad(images[n], big), s, a,
                                 measures[n][0], work_eye)
                    for n, (s, a) in params.items()}
        used = np.zeros((big, big), bool)
        for im in rendered.values():
            used |= np.asarray(im)[..., 3] > 10
        ys, xs = np.where(used)
        half_w = max(work_eye[0] - xs.min(), xs.max() + 1 - work_eye[0])
        side = int(np.ceil(max(2 * half_w, ys.max() + 1 - ys.min()))) + 2 * CANVAS_MARGIN_PX
        left = int(round(work_eye[0] - side / 2))
        top = int(round((ys.min() + ys.max() + 1) / 2 - side / 2))
        box = (left, top, left + side, top + side)
        outputs = {n: im.crop(box) for n, im in rendered.items()}
        target_eye = work_eye - np.array([left, top])
    else:
        outputs = {n: transform(images[n], s, a, measures[n][0], target_eye)
                   for n, (s, a) in params.items()}

    report, sheet_items, clipped = [], [], []
    nose_out = {}
    for name in names:
        if name not in measures:
            continue
        eye_mid, chin, r_eye, l_eye, nose = measures[name]
        scale, angle = params[name]
        out = outputs[name]
        out.save(os.path.join(OUTPUT_DIR, name))

        cut = touches_edge(out) and not touches_edge(images[name])
        if cut:
            clipped.append(name)
        new_chin = target_eye + (chin - eye_mid) * scale  # (ignores rotation)
        sheet_items.append((name, out, target_eye, new_chin))
        a = np.radians(angle)  # same forward map as transform()
        m = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        nose_out[name] = target_eye + m @ (nose - eye_mid)
        report.append({
            "file": name,
            "measure": MEASURE,
            "measure_px_before": round(size[name], 1),
            "measure_px_after": round(target, 1),
            "eye_chin_px_before": round(eye_chin[name], 1),
            "ipd_px_before": round(ipd[name], 1),
            "scale": round(scale, 4),
            "rotation_deg": round(angle, 2),
            "eyes_x_before": round(eye_mid[0], 1),
            "eyes_y_before": round(eye_mid[1], 1),
            "cut_off_at_edge": cut,
        })

    with open(os.path.join(OUTPUT_DIR, "face_size_report.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        wr.writeheader()
        wr.writerows(report)
    contact_sheet(sheet_items, os.path.join(OUTPUT_DIR, "face_size_check.png"))

    # Face center = halfway between the eyes and the (median) chin. Eyes are
    # at the same spot in every image, so one layout fits all of them.
    # A matched set reuses the reference face center, so faces from both sets
    # land on exactly the same spot.
    chin_med = np.median([c for _, _, _, c in sheet_items], axis=0)
    face_center = (target_eye + chin_med) / 2
    if MATCH_LAYOUT:
        face_center = np.array(MATCH_LAYOUT["face_center_px"], dtype=float)
    with open(os.path.join(OUTPUT_DIR, "layout.json"), "w") as f:
        json.dump({
            "measure": MEASURE,
            "target_px": round(float(target), 3),
            "canvas_px": list(out.size),
            "eyes_px": [round(float(v), 1) for v in target_eye],
            "chin_median_px": [round(float(v), 1) for v in chin_med],
            "face_center_px": [round(float(v), 1) for v in face_center],
            "matched_to": ARGS.match,
            "nose_px": {n: [round(float(v), 1) for v in p]
                        for n, p in nose_out.items()},
        }, f, indent=2)

    scales = np.array([r["scale"] for r in report])
    print(f"{len(report)} faces -> {OUTPUT_DIR} "
          f"(canvas {out.size[0]}x{out.size[1]})")
    print(f"{MEASURE} target: {target:.1f} px; eyes at "
          f"({target_eye[0]:.0f}, {target_eye[1]:.0f})")
    print(f"scale factors: min {scales.min():.3f}, median {np.median(scales):.3f}, "
          f"max {scales.max():.3f}")
    if clipped:
        print("Hair/face now cut off at the image edge: " + ", ".join(clipped))


if __name__ == "__main__":
    main()

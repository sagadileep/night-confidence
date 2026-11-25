"""
Night Confidence - prototype demo
Author: Saga Dileep Sai (M.Sc. Robotic Systems Engineering, RWTH Aachen)

What it does, frame by frame:
  1. LIFT     - gently brightens the real night image (CLAHE on the lightness channel),
                without inventing anything.
  2. DETECT   - a YOLOv8 model finds people, cyclists and animals (the hazards).
  3. TRACK    - keeps the same ID on each hazard so the frames don't flicker.
               A ground check drops 'people' whose feet are above road level
               (for example, faces on billboards).
  4. FRAME    - draws a calm amber frame around each hazard. Text labels are OFF
                by default (switch on with --labels), so the view stays clean.
  5. HUD      - a small "NIGHT CONFIDENCE" indicator, plus an amber edge glow when
                a hazard is large in view (close / getting closer).

Usage:
  python night_confidence_demo.py input.mp4 output.mp4
  python night_confidence_demo.py input.mp4 output.mp4 --side-by-side   (before | after)
  python night_confidence_demo.py input.mp4 output.mp4 --labels         (show class labels)
  python night_confidence_demo.py input.mp4 output.mp4 --model yolov8n.pt --imgsz 640   (faster, less accurate)

Limits (honest): this demo uses a normal camera video, not a thermal camera, and a
general-purpose model trained mostly on daytime images. Deer are not a separate class
in this model, so animals may be labelled as other animals. The fleet day layer and
the head-up display are not part of this demo.
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

AMBER = (61, 179, 242)      # BGR of #F2B33D
PAPER = (237, 242, 244)
SKY = (232, 184, 127)

# COCO classes treated as hazards (vulnerable road users and animals)
HAZARDS = {0: 'Person', 1: 'Cyclist', 3: 'Motorbike', 14: 'Bird', 15: 'Animal', 16: 'Animal',
           17: 'Animal', 18: 'Animal', 19: 'Animal', 20: 'Animal', 21: 'Animal', 22: 'Animal', 23: 'Animal'}


def lift(frame, strength=0.65):
    """Gently brighten the night: CLAHE on lightness, blended with the original."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    # soft gamma lift for the darkest areas
    lut = np.array([((i / 255.0) ** 0.8) * 255 for i in range(256)], dtype=np.uint8)
    l2 = cv2.LUT(l2, lut)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(out, strength, frame, 1 - strength, 0)


def rounded_rect(img, x0, y0, x1, y1, color, thick, r=10):
    r = int(min(r, (x1 - x0) / 3, (y1 - y0) / 3))
    for (cx, cy, a0) in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, thick, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thick, cv2.LINE_AA)


def draw_frame(img, box, label=None):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = max(6, int((x1 - x0) * 0.06))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    s = max(2, int(h / 360))
    glow = np.zeros_like(img)
    rounded_rect(glow, x0, y0, x1, y1, AMBER, s * 5)
    glow = cv2.GaussianBlur(glow, (0, 0), s * 3)
    cv2.add(img, (glow * 0.7).astype(np.uint8), dst=img)
    rounded_rect(img, x0, y0, x1, y1, AMBER, s)
    if label:
        fs = 0.5 * s
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        cv2.rectangle(img, (x0, y0 - th - 14), (x0 + tw + 16, y0 - 4), AMBER, -1)
        cv2.putText(img, label, (x0 + 8, y0 - 10), cv2.FONT_HERSHEY_DUPLEX, fs, (42, 26, 19), 1, cv2.LINE_AA)


def hud(img, n_hazards, alert):
    h, w = img.shape[:2]
    s = h / 1080
    x, y = int(40 * s), int(60 * s)
    cv2.circle(img, (x, y - int(9 * s)), int(9 * s), AMBER, -1, cv2.LINE_AA)
    cv2.putText(img, 'NIGHT CONFIDENCE', (x + int(24 * s), y), cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, AMBER, max(1, int(2 * s)), cv2.LINE_AA)
    txt = f'{n_hazards} hazard' + ('s' if n_hazards != 1 else '') + ' framed' if n_hazards else 'Road clear'
    cv2.putText(img, txt, (x + int(24 * s), y + int(40 * s)), cv2.FONT_HERSHEY_DUPLEX, 0.7 * s, PAPER, max(1, int(1.5 * s)), cv2.LINE_AA)
    if alert > 0:
        edge = np.zeros_like(img)
        t = int(60 * s)
        cv2.rectangle(edge, (0, 0), (w - 1, h - 1), AMBER, t)
        edge = cv2.GaussianBlur(edge, (0, 0), t * 0.6)
        cv2.add(img, (edge * 0.45 * alert).astype(np.uint8), dst=img)


def tag(img, text, color):
    h, w = img.shape[:2]
    s = h / 1080
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, 2)
    x = (w - tw) // 2
    y = h - int(50 * s)
    cv2.rectangle(img, (x - int(20 * s), y - th - int(16 * s)), (x + tw + int(20 * s), y + int(14 * s)), color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, (42, 26, 19), max(1, int(2 * s)), cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--ground', type=float, default=0.5,
                    help='ignore hazards whose bottom edge is above this fraction of the image height (e.g. people on billboards)')
    ap.add_argument('--labels', action='store_true')
    ap.add_argument('--side-by-side', action='store_true')
    ap.add_argument('--max-width', type=int, default=1280)
    import sys
    if len(sys.argv) == 1:
        # Started from IDLE (F5) without arguments: ask for the video instead.
        src = input('Path to your night video (e.g. C:\\Users\\Bunny\\Videos\\night.mp4): ').strip().strip('"')
        dst = src.rsplit('.', 1)[0] + '_night_confidence.mp4'
        sbs = input('Side-by-side before/after? (y/n): ').strip().lower().startswith('y')
        args = ap.parse_args([src, dst] + (['--side-by-side'] if sbs else []))
    else:
        args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.max_width / W0)
    W, H = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2
    out_w = W * 2 if args.side_by_side else W
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, H))
    alert = 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        lifted = lift(frame)
        res = model.track(lifted, conf=args.conf, imgsz=args.imgsz, persist=True, verbose=False, classes=list(HAZARDS))[0]
        view = lifted.copy()
        big = 0.0
        count = 0
        for b in res.boxes:
            cls = int(b.cls[0])
            box = b.xyxy[0].tolist()
            if box[3] < args.ground * H:   # ground check: real road users stand on the ground
                continue
            draw_frame(view, box, HAZARDS[cls] if args.labels else None)
            count += 1
            big = max(big, (box[3] - box[1]) / H)
        alert = 0.8 * alert + 0.2 * (1.0 if big > 0.35 else 0.0)
        hud(view, count, alert)
        if args.side_by_side:
            left = frame.copy()
            tag(left, 'WITHOUT', (200, 200, 200))
            tag(view, 'WITH NIGHT CONFIDENCE', AMBER)
            view = np.hstack([left, view])
        writer.write(view)
        n += 1
    writer.release()
    print(f'done: {n} frames -> {args.output}')


if __name__ == '__main__':
    main()

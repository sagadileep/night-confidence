# Night Confidence: prototype demo

**Giving drivers eyes at night.** A concept for safer, calmer night driving, with a working computer-vision prototype.

Author: Saga Dileep Sai, M.Sc. Robotic Systems Engineering, RWTH Aachen

## The idea

Night driving is stressful and dangerous. Many drivers strain their eyes, get blinded by glare, or avoid driving after dark altogether; older drivers most of all. Night Confidence combines sensors a modern car already has (thermal camera, radar, front cameras, head-up display) with AI computer vision and a fleet-refreshed daytime road layer. The driver sees the real night, gently lifted, with every hazard framed early in calm amber.

## Concept film

`concept_film.mp4` (2:12) tells the full story: three friends and the problems of night driving, the equipment, a before-and-after comparison, more ideas for the future, and this working prototype.

## What this prototype does

For each frame of a night driving video:

1. **Lift:** gently brightens the real image (CLAHE on the lightness channel). Nothing is invented.
2. **Detect:** a YOLOv8 model finds people, cyclists, motorbikes and animals.
3. **Track:** keeps a stable ID on each hazard, so frames don't flicker. A ground check ignores "people" whose feet are above road level, such as faces on billboards.
4. **Frame:** draws a calm amber frame around each hazard. Text labels are off by default and can be switched on.
5. **HUD:** a small "Night Confidence" indicator with a hazard count, plus an amber edge glow when a hazard is close.

## Run it

```bash
pip install -r requirements.txt
python night_confidence_demo.py my_night_drive.mp4 output.mp4 --side-by-side
```

Options:

- `--side-by-side`: shows "without" and "with" next to each other
- `--labels`: shows class labels (Person, Cyclist, Animal) on the frames
- `--conf 0.3`: detection confidence threshold
- `--max-width 1280`: resizes large videos for speed
- `--model yolov8n.pt --imgsz 640`: faster but misses more small pedestrians (default is `yolov8s.pt` at 960 px)

The model weights (`yolov8n.pt`) download automatically on the first run.

## Results on real night footage

`demo_night_city.mp4` shows a real night drive through a city, without (left) and with Night Confidence (right). Pedestrians on the sidewalks are framed in amber; the view is gently lifted.

Footage: "Taxis driving through the streets at night" from [Mixkit](https://mixkit.co), used under the Mixkit License.

`demo_rainy_crossing.mp4` is a stress test: a rainy night crossing in Times Square, filmed from street level. Rain, wet reflections, umbrellas and billboards full of faces. The prototype framed up to 13 pedestrians at once, ignored the faces on the billboards, and turned on the amber edge glow when people came close to the camera.

Footage: "Times Square during a rainy night" from [Mixkit](https://mixkit.co), used under the Mixkit License.

What I learned while testing:

- The smallest model (YOLOv8n at 640 px) missed several distant pedestrians. YOLOv8s at 960 px found most of them, at about 3× the compute.
- The detector framed people printed on a billboard. A simple ground check (feet must be below road level) removed these false alarms.

## Honest limits

- It uses a normal camera video, not a thermal camera, so very dark objects can still be missed.
- The model is a general-purpose one trained mostly on daytime images. Deer are not a separate class, so animals may get a general label.
- The fleet day layer, runway-style road lights and head-up display from the concept are not part of this prototype yet.

## Next steps

- Test on public thermal driving datasets
- Fine-tune the detector on night images, including deer and wild boar
- Measure how much earlier hazards are framed compared with headlights alone

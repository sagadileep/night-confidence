# Night Confidence: prototype demo

**Giving drivers eyes at night.** A concept for safer, calmer night driving, with a working computer-vision prototype.

Author: Saga Dileep Sai, M.Sc. Robotic Systems Engineering, RWTH Aachen

## The idea

Night driving is stressful and dangerous. Many drivers strain their eyes, get blinded by glare, or avoid driving after dark altogether; older drivers most of all. Night Confidence combines sensors a modern car already has (thermal camera, radar, front cameras, head-up display) with AI computer vision and a fleet-refreshed daytime road layer. The driver sees the real night, gently lifted, with every hazard framed early in calm amber.

## What this prototype does

For each frame of a night driving video:

1. **Lift:** gently brightens the real image (CLAHE on the lightness channel). Nothing is invented.
2. **Detect:** a YOLOv8 model finds people, cyclists, motorbikes and animals.
3. **Track:** keeps a stable ID on each hazard, so frames don't flicker.
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

The model weights (`yolov8n.pt`) download automatically on the first run.

## Honest limits

- It uses a normal camera video, not a thermal camera, so very dark objects can still be missed.
- The model is a general-purpose one trained mostly on daytime images. Deer are not a separate class, so animals may get a general label.
- The fleet day layer, runway-style road lights and head-up display from the concept are not part of this prototype yet.

## Next steps

- Test on public thermal driving datasets
- Fine-tune the detector on night images, including deer and wild boar
- Measure how much earlier hazards are framed compared with headlights alone

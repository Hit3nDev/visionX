# 👐 VisionX — gesture‑controlled camera effects

VisionX turns your two hands into a live "lens". Hold both hands up in front of a webcam: the four fingertip points form a window, and everything inside that window is re‑rendered with a real‑time computer‑vision effect. Raise different fingers on your right hand to switch effects — no keyboard, no mouse.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0097A7)

## How it works

1. **Hand tracking** — MediaPipe Hands detects up to two hands (21 landmarks each) at 0.7 confidence.
2. **Window from fingertips** — left index + thumb and right index + thumb become the four corners of a polygon mask.
3. **Gesture → mode** — finger‑up detection on the right hand picks the effect.
4. **Masked compositing** — the chosen effect is applied to the frame and pasted back only inside the polygon.

| Gesture (right hand) | Mode | Effect |
|---|---|---|
| ☝️ Index up | 1 | **Thermal vision** — histogram‑equalised greyscale + Inferno colour map |
| ✌️ Middle up | 2 | **Invisible face** — Haar‑cascade face detection, face replaced with the background |
| 🖖 Ring up | 3 | **TV scan** — scan‑line / CRT distortion |

## Project structure

```
main.py            webcam loop, mask building, compositing
hand_tracker.py    MediaPipe wrapper: landmarks, fingertips, gesture → mode
effects.py         thermal, invisible_face, tv_scan, energy overlay
test.py            quick check that OpenCV renders a window
```

## Run it

```bash
git clone https://github.com/Hit3nDev/visionX.git
cd visionX
pip install opencv-python mediapipe numpy
python main.py          # press Q to quit
```

Needs a webcam. Works on CPU in real time at 1280×720.

## Ideas next

- Smooth the polygon corners across frames to reduce jitter
- Add a fourth gesture for a "cartoon" edge‑preserving filter
- Record the output to video

---

Built by [Hiten Mandhyan](https://github.com/Hit3nDev)

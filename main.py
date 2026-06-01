import cv2
import numpy as np

from hand_tracker import HandTracker
from effects import *

tracker = HandTracker()

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

mode = 1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    frame = cv2.filter2D(
        frame,
        -1,
        kernel
    )

    hands = tracker.get_data(frame)

    if hands and len(hands) >= 2:

        left = hands[0]
        right = hands[1]

        mode_detected = tracker.detect_mode(
            right
        )

        if mode_detected != 0:
            mode = mode_detected

        lt = left["index"]
        lb = left["thumb"]

        rt = right["index"]
        rb = right["thumb"]

        width = int(
            np.linalg.norm(
                np.array(rt) -
                np.array(lt)
            )
        )

        height = max(
            int(width*0.6),
            200
        )

        if mode == 1:
            processed = thermal(frame)

        elif mode == 2:
            processed = invisible_face(frame)

        else:
            processed = tv_scan(frame)

        pts = np.array(
            [
                lt,
                rt,
                rb,
                lb
            ],
            dtype=np.int32
        )

        mask = np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [pts],
            255
        )

        frame[mask==255] = \
            processed[mask==255]

        overlay = frame.copy()

        cv2.polylines(
            overlay,
            [pts],
            True,
            (255,255,255),
            2
        )

        frame = cv2.addWeighted(
            overlay,
            0.85,
            frame,
            0.15,
            0
        )

        energy_height = 60

        x = min(
            lt[0],
            lb[0],
            rt[0],
            rb[0]
        )

        y = max(
            lt[1],
            rt[1]
        )

        x2 = max(
            lt[0],
            lb[0],
            rt[0],
            rb[0]
        )

        energy = blue_energy_layer(
            max(x2-x,1),
            energy_height
        )

        try:

            frame[
                y:y+energy_height,
                x:x2
            ] = energy

        except:
            pass

    cv2.imshow(
        "VisionX",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
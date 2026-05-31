import cv2
import numpy as np

from hand_tracker import HandTracker
from effects import *

tracker = HandTracker()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

mode = 1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    frame = cv2.filter2D(
        frame,
        -1,
        kernel
    )

    corners = tracker.get_screen_points(frame)

    if mode == 1:
        processed = thermal(frame)

    elif mode == 2:
        processed = tv_scan(frame)

    elif mode == 3:
        processed = invisibility(frame)

    else:
        processed = face_invisible(frame)

    if corners is not None:

        lt, rt, rb, lb = corners

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

        frame[mask == 255] = processed[mask == 255]

        cv2.polylines(
            frame,
            [pts],
            True,
            (255, 255, 255),
            3
        )

        for point in pts:

            cv2.circle(
                frame,
                tuple(point),
                8,
                (0, 255, 0),
                -1
            )

    mode_text = {
        1: "THERMAL",
        2: "TV SCAN",
        3: "INVISIBILITY",
        4: "FACE INVISIBLE"
    }

    cv2.putText(
        frame,
        mode_text[mode],
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "1 Thermal | 2 TV | 3 Invisible | 4 Face",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "VisionX",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        mode = 1

    elif key == ord('2'):
        mode = 2

    elif key == ord('3'):
        mode = 3

    elif key == ord('4'):
        mode = 4

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

background = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def thermal(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    thermal_img = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_INFERNO
    )

    return thermal_img


def tv_scan(frame):

    img = frame.copy()

    h, w, _ = img.shape

    for y in range(0, h, 4):
        img[y:y+1, :] //= 2

    b, g, r = cv2.split(img)

    r = np.roll(r, 3, axis=1)
    b = np.roll(b, -3, axis=1)

    img = cv2.merge([b, g, r])

    noise = np.random.randint(
        0,
        25,
        img.shape,
        dtype=np.uint8
    )

    img = cv2.add(img, noise)

    return img


def invisibility(frame):

    global background

    if background is None:
        background = frame.copy()

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    result = frame.copy()

    result[mask > 0] = background[mask > 0]

    return result


def face_invisible(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    result = frame.copy()

    for (x, y, w, h) in faces:

        face = result[
            y:y+h,
            x:x+w
        ]

        blurred = cv2.GaussianBlur(
            face,
            (99, 99),
            30
        )

        result[
            y:y+h,
            x:x+w
        ] = blurred

    return result
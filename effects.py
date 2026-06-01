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
        (5,5),
        0
    )

    return cv2.applyColorMap(
        gray,
        cv2.COLORMAP_INFERNO
    )


def tv_scan(frame):

    img = frame.copy()

    h,w,_ = img.shape

    for y in range(0,h,4):
        img[y:y+1]//=2

    b,g,r = cv2.split(img)

    r = np.roll(r,4,axis=1)
    b = np.roll(b,-4,axis=1)

    img = cv2.merge([b,g,r])

    noise = np.random.randint(
        0,
        30,
        img.shape,
        dtype=np.uint8
    )

    img = cv2.add(img,noise)

    return img


def invisible_face(frame):

    global background

    if background is None:
        background = frame.copy()

    result = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.2,
        5
    )

    for (x,y,w,h) in faces:

        if background.shape == frame.shape:

            face_bg = background[
                y:y+h,
                x:x+w
            ]

            blur = cv2.GaussianBlur(
                face_bg,
                (21,21),
                0
            )

            result[
                y:y+h,
                x:x+w
            ] = blur

    return result


def blue_energy_layer(width,height):

    img = np.zeros(
        (height,width,3),
        dtype=np.uint8
    )

    for y in range(height):

        value = int(
            255 *
            (1 - y/height)
        )

        img[y,:,0] = value

    return img
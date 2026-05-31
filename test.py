import cv2
import numpy as np

img = np.zeros((500,500,3), dtype=np.uint8)

cv2.putText(
    img,
    "OpenCV Working",
    (50,250),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255,255,255),
    2
)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
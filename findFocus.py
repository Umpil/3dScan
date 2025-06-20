import cv2
import time
import numpy as np
from Constants import *

camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
camera.set(cv2.CAP_PROP_FOCUS, 0)
camera.read()
time.sleep(1)
for i, focus in enumerate([100, 200, 300, 400]):
    ret, frame = camera.read()
    show_frame = cv2.resize(frame, (1024, 720))
    cv2.imshow(f"{focus - 100}", show_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    camera.set(cv2.CAP_PROP_FOCUS, focus)
    camera.read()
    time.sleep(1)

camera.release()
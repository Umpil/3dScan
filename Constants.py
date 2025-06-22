import os
import cv2
import numpy as np
PROJECTOR_ID = 1
PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
PROJECTOR_COUNT_PHOTO = 20
CIRCLE_GRID_SIZE = (11, 4)
CAMERA_ID = 1
CAMERA_WIDTH = 4656
CAMERA_HEIGHT = 3496
SPECIAL_VIDEO_WRITER = False
VIDEO_WRITER = "MJPG"
PATH_SAVE_CALIBRE_PROJECTOR = "DataSet/ProjCalib"
PATH_SAVE_ENCODED = "DataSet/Encoded"
PATH_SAVE_DECODED = "DataSet/Decoded"
PATH_SAVE_LITED = "DataSet/Lited"
PATH_CAMERA_CALIBRE = "DataSet/CamCalib"
PATH_PARAMETERS = "DataSet/Parameters"
PATH_END = "DataSet/End"
for path_name in [PATH_CAMERA_CALIBRE, PATH_SAVE_CALIBRE_PROJECTOR, PATH_SAVE_DECODED, PATH_SAVE_ENCODED, PATH_SAVE_LITED, PATH_PARAMETERS, PATH_END]:
    os.makedirs(path_name, exist_ok=True)

colored_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

# colored_image[:CAMERA_HEIGHT//2, :CAMERA_WIDTH//2, :] = [255, 0, 0]

# colored_image[:CAMERA_HEIGHT//2, CAMERA_WIDTH//2:, :] = [0, 255, 0]

# colored_image[CAMERA_HEIGHT//2:, CAMERA_WIDTH//2:, :] = [0, 0, 255]

colored_image[:, :, :] = [0, 0, 0]
colored_image[800:3450, 400:2850, :] = [0, 255, 0]

cv2.imwrite("colored.png", colored_image)
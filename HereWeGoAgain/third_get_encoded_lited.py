import numpy as np
import cv2
import time
import glob
from screeninfo import get_monitors
from constants import *
from utils import get_gray_code_images, get_Lited, get_Ld_Lg, gray_to_decimal, adaptive_illumination_correction, contrast
make_scan = True
camera, focus = setup_camera()
scan_name = "Cat"
supposed_points = [5]
if make_scan:
    image_set = get_gray_code_images()

    projector = get_monitors()[PROJECTOR_ID]

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def show_photo_projector(name_window: str, image: np.ndarray):
        cv2.namedWindow(name_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(name_window, projector.x, projector.y)
        cv2.imshow(name_window, image)
        cv2.waitKey(1000)
        if focus:
            for skipframes in range(10):
                camera.read()
                time.sleep(0.2)
        ret_, frame_ = camera.read()
        if not ret_:
            raise Exception("No frame returned")
        time.sleep(0.25)
        cv2.destroyWindow(name_window)
        return frame_

    all_frames = []
    for i, gray_code_image in enumerate(get_gray_code_images()):
        frames = show_photo_projector(f"{i}", gray_code_image)
        if not i:
            cv2.imwrite(os.path.join(PATH_ENCODED, scan_name, f"colored.png"), frames)
        gray_frame = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
        all_frames.append(gray_frame)
        if not os.path.exists(os.path.join(PATH_ENCODED, scan_name)):
            os.mkdir(os.path.join(PATH_ENCODED, scan_name))
        cv2.imwrite(os.path.join(PATH_ENCODED, scan_name, f"{i}.png"), gray_frame)
else:
    # adaptive so bad
    # contrast too
    all_frames = []
    for image in [os.path.join(PATH_ENCODED, scan_name, f"{i}.png") for i in range(46)]:
        gray_frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        all_frames.append(gray_frame)

for supposed_point in supposed_points:
    Ld, Lg = get_Ld_Lg(all_frames)
    h_codes, v_codes, h_images, v_images = get_Lited(all_frames, Ld, Lg, eps=supposed_point)

    if not os.path.exists(os.path.join(PATH_LITED, scan_name)):
        os.mkdir(os.path.join(PATH_LITED, scan_name))

    for i, h_im in enumerate(h_images):
        cv2.imwrite(os.path.join(PATH_LITED, scan_name, f"h_{i}.png"), h_im)

    for i, v_im in enumerate(v_images):
        cv2.imwrite(os.path.join(PATH_LITED, scan_name, f"v_{i}.png"), v_im)

    h_pixels = np.array([gray_to_decimal(h_codes[:, y, x])  for y in range(0, h_codes.shape[1]) for x in range(0, h_codes.shape[2])]).reshape((h_codes.shape[1], h_codes.shape[2]))
    v_pixels = np.array([gray_to_decimal(np.flip(v_codes[:, y, x]))  for y in range(0, v_codes.shape[1])for x in range(0, v_codes.shape[2])] ).reshape((v_codes.shape[1], v_codes.shape[2]))
    
    if not os.path.exists(os.path.join(PATH_DECODED, scan_name)):
        os.mkdir(os.path.join(PATH_DECODED, scan_name))
    np.save(os.path.join(PATH_DECODED, scan_name, 'h_pixels.npy'), h_pixels)
    np.save(os.path.join(PATH_DECODED, scan_name, 'v_pixels.npy'), v_pixels)

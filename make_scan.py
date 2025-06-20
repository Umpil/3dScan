import numpy as np
import cv2
import time
import glob
from screeninfo import get_monitors
from Constants import *
from utils import get_gray_code_images, get_Lited, get_Ld_Lg, gray_to_decimal, adaptive_illumination_correction, contrast
make_scan = False

if make_scan:
    image_set = get_gray_code_images()

    projector = get_monitors()[PROJECTOR_ID]
    calibre_proj_photo = cv2.imread("cal_proj.png")

    camera = cv2.VideoCapture(CAMERA_ID)
    if SPECIAL_VIDEO_WRITER:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*VIDEO_WRITER))

    if camera.get(cv2.CAP_PROP_AUTOFOCUS):
        FOCUS_SUPPORTED = True     
    else:
        FOCUS_SUPPORTED = False

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def show_photo_projector(cam: cv2.VideoCapture, name_window: str, image: np.ndarray, photo_set=False):
        cv2.namedWindow(name_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(name_window, projector.x, projector.y)
        cv2.imshow(name_window, image)
        cv2.waitKey(1000)
        if FOCUS_SUPPORTED:
            for skipframes in range(5):
                cam.read()
                time.sleep(0.3)
        if photo_set:
            ret_list = []
            for i in range(8):
                ret_, frame_ = cam.read()
                if not ret_:
                    raise Exception("No frame returned")
                ret_list.append(frame_)
                time.sleep(0.25)
                return ret_list
        ret_, frame_ = cam.read()
        if not ret_:
            raise Exception("No frame returned")
        time.sleep(0.25)
        cv2.destroyWindow(name_window)
        return [frame_]

    all_frames = []
    for i, gray_code_image in enumerate(get_gray_code_images()):
        frames = show_photo_projector(camera, f"{i}", gray_code_image)
        for frame in frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            all_frames.append(gray_frame)
            if not os.path.exists(os.path.join(PATH_SAVE_ENCODED,"HeadP")):
                os.mkdir(os.path.join(PATH_SAVE_ENCODED,"HeadP"))
            cv2.imwrite(os.path.join(PATH_SAVE_ENCODED,"HeadP", f"{i}.png"), gray_frame)

else:
    # adaptive so bad
    # contrast too
    all_frames = []
    for image in [os.path.join(PATH_SAVE_ENCODED, f"{i}.png") for i in range(46)]:#glob.glob(os.path.join(PATH_SAVE_ENCODED, "*.png")):
        gray_frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # gray_n = contrast(gray_frame)
        # cv2.imshow(image, cv2.resize(gray_n, (1920, 1080)))
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()
        all_frames.append(gray_frame)
for supposed_points in [7, 9, 10, 15]:
    print(supposed_points)
    now_name = f"Cat_plus{supposed_points}"
    Ld, Lg = get_Ld_Lg(all_frames)
    h_codes, v_codes, h_images, v_images = get_Lited(all_frames, Ld, Lg, eps=supposed_points)
    if not os.path.exists(os.path.join(PATH_SAVE_LITED, now_name)):
        os.mkdir(os.path.join(PATH_SAVE_LITED, now_name))
    for i, h_im in enumerate(h_images):
        cv2.imwrite(os.path.join(PATH_SAVE_LITED, now_name, f"h_{i}.png"), h_im)

    for i, v_im in enumerate(v_images):
        cv2.imwrite(os.path.join(PATH_SAVE_LITED, now_name, f"v_{i}.png"), v_im)

    h_pixels = np.array([gray_to_decimal(h_codes[:, y, x])  for y in range(0, h_codes.shape[1]) for x in range(0, h_codes.shape[2])]).reshape((h_codes.shape[1], h_codes.shape[2]))
    v_pixels = np.array([gray_to_decimal(np.flip(v_codes[:, y, x]))  for y in range(0, v_codes.shape[1])for x in range(0, v_codes.shape[2])] ).reshape((v_codes.shape[1], v_codes.shape[2]))
    if not os.path.exists(os.path.join(PATH_SAVE_DECODED, now_name)):
        os.mkdir(os.path.join(PATH_SAVE_DECODED, now_name))
    np.save(os.path.join(PATH_SAVE_DECODED, now_name, 'h_pixels.npy'), h_pixels)
    np.save(os.path.join(PATH_SAVE_DECODED, now_name, 'v_pixels.npy'), v_pixels)

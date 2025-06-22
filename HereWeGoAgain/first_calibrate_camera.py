import numpy as np
import cv2
import os
import time
import glob
from constants import *

camera, AUTO_FOCUS = setup_camera()

make_photos = True

detector_parameters = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
charuco_detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

def detect_charuco(gray):
        ret = False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_parameters)
        allCorners = []
        allIds = []

        if corners is not None and len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

            if charuco_ids is not None and len(charuco_ids) > 0:
                allCorners = charuco_corners
                allIds = charuco_ids

        return ret, allCorners, allIds

all_corners = []
all_ids = []
if not make_photos:
    for fname in glob.glob(os.path.join(PATH_CAMERA_CALIBRE, "*png")):
        image = cv2.imread(fname)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, now_corners, now_ids = detect_charuco(gray_image)

        if ret:
            all_corners.append(now_corners)
            all_ids.append(now_ids)
            print("Good image:", fname)
        else:
            print("Bad image:", fname)
    
    print("Camera calibration")
    mat_init = np.array([[ 1000.,    0., CAMERA_WIDTH/2.],
                                [    0., 1000., CAMERA_HEIGHT/2.],
                                [    0.,    0.,           1.]])
    dist_init = np.zeros((5,1))

    ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, 
                                                                            all_ids, board, 
                                                                            (CAMERA_WIDTH, CAMERA_HEIGHT), 
                                                                            mat_init, 
                                                                            dist_init,
                                                                            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    np.save(os.path.join(PATH_PARAMETERS, "cam_mtx.npy"), cam_mtx)
    np.save(os.path.join(PATH_PARAMETERS, "cam_dist.npy"), cam_dist)
else:
    for i in range(CAMERA_CALIBRE_PHOTO):
        good_photo = False
        while not good_photo:
            ret, frame = camera.read()
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            ret, now_corners, now_ids = detect_charuco(gray_image)
            if ret:
                good_photo = True
                print(f"Photo {i+1} getted")
                cv2.imwrite(os.path.join(PATH_CAMERA_CALIBRE, f"{i}.png"), frame)
                all_corners.append(now_corners)
                all_ids.append(now_ids)
                time.sleep(1)
            time.sleep(0.25)
    camera.release()
    print("Camera calibration")
    mat_init = np.array([[ 1000.,    0., CAMERA_WIDTH/2.],
                                [    0., 1000., CAMERA_HEIGHT/2.],
                                [    0.,    0.,           1.]])
    dist_init = np.zeros((5,1))

    ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, 
                                                                            all_ids, board, 
                                                                            (CAMERA_WIDTH, CAMERA_HEIGHT), 
                                                                            mat_init, 
                                                                            dist_init,
                                                                            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    np.save(os.path.join(PATH_PARAMETERS, "cam_mtx.npy"), cam_mtx)
    np.save(os.path.join(PATH_PARAMETERS, "cam_dist.npy"), cam_dist)

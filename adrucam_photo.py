import glob
import os
import cv2
import numpy as np
from Constants import *

detector_parameters = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
charuco_detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

cam_mtx = None
cam_dist = None

def detect_markers(frame, gray, draw=True):
        rvec = None
        tvec = None
        ret = False
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_parameters)
        allCorners = []
        allIds = []
        if corners is not None and len(corners) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if charuco_ids is not None and len(charuco_ids) > 0:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

                allCorners = charuco_corners
                allIds = charuco_ids

        return ret, frame, allCorners, allIds

def calibrate(calib_flags=cv2.CALIB_USE_INTRINSIC_GUESS):
        allCorners = []
        allIds = []

        images = glob.glob(os.path.join(PATH_CAMERA_CALIBRE, '*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            print('Reading image:', fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, _, corners, ids = detect_markers(img, gray, draw=False)

            if ret:
                allCorners.append(corners)
                allIds.append(ids)
            else:
                print('Bad image:', fname)

        print('Camera calibration')
        cameraMatrixInit = np.array([[ 1000.,    0., CAMERA_WIDTH/2.],
                                        [    0., 1000., CAMERA_HEIGHT/2.],
                                        [    0.,    0.,           1.]])
        distCoeffsInit = np.zeros((5,1))


        ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, (CAMERA_WIDTH, CAMERA_HEIGHT), cameraMatrixInit, distCoeffsInit, flags = calib_flags, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        print(cam_mtx)
        print(cam_dist)
        print('Error', ret)

        return cam_mtx, cam_dist


camera_mtx, camera_dist = calibrate()
np.save(os.path.join(PATH_PARAMETERS, "cam_mtx.npy"), camera_mtx)
np.save(os.path.join(PATH_PARAMETERS, "cam_dist.npy"), camera_dist)
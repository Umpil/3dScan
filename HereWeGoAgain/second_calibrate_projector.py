import numpy as np
import cv2
import os
import time
import glob
from screeninfo import get_monitors
from constants import *

camera, AUTO_FOCUS = setup_camera()
cam_mtx = np.load(os.path.join(PATH_PARAMETERS, "cam_mtx.npy"))
cam_dist = np.load(os.path.join(PATH_PARAMETERS, "cam_dist.npy"))
make_photos = True

projector = get_monitors()[PROJECTOR_ID]
calibre_proj_photo = cv2.imread("cal_proj.png")

detector_parameters = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
charuco_detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

def show_photo_projector(name_window: str, image: np.ndarray):
    cv2.namedWindow(name_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(name_window, projector.x, projector.y)
    cv2.imshow(name_window, image)
    cv2.waitKey(1000)
    good_photo = False
    while not good_photo:
        ret, frame = camera.read()
        if not ret:
            return None, None
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        H = detect_charuco_homography(gray_frame)
        if H:
            ret, circle_grid_2d, circle_grid_3d = find_circles(gray_frame, H)
            if ret:
                print("Getted image:", int(name_window) + 1)
                good_photo = True
                cv2.imwrite(os.path.join(PATH_PROJECTOR_CALIBRE, name_window, ".png"), frame)
                cv2.destroyWindow(name_window)
                return circle_grid_2d, circle_grid_3d
    cv2.destroyWindow(name_window)
    return None, None


def detect_charuco_homography(gray_frame_):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    charuco_corners, charuco_ids, rejected = cv2.aruco.detectMarkers(gray_frame_, dictionary, parameters=detector_parameters)
    Homography = None
    if charuco_corners:
        for corner in charuco_corners:
            cv2.cornerSubPix(gray_frame_, corner, (3, 3), (-1, -1), criteria)
        
        if charuco_ids:
            object_points = np.reshape(np.array(board.getObjPoints())[:, :, :, :2], len(charuco_corners) * 4, 2)
            flat_corners = np.reshape(charuco_corners, len(charuco_corners) * 4, 2)
            Homography, _ = cv2.findHomography(flat_corners, object_points, cv2.RANSAC, 4.0)
    
    return Homography

def find_circles(gray_frame_, homography_):
    ret_, circle_grid_ = cv2.findCirclesGrid(gray_frame_, CIRCLE_GRID_SIZE, cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret_:
        circle_grid_3d_ = cv2.perspectiveTransform(circle_grid_, homography_)
        circle_grid_3d_ = np.pad(circle_grid_3d_, ((0,0), (0,0), (0,1)), 'constant', constant_values=0)
        circle_grid_3d_ = circle_grid_3d_.astype(np.float64)
    return ret_, circle_grid_, circle_grid_3d_

projector_circles = []
projector_circles_objects = []
camera_circles = []
if not make_photos:
    for fname in glob.glob(os.path.join(PATH_PROJECTOR_CALIBRE, "*png")):
        image = cv2.imread(fname)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H = detect_charuco_homography(gray_image)
        if H:
            ret, now_circles_2d, now_circles_3d = find_circles(gray_image, H)
            projector_circles.append(now_circles_2d)
            projector_circles_objects.append(now_circles_3d)
            camera_circles.append(now_circles_2d)
            print("Good image:", fname)
        else:
            print("Bad image:", fname)
else:
    for i in range(PROJECTOR_COUNT_PHOTO):
        now_circles_2d, now_circles_3d = show_photo_projector(f"{i}", calibre_proj_photo)
        if now_circles_2d:
            projector_circles.append(now_circles_2d)
            projector_circles_objects.append(now_circles_3d)
            camera_circles.append(now_circles_2d)

camera.release()

proj_mtx = np.array([
    [ 3000.,    0., PROJECTOR_WIDTH/2.],
    [    0., 3000., PROJECTOR_HEIGHT/2.],
    [    0.,    0., 1.]])

ret, proj_mtx, proj_dist, _ = cv2.calibrateCamera(projector_circles_objects,
                                                  projector_circles,
                                                  (PROJECTOR_WIDTH, PROJECTOR_HEIGHT),
                                                  proj_mtx,
                                                  None,
                                                  flags=cv2.CALIB_USE_INTRINSIC_GUESS)

error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T,_,_ = cv2.stereoCalibrate(projector_circles_objects, 
                                                                                        camera_circles, 
                                                                                        projector_circles, 
                                                                                        cam_mtx, 
                                                                                        cam_dist, 
                                                                                        proj_mtx, 
                                                                                        proj_dist, 
                                                                                        (CAMERA_WIDTH, CAMERA_HEIGHT), 
                                                                                        flags=cv2.CALIB_FIX_INTRINSIC)

np.save(os.path.join(PATH_PARAMETERS, "proj_mtx.npy"), proj_mtx)
np.save(os.path.join(PATH_PARAMETERS, "proj_dist.npy"), proj_dist)
np.save(os.path.join(PATH_PARAMETERS, "proj_R.npy"), proj_R)
np.save(os.path.join(PATH_PARAMETERS, "proj_T.npy"), proj_T)
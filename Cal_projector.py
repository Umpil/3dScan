import cv2
import numpy as np
import time
from screeninfo import get_monitors
from Constants import *
from utils import process_focus

CIRCLE_RADIUS = 45
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
cam_mtx = np.load(os.path.join(PATH_PARAMETERS, "cam_mtx.npy"))
cam_dist = np.load(os.path.join(PATH_PARAMETERS, "cam_dist.npy"))

detector_parameters = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
charuco_detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)


def show_photo_projector(cam: cv2.VideoCapture, name_window: str, image: np.ndarray, photo_set=False):
    """
        Parameters:
        ---------
        cam : VideoCapture
            camera instance
        name_window : str
            window name
        image : Mat
            image to show

        Returns:
        -------
        ret : bool
            True if at least one marker and the board were detected
        frame : Mat
            Image with detected markers
        H : Mat
            Homography matrix of the camera to the charuco board
    """
    cv2.namedWindow(name_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(name_window, projector.x, projector.y)
    cv2.imshow(name_window, image)
    cv2.waitKey(2000)
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

def detect_charuco_homography(gray_frame_):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    charuco_corners, charuco_ids, rejected = cv2.aruco.detectMarkers(gray_frame_, dictionary, parameters=detector_parameters)
    Homography = None
    if charuco_corners:
        for corner in charuco_corners:
            cv2.cornerSubPix(gray_frame_, corner, (3, 3), (-1, -1), criteria)
        
        ret, corners, ids = cv2.aruco.interpolateCornersCharuco(charuco_corners, charuco_ids, gray_frame_, board)
        if charuco_ids:
            ret, rot_vector, trans_vector = cv2.aruco.estimatePoseCharucoBoard(corners, ids, board, cam_mtx, cam_dist, None, None)
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


print("CALIBRE PROJECTOR, Photos to take:", PROJECTOR_COUNT_PHOTO)
projector_circles = []
projector_circles_objects = []
camera_circles = []
for i in range(PROJECTOR_COUNT_PHOTO):
    print("PHOTO:", i)
    good_position = False
    
    while not good_position:
        frames = show_photo_projector(camera, f"Photo {i}", calibre_proj_photo, photo_set=True)
        gray_frames = []
        for j, frame in enumerate(frames):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            homography_matrix = detect_charuco_homography(gray_frame)
            if homography_matrix:
                ret, circle_grid_2d, circle_grid_3d = find_circles(gray_frame, homography_matrix)
                if ret:
                    print("Good image:", i, j)
                    cv2.imwrite(os.path.join(PATH_SAVE_CALIBRE_PROJECTOR, f"{i}_{j}.png"))
                    projector_circles.append(circle_grid_2d)
                    projector_circles_objects.append(circle_grid_3d)
                    camera_circles.append(circle_grid_2d)
                    good_position = True

camera.release()
proj_mtx = np.array([
    [ 3000.,    0., PROJECTOR_WIDTH/2.],
    [    0., 3000., PROJECTOR_HEIGHT/2.],
    [    0.,    0., 1.]])

ret, proj_mtx, proj_dist, _ = cv2.calibrateCamera(projector_circles_objects, projector_circles, (PROJECTOR_WIDTH, PROJECTOR_HEIGHT), proj_mtx, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T,_,_ = cv2.stereoCalibrate(projector_circles_objects, camera_circles, projector_circles, cam_mtx, cam_dist, proj_mtx, proj_dist, (2560,1440), flags=cv2.CALIB_FIX_INTRINSIC)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam_mtx, cam_dist, proj_mtx, proj_dist, (CAMERA_WIDTH, CAMERA_HEIGHT), proj_R, proj_T)
np.save(os.path.join(PATH_PARAMETERS, "proj_mtx.npy"), proj_mtx)
np.save(os.path.join(PATH_PARAMETERS, "proj_dist.npy"), proj_dist)
np.save(os.path.join(PATH_PARAMETERS, "proj_R.npy"), proj_R)
np.save(os.path.join(PATH_PARAMETERS, "proj_T.npy"), proj_T)
np.save(os.path.join(PATH_PARAMETERS, "R1.npy"), R1)
np.save(os.path.join(PATH_PARAMETERS, "R2.npy"), R2)
np.save(os.path.join(PATH_PARAMETERS, "P1.npy"), P1)
np.save(os.path.join(PATH_PARAMETERS, "P2.npy"), P2)
np.save(os.path.join(PATH_PARAMETERS, "Q.npy"), Q)


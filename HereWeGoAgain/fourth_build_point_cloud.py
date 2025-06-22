import cv2
import os
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from utils import plot_cloud
from open3d import core as o3c
from constants import *
from scipy import stats


proj_w, proj_h = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
calib_proj_w, calib_proj_h = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
cam_w, cam_h = (CAMERA_WIDTH, CAMERA_HEIGHT)
now_name = "Cat_plus5_res"
cam_mtx = np.load(os.path.join(PATH_PARAMETERS,'cam_mtx.npy'))
cam_dist = np.load(os.path.join(PATH_PARAMETERS,'cam_dst.npy'))
proj_mtx = np.load(os.path.join(PATH_PARAMETERS,'proj_mtx.npy'))
proj_dist = np.load(os.path.join(PATH_PARAMETERS,'proj_dst.npy'))
proj_R = np.load(os.path.join(PATH_PARAMETERS,'proj_R.npy'))
proj_T = np.load(os.path.join(PATH_PARAMETERS,'proj_T.npy'))
h_pixels = np.load(os.path.join(PATH_DECODED, now_name, 'h_pixels.npy'))
v_pixels = np.load(os.path.join(PATH_DECODED, now_name, 'v_pixels.npy'))

#color_image = cv2.imread(os.path.join(PATH_SAVE_ENCODED, "0.png"))
color_image = cv2.imread("colored.png")

def get_colors_camproj(img_color=color_image):
        cam_pts = []
        proj_pts = []
        colors = []
        for i in range(CAMERA_WIDTH): # CAM_W
            for j in range(CAMERA_HEIGHT): # CAM_H
                h_value = h_pixels[j, i]
                v_value = v_pixels[j, i]
                if h_value == -1 or v_value == -1:
                    continue
                else:
                    cam_pts.append([i,j])
                    h_value = min(PROJECTOR_WIDTH - 1, h_value)
                    v_value = min(PROJECTOR_HEIGHT - 1, v_value)
                    proj_pts.append([h_value, v_value])
                    if img_color is not None:
                        colors.append(img_color[j, i, :])

        cam_pts = np.array(cam_pts, dtype=np.float32)
        proj_pts = np.array(proj_pts, dtype=np.float32)
        if img_color is not None:
            colors_array = np.array(colors).astype(np.float64)/255.0

        return cam_pts, proj_pts, colors_array

def triangulate(cam_pts, proj_pts):
        cam_pts_homo = cv2.convertPointsToHomogeneous(cv2.undistortPoints(np.expand_dims(cam_pts, axis=1), cam_mtx, cam_dist, R=proj_R))[:,0].T
        proj_pts_homo = cv2.convertPointsToHomogeneous(cv2.undistortPoints(np.expand_dims(proj_pts, axis=1), proj_mtx, proj_dist))[:,0].T
        T = proj_T[:,0]

        TLen = np.linalg.norm(T)
        NormedL = cam_pts_homo/np.linalg.norm(cam_pts_homo, axis=0)
        alpha = np.arccos(np.dot(-T, NormedL)/TLen)
        beta = np.arccos(np.dot(T, proj_pts_homo)/(TLen*np.linalg.norm(proj_pts_homo, axis=0)))
        gamma = np.pi - alpha - beta
        P_len = TLen*np.sin(beta)/np.sin(gamma)
        Pts = NormedL*P_len

        return Pts

def filter_3d_pts(Pts, colors, threshold=.5, threshold_zscore=3.0):
    filter = (Pts[2] < threshold) & (Pts[2] > -threshold) & (Pts[1] < threshold) & (Pts[1] > -threshold) & (Pts[0] < threshold) & (Pts[0] > -threshold)
    Pts_filtered = Pts[:, filter]
    colors_filtered = colors[filter]
    #Pts_filtered_t = Pts_filtered.T
    # Pts_filtered_t = Pts.T
    # z_scores = np.abs(stats.zscore(Pts_filtered_t, axis=0))
    # filtered_mask = (z_scores < threshold_zscore).all(axis=1)
    # Pts_filtered = Pts[:, filtered_mask]
    # colors_filtered = colors[filtered_mask]
    return Pts_filtered, colors_filtered

cam_points, proj_points, colors = get_colors_camproj(color_image)
points_3d = triangulate(cam_points, proj_points)
points_3d, colors= filter_3d_pts(points_3d, colors)
plot_cloud(points_3d, colors)

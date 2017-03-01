#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:09:34 2017

@author: Champ
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

class Camera():
    """
    A class representing a dash-cam
    """

    # source points in dash-cam perspective
    SRC_POINTS = np.array([
            [265, 683],
            [560, 480],
            [729, 480],
            [1049, 683]],
            dtype=np.float32)

    # destination points in top-view perspective
    DST_POINTS = np.array([
            [400, 700],
            [400, 50],
            [860, 50],
            [860, 700]],
            dtype=np.float32)

    def __init__(self):
        self.PERSPECTIVE_TRANS_MATRIX = cv2.getPerspectiveTransform(
                self.SRC_POINTS, self.DST_POINTS)
        self.INV_PERSPECTIVE_TRANS_MATRIX = cv2.getPerspectiveTransform(
                self.DST_POINTS, self.SRC_POINTS)

    def calibrate_camera(self, cal_images, nx=9, ny=6, show_img=-1):
        """
        This function computes the camera calibration matrix and distortion
        coefficients given a set of chessboard images.

        params:
            cal_images - path and file name pattern (to be used with glob) that point
                to the calibration images of chessboard
                nx - number of inner corners in x
                ny - number of inner corners in y
                show_img - if > 0, will show the image at that numerical position with
                detected corners drawn on it for debugging purpose.
        returns:
            mtx - camera's transformation matrix
            dist - camera's distortion coefficients
        """
        print("Computing camera's distortion")

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

    #    imshown = False
        images = glob.glob(cal_images)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(img_gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                print("Found corners for %i th image" % (idx))
                objpoints.append(objp)
                imgpoints.append(corners)

            if idx == show_img:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.figure()
                plt.imshow(img)
#               cv2.imshow("chessboard corners", img)
#               cv2.imwrite("camera_cal_corners.jpg", img)
#               imshown = True

        print("Computing camera and distortion matrices")
        # Assume that the size of all calibration images are the same, I'm computing
        # the image size from the last one
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None)

#       if imshown:
#           cv2.waitKey(1000)
#           cv2.destroyAllWindows()

        # TODO actually the mtx and dist should be camera's properties too.
        return mtx, dist

    def undistort_image(self, img, mtx, dist):
        """
        Undistorts the given image using camera's transformation matrix and
        distortion coefficients
        """
        return cv2.undistort(img, mtx, dist, None, mtx)


    def get_top_view(self, img, plot_fig=False):
        """
        Transform the given image to a top-view perspective
        """
        h = img.shape[0]
        w = img.shape[1]
        img_bin_top_view = cv2.warpPerspective(
                img, self.PERSPECTIVE_TRANS_MATRIX, (w, h),
                flags=cv2.INTER_LINEAR)

        if plot_fig:
            plt.figure()
            plt.title("Bird-eyed perspective")
            plt.imshow(img_bin_top_view)

        return img_bin_top_view

    def get_front_view(self, img, plot_fig=False):
        """
        Transform the given image to a front-view perspective
        """
        h = img.shape[0]
        w = img.shape[1]
        img_front_view = cv2.warpPerspective(
                img, self.INV_PERSPECTIVE_TRANS_MATRIX, (w, h),
                flags=cv2.INTER_LINEAR)

        if plot_fig:
            plt.figure()
            plt.title("Front view perspective")
            plt.imshow(img_front_view)

        return img_front_view

    def undistort_and_draw_overlay(self, img, mtx, dist, overlay, plot_fig=False):
        """
        Undistort the given image and put the given overlay on it
        """
        img_undist = self.undistort_image(img, mtx, dist)
        result = cv2.addWeighted(img_undist, 1, overlay, 0.3, 0)

        if plot_fig:
            plt.figure()
            plt.title("Overlay image")
            plt.imshow(result)

        return result





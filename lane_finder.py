#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:48:00 2017

@author: Champ
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

VERTICAL_DIR = np.pi/2

SOBEL_KERNEL = 31
GRAD_X_THRESH = (245, 255)
GRAD_Y_THRESH = (245, 255)
GRAD_MAG_THRESH = (30, 255)
GRAD_DIR_THRESH_1 = (-np.pi, -0.6 * np.pi)
GRAD_DIR_THRESH_2 = (-0.4 * np.pi, 0.4 * np.pi)
GRAD_DIR_THRESH_3 = (0.6 * np.pi, np.pi)

THRESH_H = (10, 35)
THRESH_L = (0, 255)
THRESH_S = (90, 255)

CONV_SEGMENTS = 9
CONV_KERNEL = (100, 80)
CONV_THRESHOLD = 1500

YM_PER_PIX = 10 / (640 - 555)  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / (860 - 480) # meters per pixel in x dimension
Y_EVAL_LOCATION = 0.75
Y_CENTER_EVAL_LOCATION = 0.8

class LaneLine():

    def __init__(self, raw_dots, poly_fit, curv, y, fitx):
        self.raw_dots = raw_dots
        self.poly_fit = poly_fit
        self.curv = curv
        self.y = y
        self.fitx = fitx

def _abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Calculate directional gradient and return binary image of pixels within the
    given thresholds
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel = np.zeros_like(img_gray)

    if orient == "x":
        img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
    elif orient == "y":
        img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, None, sobel_kernel)
    else:
        raise ValueError()

    img_sobel = np.uint8(255 * img_sobel / img_sobel.max())
    binary_output = np.zeros_like(img_sobel)
    binary_output[(img_sobel > thresh[0]) & (img_sobel < thresh[1])] = 1
    return binary_output

def _mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Calculate gradient magnitude and return binary image of pixels within the
    given threshold
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, None, sobel_kernel)
    img_sobel = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
    img_sobel = np.uint8(255 * img_sobel / img_sobel.max())
    binary_output = np.zeros_like(img_sobel)
    binary_output[(img_sobel > mag_thresh[0]) & (img_sobel < mag_thresh[1])] = 1
    return binary_output

def _dir_threshold(img, sobel_kernel=3, thresh1=GRAD_DIR_THRESH_1,
        thresh2=GRAD_DIR_THRESH_2, thresh3=GRAD_DIR_THRESH_3):
    """
    Calculate gradient direction and return binary image of pixels within the
    given thresholds

    Note that you have 3 thresholds to cover the angles that should include the
    gradient vectors of interested from -pi to pi.

    params
        thresh1 - the first angle interval (-pi, pi)
        thresh2 - the second angle interval (-pi, pi)
        thresh3 - the third angle interval (-pi, pi)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, None, sobel_kernel)
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, None, sobel_kernel)
    img_grad_dir = np.arctan2(np.abs(img_sobel_y), np.abs(img_sobel_x))

    binary_output = np.zeros_like(img_grad_dir)
    binary_output[
            ((img_grad_dir > thresh1[0]) & (img_grad_dir < thresh1[1])) |
            ((img_grad_dir > thresh2[0]) & (img_grad_dir < thresh2[1])) |
            ((img_grad_dir > thresh3[0]) & (img_grad_dir < thresh3[1]))
            ] = 1
    return binary_output

def _find_lane_by_gradient(img_undist, plot_fig):
    """
    Find lane lines from the given undistorted image and return a binary image
    of detected lane lines combining results from various gradient techniques
    """
    # Compute binary filters based on various gradient techniques
    img_bin_grad_x = _abs_sobel_thresh(
            img_undist, orient='x', sobel_kernel=SOBEL_KERNEL,
            thresh=GRAD_X_THRESH)
    img_bin_grad_y = _abs_sobel_thresh(
            img_undist, orient='y', sobel_kernel=SOBEL_KERNEL,
            thresh=GRAD_Y_THRESH)
    img_bin_grad_mag = _mag_thresh(
            img_undist, sobel_kernel=SOBEL_KERNEL, mag_thresh=GRAD_MAG_THRESH)
    img_bin_grad_dir = _dir_threshold(
            img_undist, sobel_kernel=SOBEL_KERNEL, thresh1=GRAD_DIR_THRESH_1,
            thresh2=GRAD_DIR_THRESH_2, thresh3=GRAD_DIR_THRESH_3)

    # Combine gradiant-based binary filters
    img_bin_grad = np.zeros_like(img_bin_grad_x)
    img_bin_grad[
#            ((img_bin_grad_x == 1) & (img_bin_grad_y == 1)) |
            ((img_bin_grad_mag == 1) & (img_bin_grad_dir == 1))
            ] = 1

    if plot_fig:
        plt.figure()
        plt.subplot(221)
        plt.title("Grad x")
        plt.imshow(img_bin_grad_x)
        plt.subplot(222)
        plt.title("Grad y")
        plt.imshow(img_bin_grad_y)
        plt.subplot(223)
        plt.title("Grad mag")
        plt.imshow(img_bin_grad_mag)
        plt.subplot(224)
        plt.title("Grad dir")
        plt.imshow(img_bin_grad_dir)
        plt.show()

        plt.figure()
        plt.title("Combined grad results")
        plt.imshow(img_bin_grad)

    return img_bin_grad

def _find_lane_by_hls(img_undist, plot_fig, thresh_h=THRESH_H, thresh_l=THRESH_L,
        thresh_s=THRESH_S):
    """
    Find lane lines by analysing the image in HLS color space and return a
    binary image combining results from appropriate thresholding of those color
    space elements

    params
        thresh_h - within (0, 180)
        thresh_l - within (0, 255)
        thresh_s - within (0, 255)
    """
    hls = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    img_bin_h = np.zeros_like(h)
    img_bin_h[(h > thresh_h[0]) & (h < thresh_h[1])] = 1

    img_bin_l = np.zeros_like(h)
    img_bin_l[(l > thresh_l[0]) & (l < thresh_l[1])] = 1

    img_bin_s = np.zeros_like(h)
    img_bin_s[(s > thresh_s[0]) & (s < thresh_s[1])] = 1

    # Combine HLS filters
    img_bin_hls= np.zeros_like(img_bin_h)
    img_bin_hls[(img_bin_h == 1) & (img_bin_l == 1) & (img_bin_s == 1)] = 1

    if plot_fig:
        plt.figure()
        plt.subplot(321)
        plt.title("H")
        plt.imshow(h)
        plt.subplot(322)
        plt.title("H thresholded")
        plt.imshow(img_bin_h)

        plt.subplot(323)
        plt.title("L")
        plt.imshow(l)
        plt.subplot(324)
        plt.title("L thresholded")
        plt.imshow(img_bin_l)

        plt.subplot(325)
        plt.title("S")
        plt.imshow(s)
        plt.subplot(326)
        plt.title("S thresholded")
        plt.imshow(img_bin_s)

        plt.figure()
        plt.title("Combined HLS result")
        plt.imshow(img_bin_hls)

    return img_bin_hls

def _mask_ROI(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 1)
    return cv2.bitwise_and(mask, img)

def _combine_grad_hls(img_bin_grad, img_bin_hls, plot_fig):
    """
    Combine 2 binary images with OR operation
    """
    img_bin_combined = np.zeros_like(img_bin_grad)
    img_bin_combined[(img_bin_grad == 1) | (img_bin_hls == 1)] = 1

    if plot_fig:
        plt.figure()
        plt.title("Combined lane line result")
        plt.imshow(img_bin_combined)

    return img_bin_combined

def _convolute(img_segment, kernel):
    hist = []
    width = img_segment.shape[1]
    kernel_w = kernel[1]

    # convolve the kernel from left to right over the image segment
    for index in range(width):
        min_w = int(max(0, index - (kernel_w / 2)))
        max_w = int(min(width, index + (kernel_w / 2)))
        conv_window = np.zeros_like(img_segment)
        conv_window[:, min_w: max_w] = 1
        conv_value = np.sum(conv_window * img_segment)
        hist.append(conv_value)

    return np.array(hist)

def _get_previous_line_center(prev_lines, current_line, y_eval):
    """
    Compute previous line center, evaluated at the given y_eval
    """
    if len(prev_lines) == 0:
        return current_line
    else:
        mean_center = 0
        for line in prev_lines:
            mean_center = mean_center + line.fitx[y_eval]
        return mean_center / len(prev_lines)

def _find_locations_of_peak(var_array, start, stop):
    start = int(start)
    stop = int(stop)
    val_list = var_array.tolist()
    peak_value = max(val_list[start: stop])
    locations = []
    loop_start = start

    if peak_value > CONV_THRESHOLD:
        while (loop_start <= stop):
            try:
                location = val_list.index(peak_value, loop_start, stop)
                locations.append(location)
                loop_start = location + 1
            except:
                break

    return np.array(locations)


def _find_lane_by_convolution(img, segments, kernel, prev_left, prev_right,
        plot_fig=False):
    """
    Find coordinates of left and right lane lines for each segment by
        - running convolution kernel from left to right ove the givem (binary)
          image.
        - pick x values that have the highest peaks (one for left, one for right)
          as representatives for that segment's y value.
        - update lane center and search ranges for left / right lane lines.

    params:
        segment - number of vertical segments to do convolution
        kernel - (h, w) of the convolution kernel
    returns:
        lane_left, lane_right - np arrays for left / right lane line
            coordinates [[h, w], [h, w], ...]
    """
    img_w = img.shape[1]
    img_h = img.shape[0]
    lane_left = []
    lane_right = []

    # starting locations (will be overriden by past detection history if given)
    current_left = img_w / 3
    current_right = 2 * img_w / 3

    # search window width
    window_width = 200

    # height between segments
    img_h_segment_offset = np.floor((img_h - kernel[0]) / (segments - 1))

    # iterates backward to start from the bottom of the img first
    for index in range(segments - 1, -1, -1):
        # compute convolution values along the width of this segment
        min_h = int(index * img_h_segment_offset)
        max_h = int(min_h + kernel[0])
        center_h = (min_h + max_h) / 2
        segment_hist = _convolute(img[min_h: max_h, :], kernel)

        # compute likely left / right lane line centers from previous history
        current_left = _get_previous_line_center(
                prev_left, current_left, int(center_h))
        current_right = _get_previous_line_center(
                prev_right, current_right, int(center_h))

        # get peak locations for left / right lane lines
        locations_left = _find_locations_of_peak(
                segment_hist,
                current_left - (window_width / 2),
                current_left + (window_width / 2))
        locations_right = _find_locations_of_peak(
                segment_hist,
                current_right - (window_width / 2),
                current_right + (window_width / 2))

        # if lane line is detected, update position
        if len(locations_left) > 0:
            avg_location_left = np.average(locations_left)
            current_left = avg_location_left
        else:
            avg_location_left = current_left

        if len(locations_right) > 0:
            avg_location_right = np.average(locations_right)
            current_right = avg_location_right
        else:
            avg_location_right = current_right

        # store in arrays
        lane_left.append([center_h, avg_location_left])
        lane_right.append([center_h, avg_location_right])

    img_lane_lines = img

    if plot_fig:
        plt.figure()
        plt.title("Convolution result on a segment")
        plt.plot(segment_hist)

        # Visualization on top-view image
        img_lane_lines = np.dstack((img, img, img)) * 255

        # Plot raw dots with red
        for dot in lane_left:
            img_lane_lines = cv2.circle(
                    img_lane_lines, (int(dot[1]), int(dot[0])), 10,
                    (255, 0, 0), 10)
        for dot in lane_right:
            img_lane_lines = cv2.circle(
                    img_lane_lines, (int(dot[1]), int(dot[0])), 10,
                    (0, 0, 255), 10)

        plt.figure()
        plt.title("Bird-eyed perspective with detected lane lines")
        plt.imshow(img_lane_lines)

    return np.array(lane_left), np.array(lane_right), img_lane_lines

def _fit_lane_lines(lane_left, lane_right, img, plot_fig=False,
        real_scale=False):
    """
    Fit a 2nd-degree polynomial to given left / right lane points

    params:
        real_scale - True to scale coordinates up to real-world scale
    return:
        left_fit
        right_fit
        offset - car_offset from the middle of the lane (in pixel unit if
            real_scale is False, meter if True
    """

    # Note that we use height (y) as the variable and width (x) as the dependent
    # variable instead.
    if real_scale:
        left_fit = np.polyfit(
                YM_PER_PIX * lane_left[:, 0], XM_PER_PIX * lane_left[:, 1], 2)
        right_fit = np.polyfit(
                YM_PER_PIX * lane_right[:, 0], XM_PER_PIX * lane_right[:, 1], 2)
        y_eval = YM_PER_PIX * Y_CENTER_EVAL_LOCATION * img.shape[0]
        car_center_x = XM_PER_PIX * img.shape[1] / 2
    else:
        left_fit = np.polyfit(lane_left[:, 0], lane_left[:, 1], 2)
        right_fit = np.polyfit(lane_right[:, 0], lane_right[:, 1], 2)
        y_eval = Y_CENTER_EVAL_LOCATION * img.shape[0]
        car_center_x = XM_PER_PIX * img.shape[1] / 2

    # Compute center offset
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_center_x = (left_x + right_x) / 2
    offset = np.abs(car_center_x - lane_center_x)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    if plot_fig & (img is not None):
        plt.figure()
        plt.title("Bird-eyed perspective with detected lane lines and their fit")
        plt.imshow(img)
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='blue')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit, offset, ploty, left_fitx, right_fitx

def _compute_radius_of_curvature(poly_fit, y):
    """
    Compute radius of curvature for the given poly_fit curve, evalulated at the
    given y position. Note that the y position is the primary variable and the
    x = f(y) is the dependent variable

    params:
        poly_fit - the polyfit for 2nd degree polynomial
    return:
        curv_rad - radius of curvature
    """
    return ((1 + ((2 * poly_fit[0] * y) + poly_fit[1])**2)**1.5) / np.absolute(2 * poly_fit[0])

def find_lane(img, mtx, dist, camera, prev_left, prev_right, plot_fig=False):
    """
    Find lane lines and overlay the found lane and other information on the
    given image of the road.

    params:
        img - the image in RGB
        mtx - camera's transformation matrix
        dist - camera's distortion coefficients
        camera - a Camera object
        prev_left / right - a list of previously detected lane lines
        plot_fig - If True, plots images of each step and saves to files
    """
    # Apply a distortion correction to raw images.
    img_undist = camera.undistort_image(img, mtx, dist)

    # Detect lane lines
    h = img_undist.shape[0]
    w = img_undist.shape[1]
    horizon_h = 450
    mask_vertices = np.array([[
        (0, h),
        (w * (5 / 12), horizon_h),
        (w * (8 / 12), horizon_h),
        (w, h)]],
        dtype=np.int32
    )
    img_bin_grad = _find_lane_by_gradient(img_undist, plot_fig)
    img_bin_hls = _find_lane_by_hls(img_undist, plot_fig)
    img_bin_combined = _combine_grad_hls(
            _mask_ROI(img_bin_grad, mask_vertices),
            _mask_ROI(img_bin_hls, mask_vertices),
            plot_fig)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    img_bin_top_view = camera.get_top_view(img_bin_combined, plot_fig)

    # Detect lane pixels and fit to find the lane boundary.
    lane_left_dots, lane_right_dots, img_lane_lines = _find_lane_by_convolution(
            img_bin_top_view, CONV_SEGMENTS, CONV_KERNEL, prev_left,
            prev_right, plot_fig)

    # Do polyfit on the dots and vehicle position with respect to center.
    left_fit, right_fit, _, ploty, left_fitx, right_fitx = _fit_lane_lines(
            lane_left_dots, lane_right_dots, img_lane_lines, plot_fig)
    # Do polyfit again with real-world scale
    left_fit_m, right_fit_m, offset_m, _, _, _ = _fit_lane_lines(
            lane_left_dots, lane_right_dots, img_lane_lines, False, True)

    # Determine the curvature of the lane
    y_eval = Y_EVAL_LOCATION * img.shape[0]
    lane_left_curv = _compute_radius_of_curvature(left_fit_m, y_eval)
    lane_right_curv = _compute_radius_of_curvature(right_fit_m, y_eval)

    lane_left = LaneLine(
            lane_left_dots, left_fit, lane_left_curv, ploty, left_fitx)
    lane_right = LaneLine(
            lane_right_dots, right_fit, lane_right_curv, ploty, right_fitx)

    return lane_left, lane_right, offset_m

def draw_lane_overlay(h, w, left_line, right_line):
    """
    Draw lane lines and area in between on a new RGB image
    """
    warped_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_y = left_line.y.astype(np.int)
    left_x = left_line.fitx.astype(np.int)
    right_y = right_line.y.astype(np.int)
    right_x = right_line.fitx.astype(np.int)

    pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warped_img, np.int_([pts]), (0,255, 0))

    return warped_img










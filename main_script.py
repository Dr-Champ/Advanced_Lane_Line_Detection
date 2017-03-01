#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main script that calls other helper functions / modules to go
through the entire process, from one-off camera calibrarion to processing of
test videos for lane line detection.

Created on Sun Feb 26 13:08:08 2017

@author: Champ
"""

import camera_util as cam
import lane_finder as lfd
import numpy as np
import pickle
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# ====================== Configurable params ========================

USE_PICKLED_DIST = True
PICKLE_FILE = "dist_pickle.p"
CAL_IMAGES = "data/camera_cal/calibration*.jpg"
TEST_IMAGE_FILE = None
#TEST_IMAGE_FILE = "data/test_images/test3.jpg"                 # set to None to use video
#TEST_IMAGE_FILE = "data/test_images/straight_lines2.jpg"       # set to None to use video
VIDEO_FILE = "data/project_video.mp4"
HIST_THRESHOLD = 6
LINE_SEPARATION_MISMATCH_LIMIT = 0.15

# ==================== Calibrating the camera =======================

camera = cam.Camera()

if USE_PICKLED_DIST:
    with open(PICKLE_FILE, "rb") as f:
        dist_pickle = pickle.load(f)
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
else:
    mtx, dist = camera.calibrate_camera(CAL_IMAGES, show_img = -1)
    # Save camera's calibration parameters for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(PICKLE_FILE, "wb"))

# ========================= Main workflow ===========================

prev_left_line = []
prev_right_line = []

def _compute_average_curv(prev_line):
    avg_curv = 0
    if len(prev_line) > 0:
        for line in prev_line:
            avg_curv = avg_curv + line.curv
        avg_curv = avg_curv / len(prev_right_line)
    return avg_curv

def _add_info_text(overlay, prev_left_line, prev_right_line, offset_m):
    avg_left_curv = _compute_average_curv(prev_left_line)
    avg_right_curv = _compute_average_curv(prev_right_line)
    info_str = "L-rad %.2f m, R-rad %.2f m, offset %.2f m" \
            % (avg_left_curv, avg_right_curv, offset_m)
    info_org = (int(0.1 * img.shape[1]), int(0.1 * img.shape[0]))
    cv2.putText(
            overlay, info_str, info_org, cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 3)

def _evaluate_detection_result(prev_left_line, prev_right_line, left_line,
        right_line):
    """
    Add the current_line into the prev_lines list if it passes sanity checks.
    Removes the oldest line away from the list nevertheless
    """
    # Determine line separation at both ends
    top_separation = right_line.fitx[0] - left_line.fitx[0]
    bottom_separation = right_line.fitx[-1] - left_line.fitx[-1]

    # See if it diverges too much LINE_SEPARATION_MISMATCH_LIMIT
    difference = np.absolute(top_separation - bottom_separation)
    min_separation = min(top_separation, bottom_separation)

    if (difference / min_separation) < LINE_SEPARATION_MISMATCH_LIMIT:
        # Pass sanity check
        prev_left_line.append(left_line)
        prev_right_line.append(right_line)

    # always remove the oldest line from the list if len exceeds HIST_THRESHOLD
    if len(prev_left_line) > HIST_THRESHOLD:
        prev_left_line.pop(0)

    if len(prev_right_line) > HIST_THRESHOLD:
        prev_right_line.pop(0)

def _detect_lane_and_draw_overlay(img, plot_fig=False):
    # Detection steps leveraging prior knowledge of the previous lane lines
    # - provide previous line locations (y, fitx) to the find_lane to use as
    #   initial search area for each convolutional segments
    # - sanity check. See if lines are parallel (distance between lines are
    #   pretty much the same at the bottom and top)
    # - if so, add these new lines to the running average
    left_line, right_line, offset_m = lfd.find_lane(
            img, mtx, dist, camera, prev_left_line, prev_right_line, plot_fig)

    _evaluate_detection_result(
            prev_left_line, prev_right_line, left_line, right_line)

    # Draw detected lane boundaries and other info back onto the original image
    warped_overlay = lfd.draw_lane_overlay(
            img.shape[0], img.shape[1], left_line, right_line)
    overlay = camera.get_front_view(warped_overlay, plot_fig)
    _add_info_text(overlay, prev_left_line, prev_right_line, offset_m)
    result_img = camera.undistort_and_draw_overlay(
            img, mtx, dist, overlay, plot_fig)
    return result_img

if TEST_IMAGE_FILE is None:
    print("Processing video %s" % (VIDEO_FILE))
    # TODO
    # loop and load one frame at a time. Feed a frame into the find_lane() and
    # check if lane lines were found. If so, update lane line info. If not
    # (inferred from doing sanity checks on the newly found lane lines), then
    # skipe this frame
    clip = VideoFileClip(VIDEO_FILE)
    overlay_clip = clip.fl_image(_detect_lane_and_draw_overlay)
    overlay_clip.write_videofile("result_video.mp4", audio=False)
else:
    img = mpimg.imread(TEST_IMAGE_FILE)
    print("Processing test image %s, size %i, %i"
          % (TEST_IMAGE_FILE, img.shape[0], img.shape[1]))
    _detect_lane_and_draw_overlay(img, True)












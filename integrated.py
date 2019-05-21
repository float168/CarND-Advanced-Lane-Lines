# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Advanced Lane Finding Project
#
# The goals / steps of this project are the following:
#
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#
# ---
# ## First, I'll compute the camera calibration using chessboard images

# +
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline


def calibrate_camera(rgb_images, chessboard_gridsize):
    if len(rgb_images) == 0:
        raise ValueError("Should not be empty: image_files")
        
    xsize, ysize = chessboard_gridsize
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ysize*xsize,3), np.float32)
    objp[:,:2] = np.mgrid[0:xsize,0:ysize].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for rgb in rgb_images:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (xsize,ysize), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    if len(imgpoints) == 0:
        raise ValueError("Not found valid chessboard images: image_files")
        
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        return mtx, dist
    else:
        raise RuntimeError("Failed 'cv2.calibrateCamera'")
        
# Make a list of calibration images
calib_image_files = glob.glob("data/camera_cal/*.jpg")
calib_rgb_images = [mpimg.imread(f) for f in calib_image_files]
camera_matrix, dist_coeff = calibrate_camera(calib_rgb_images, (9,6))

print(camera_matrix)
print(dist_coeff)


# +
def undistort_image(image, camera_matrix, dist_coeff):
    return cv2.undistort(image, camera_matrix, dist_coeff, None, camera_matrix)

test_image_files = glob.glob("data/test_images/*.jpg")
undist_rgb_images = [undistort_image(mpimg.imread(f), camera_matrix, dist_coeff) for f in test_image_files]

plt.imshow(undist_rgb_images[0])


# +
def apply_threshold(rgb_image):
    def abs_sobel_thresh(rgb, orient='x', sobel_kernel=3, thresh=(0,255)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sobel = None
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
        else:
            raise ValueError("Unknown orient type: {}".format(orient))
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        mask = np.zeros_like(gray)
        mask[(thresh[0] <= scaled_sobel) & (scaled_sobel <= thresh[1])] = 1
        return mask
    
    def mag_sobel_thresh(rgb, sobel_kernel=3, thresh=(0,255)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        x_sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
        y_sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
        mag_sobel = np.sqrt(x_sobel**2 + y_sobel**2)
        scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
        mask = np.zeros_like(gray)
        mask[(thresh[0] <= scaled_sobel) & (scaled_sobel <= thresh[1])] = 1
        return mask
    
    def dir_sobel_thresh(rgb, sobel_kernel=3, thresh=(0,np.pi/2)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        x_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel))
        y_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel))
        theta = np.arctan2(y_abs_sobel, x_abs_sobel)
        mask = np.zeros_like(gray)
        mask[(thresh[0] <= theta) & (theta <= thresh[1])] = 1
        return mask
    
    def hls_satur_thresh(rgb, thresh=(0,100)):
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
        S = hls[:,:,2]
        mask = np.zeros_like(S)
        mask[(S > thresh[0]) & (S >= thresh[1])] = 1
        return mask
    
    x_abs_sobel_bin = abs_sobel_thresh(rgb_image, orient='x', sobel_kernel=5, thresh=(80,150))
    y_abs_sobel_bin = abs_sobel_thresh(rgb_image, orient='y', sobel_kernel=5, thresh=(80,150))
    mag_sobel_bin   = mag_sobel_thresh(rgb_image, sobel_kernel=5, thresh=(80,150))
    dir_sobel_bin   = dir_sobel_thresh(rgb_image, sobel_kernel=15, thresh=(0.7, 1.3))
    hls_satur_bin   = hls_satur_thresh(rgb_image, thresh=(60,90))
    
    shape = rgb_image.shape[:2]
    combined_bin = np.zeros(shape)
    combined_bin[((x_abs_sobel_bin == 1) & (y_abs_sobel_bin == 1)) | ((mag_sobel_bin == 1) & (dir_sobel_bin == 1)) | (hls_satur_bin == 1)] = 1
    
    # DEBUG
    fig, ax = plt.subplots(4, 2, figsize=(12,16))
    ax[0,0].imshow(rgb_image)
    ax[0,1].imshow(x_abs_sobel_bin, cmap='gray')
    ax[0,1].set_title("x")
    ax[1,0].imshow(y_abs_sobel_bin, cmap='gray')
    ax[1,0].set_title("y")
    ax[1,1].imshow(mag_sobel_bin, cmap='gray')
    ax[1,1].set_title("mag")
    ax[2,0].imshow(dir_sobel_bin, cmap='gray')
    ax[2,0].set_title("dir")
    ax[2,1].imshow(hls_satur_bin, cmap='gray')
    ax[2,1].set_title("sat")
    ax[3,0].imshow(combined_bin, cmap='gray')
    ax[3,0].set_title("res")
    plt.show()
    
    return combined_bin

binary_images = [apply_threshold(rgb) for rgb in undist_rgb_images]
plt.imshow(binary_images[0], cmap='gray')

# +
height, width = binary_images[0].shape

class Warper:
    src_upper_x = 526
    src_upper_y = 470
    src_lower_x = 0
    dst_x = 300
    
    def __init__(self):
        def make_sym_trapez_pts(upper_pt, lower_pt):
            return np.float32([
                    upper_pt, [width-1-upper_pt[0], upper_pt[1]],
                    lower_pt, [width-1-lower_pt[0], lower_pt[1]],
                    ])
        
        self.src_pts = make_sym_trapez_pts(
                [Warper.src_upper_x, Warper.src_upper_y], [Warper.src_lower_x, height-1])
        self.dst_pts = make_sym_trapez_pts(
                [Warper.dst_x, 0], [Warper.dst_x, height-1])
        
        self.M_for = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        
    def forward_warp(self, img):
        return self.__warp(img, self.M_for)
    
    def inverse_warp(self, img):
        return self.__warp(img, self.M_inv)
    
    def __warp(self, img, M):
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, M, (w,h))
        
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        ax[0].imshow(img, cmap='gray')
        ax[0].scatter(self.src_pts[:,0], self.src_pts[:,1], s=30, c='red', marker='o')
        ax[1].imshow(warped, cmap='gray')
        plt.show()
        
        return warped

warper = Warper()
birdview_images = [warper.forward_warp(img) for img in binary_images]
# + {}
def detect_lane_pixel(birdview_binary):
    # params
    n_windows = 9
    win_margin = 70
    min_pix_reposit = 50

    height, width = birdview_binary.shape
    bottom_half = birdview_binary[(height//2):,:]
    histogram = np.sum(bottom_half, axis=0)

    mid_idx = np.int(histogram.shape[0]//2)
    left_x = np.argmax(histogram[:mid_idx])
    right_x = np.argmax(histogram[mid_idx:]) + mid_idx

    win_height = np.int(height // n_windows)
    nonzero_idxs = birdview_binary.nonzero()
    nonzero_y_idxs = np.array(nonzero_idxs[0])
    nonzero_x_idxs = np.array(nonzero_idxs[1])

    left_lane_idxs = []
    right_lane_idxs = []
    
    # DEBUG
    debug_image = np.dstack((birdview_binary,birdview_binary,birdview_binary))

    for win_idx in range(n_windows):
        win_y_hi = height - win_idx * win_height
        win_y_low = win_y_hi - win_height

        left_win_x_low  = left_x - win_margin
        left_win_x_hi   = left_x + win_margin
        right_win_x_low = right_x - win_margin
        right_win_x_hi  = right_x + win_margin

        # DEBUG
        cv2.rectangle(debug_image, (left_win_x_low, win_y_low), (left_win_x_hi, win_y_hi),
                (0,255,0), 2)
        cv2.rectangle(debug_image, (right_win_x_low, win_y_low), (right_win_x_hi, win_y_hi),
                (0,255,0), 2)

        good_left_idxs = ((win_y_low <= nonzero_y_idxs) & (nonzero_y_idxs < win_y_hi) &
                (left_win_x_low <= nonzero_x_idxs) & (nonzero_x_idxs < left_win_x_hi)).nonzero()[0]
        good_right_idxs = ((win_y_low <= nonzero_y_idxs) & (nonzero_y_idxs < win_y_hi) &
                (right_win_x_low <= nonzero_x_idxs) & (nonzero_x_idxs < right_win_x_hi)).nonzero()[0]

        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        if len(good_left_idxs) > min_pix_reposit:
            left_x = np.int(np.mean(nonzero_x_idxs[good_left_idxs]))
        if len(good_right_idxs) > min_pix_reposit:
            right_x = np.int(np.mean(nonzero_x_idxs[good_right_idxs]))

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    left_x_idxs = nonzero_x_idxs[left_lane_idxs]
    left_y_idxs = nonzero_y_idxs[left_lane_idxs]
    right_x_idxs = nonzero_x_idxs[right_lane_idxs]
    right_y_idxs = nonzero_y_idxs[right_lane_idxs]
    
    # DEBUG
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    ax.imshow(debug_image)
    plt.scatter(left_x_idxs, left_y_idxs, s=1, c='red', marker='o')
    plt.scatter(right_x_idxs, right_y_idxs, s=1, c='blue', marker='o')

    return left_x_idxs, left_y_idxs, right_x_idxs, right_y_idxs

idxs_list = [detect_lane_pixel(img) for img in birdview_images]


# +
def fit_polynomial(left_x_idxs, left_y_idxs, right_x_idxs, right_y_idxs):
    left_fit = np.polyfit(left_y_idxs, left_x_idxs, 2)
    right_fit = np.polyfit(right_y_idxs, right_x_idxs, 2)
    return left_fit, right_fit

fits = [fit_polynomial(lx, ly, rx, ry) for lx, ly, rx, ry in idxs_list]

# +
def create_lane_image(height, width, left_fit, right_fit):
    def quadratic(coeffs, y):
        A, B, C = coeffs
        return A * y**2 + B * y + C

    y = np.arange(height)
    left_x = quadratic(left_fit, y).astype('int64')
    right_x = quadratic(right_fit, y).astype('int64')
    
    left_pts = np.array([left_x, y], dtype=np.int32).T
    right_pts = np.array([right_x, y], dtype=np.int32).T
    polygon = np.concatenate([left_pts, right_pts[::-1]])

    lane_img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.polylines(lane_img, [left_pts], False, (255,0,255), 30)
    cv2.polylines(lane_img, [right_pts], False, (255,0,255), 30)
    cv2.fillPoly(lane_img, [polygon], (0,0,255))
    
    # DEBUG
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    ax.imshow(lane_img)
    
    return lane_img

lane_warped = [create_lane_image(height, width, lf, rf) for lf, rf in fits]


# +
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

lane_images = [warper.inverse_warp(img) for img in lane_warped]
done_images = [weighted_img(lane, raw) for raw, lane in zip(undist_rgb_images, lane_images)]

for img in done_images:
    fig, ax = plt.subplots(1,1)
    ax.imshow(img)


# -

# ## And so on and so forth...

def process_image(image):
    return image


# # Reading movie

# +
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

input_video_file = "data/videos/project_video.mp4"
output_video_file = "output_videos/project_video.mp4"

#clip = VideoFileClip(input_videos_path)
#clip = clip.fl_image(process_image)
# #%time clip.write_videofile(output_video_path, audio=False)
# -



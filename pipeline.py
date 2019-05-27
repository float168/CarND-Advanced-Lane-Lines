import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import defaultdict


#------------------------------------------------------------------------------
# Helper function
#------------------------------------------------------------------------------

def load_image(fpath, color_mode='rgb'):
    if color_mode == 'rgb':
        return mpimg.imread(fpath)
    elif color_mode == 'bgr':
        return cv2.imread(fpath)
    elif color_mode == 'gray':
        return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('Invalid color_mode: {}'.format(color_mode))

def save_image(fpath, img, color_mode='rgb'):
    if color_mode == 'rgb':
        return mpimg.imsave(fpath, img)
    elif color_mode == 'bgr':
        return cv2.imwrite(fpath, img)
    elif color_mode == 'gray':
        return cv2.imwrite(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('Invalid color_mode: {}'.format(color_mode))

#------------------------------------------------------------------------------
# Pipelines
#------------------------------------------------------------------------------

# Calibrate camera and undistort images
class Undistorter:
    def __init__(self, chessboard_rgb_images, chessboard_gridsize):
        xgrid, ygrid = chessboard_gridsize

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ygrid*xgrid,3), np.float32)
        objp[:,:2] = np.mgrid[0:xgrid,0:ygrid].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        for rgb in chessboard_rgb_images:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (xgrid,ygrid), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, camera_matrix, distort_coeffs, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
        if not ret:
            raise RuntimeError("Failed 'cv2.calibrateCamera'")

        self.camera_matrix = camera_matrix
        self.distort_coeffs = distort_coeffs

    def apply(self, img):
        return cv2.undistort(img, self.camera_matrix, self.distort_coeffs)


# Apply threshold to images
class LaneFeatureExtractor:
    def __init__(self, ksize_dict, thresh_dict):
        self.ksize_dict  = defaultdict(lambda: 3, ksize_dict)
        self.thresh_dict = defaultdict(lambda: (0,255), thresh_dict)

    def extract(self, rgb_image, debug=False):
        def apply_threshold(img, thresh):
            binary = np.zeros_like(img, dtype='?')
            binary[(thresh[0] <= img) & (img <= thresh[1])] = True
            return binary

        def apply_abs_sobel_thresh(gray, orient='x', ksize=3, thresh=(0,255)):
            sobel = None
            if orient == 'x':
                sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            elif orient == 'y':
                sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            else:
                raise ValueError("Unknown orient type: {}".format(orient))
            abs_sobel = np.absolute(sobel)
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            return apply_threshold(scaled_sobel, thresh)

        def apply_mag_sobel_thresh(gray, ksize=3, thresh=(0,255)):
            x_sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            y_sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag_sobel = np.sqrt(x_sobel**2 + y_sobel**2)
            scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
            return apply_threshold(scaled_sobel, thresh)

        def apply_dir_sobel_thresh(gray, ksize=3, thresh=(0,np.pi/2)):
            x_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize))
            y_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize))
            theta = np.arctan2(y_abs_sobel, x_abs_sobel)
            return apply_threshold(theta, thresh)

        def apply_satur_thresh(rgb, thresh=(0,255)):
            hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
            S = hls[:,:,2]
            return apply_threshold(S, thresh)

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        x_abs_sobel_binary = apply_abs_sobel_thresh(gray_image, orient='x',
                ksize=self.ksize_dict['abs_sobel'],
                thresh=self.thresh_dict['abs_sobel'])
        y_abs_sobel_binary = apply_abs_sobel_thresh(gray_image, orient='y',
                ksize=self.ksize_dict['abs_sobel'],
                thresh=self.thresh_dict['abs_sobel'])
        mag_sobel_binary   = apply_mag_sobel_thresh(gray_image,
                ksize=self.ksize_dict['mag_sobel'],
                thresh=self.thresh_dict['mag_sobel'])
        dir_sobel_binary   = apply_dir_sobel_thresh(gray_image,
                ksize=self.ksize_dict['dir_sobel'],
                thresh=self.thresh_dict['dir_sobel'])
        satur_binary       = apply_satur_thresh(rgb_image,
                thresh=self.thresh_dict['satur'])

        combined_binary = np.zeros_like(gray_image, dtype='?')
        combined_binary[((x_abs_sobel_binary == True) & (y_abs_sobel_binary == True))
                | ((mag_sobel_binary == True) & (dir_sobel_binary == True))
                | (satur_binary == True)] = True

        if debug:
            fig, ax = plt.subplots(3, 2, figsize=(12,12))

            ax[0,0].set_title("input")
            ax[0,0].imshow(rgb_image)

            def make_intersect_image(bin_a, bin_b):
                shape = (bin_a.shape[0], bin_a.shape[1], 3)
                rgb = np.zeros(shape, dtype='uint8')
                i = bin_a == True
                j = bin_b == True
                dark = 64
                rgb[i] = (0, dark, dark)
                rgb[j] = (dark, dark, 0)
                rgb[i & j] = (255, 64, 64)
                return rgb

            def convert_binary_to_image(binary):
                return binary.astype('uint8') * 255

            ax[0,1].set_title("x_abs_sobel(cyan) / y_abs_sobel(yellow)")
            rgb_debug = make_intersect_image(x_abs_sobel_binary, y_abs_sobel_binary)
            ax[0,1].imshow(rgb_debug)

            ax[1,0].set_title("mag_sobel(cyan) / dir_sobel(yellow)")
            rgb_debug = make_intersect_image(mag_sobel_binary, dir_sobel_binary)
            ax[1,0].imshow(rgb_debug)

            ax[1,1].set_title("saturation")
            gray_debug = convert_binary_to_image(satur_binary)
            ax[1,1].imshow(gray_debug, cmap='gray')

            ax[2,0].set_title("output")
            gray_debug = convert_binary_to_image(combined_binary)
            ax[2,0].imshow(gray_debug, cmap='gray')

            plt.show()

        return combined_binary

# Warp perspective
#class Warper:

# Detect lane position from warped view
# This class can be used as batch mode for stream processing
#class LaneDetector:



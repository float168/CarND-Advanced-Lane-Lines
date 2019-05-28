from collections import defaultdict

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
            binary = np.zeros_like(img, dtype='uint8')
            binary[(thresh[0] <= img) & (img <= thresh[1])] = 255
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

        combined_binary = np.zeros_like(gray_image, dtype='uint8')
        combined_binary[((x_abs_sobel_binary == 255) & (y_abs_sobel_binary == 255))
                | ((mag_sobel_binary == 255) & (dir_sobel_binary == 255))
                | (satur_binary == 255)] = 255

        if debug:
            fig, ax = plt.subplots(3, 2, figsize=(12,12))

            ax[0,0].set_title("input")
            ax[0,0].imshow(rgb_image)

            def make_intersect_image(bin_a, bin_b):
                shape = (bin_a.shape[0], bin_a.shape[1], 3)
                rgb = np.zeros(shape, dtype='uint8')
                i = bin_a == 255
                j = bin_b == 255
                dark = 64
                rgb[i] = (0, dark, dark)
                rgb[j] = (dark, dark, 0)
                rgb[i & j] = (255, 64, 64)
                return rgb

            ax[0,1].set_title("x_abs_sobel(cyan) / y_abs_sobel(yellow)")
            rgb_debug = make_intersect_image(x_abs_sobel_binary, y_abs_sobel_binary)
            ax[0,1].imshow(rgb_debug)

            ax[1,0].set_title("mag_sobel(cyan) / dir_sobel(yellow)")
            rgb_debug = make_intersect_image(mag_sobel_binary, dir_sobel_binary)
            ax[1,0].imshow(rgb_debug)

            ax[1,1].set_title("saturation")
            ax[1,1].imshow(satur_binary, cmap='gray')

            ax[2,0].set_title("output")
            ax[2,0].imshow(combined_binary, cmap='gray')

            plt.show()

        return combined_binary


# Warp perspective
class Warper:
    def __init__(self, upper_left_point_pair, lower_left_point_pair, width):
        src_upper_left, dst_upper_left = upper_left_point_pair
        src_lower_left, dst_lower_left = lower_left_point_pair

        def make_symmetric_trapez(upper_left, lower_left, width):
            return np.float32([
                    upper_left, [width-1-upper_left[0], upper_left[1]],
                    lower_left, [width-1-lower_left[0], lower_left[1]],
                    ])
        self.src_points = make_symmetric_trapez(
                src_upper_left, src_lower_left, width)
        self.dst_points = make_symmetric_trapez(
                dst_upper_left, dst_lower_left, width)

        self.M_forward = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inverse = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def forward_warp(self, img, debug=False):
        warped = self.__warp(img, self.M_forward, debug=debug)
        if debug:
            self.__debug_plot(img, warped, self.src_points, self.dst_points)
        return warped

    def inverse_warp(self, img, debug=False):
        warped = self.__warp(img, self.M_inverse, debug=debug)
        if debug:
            self.__debug_plot(img, warped, self.dst_points, self.src_points)
        return warped

    def __warp(self, img, M, debug=False):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, M, (w,h))

    def __debug_plot(self, in_img, out_img, in_pts, out_pts):
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        def subplot(a, img, pts):
            a.imshow(img)
            a.scatter(pts[:,0], pts[:,1], s=30, c='red', marker='o')
        subplot(ax[0], in_img, in_pts)
        subplot(ax[1], out_img, out_pts)
        plt.show()


# Detect lane position from warped view
# This class can be used as batch mode for stream processing
class LaneDetector:
    class LineHistory:
        class RecentList:
            def __init__(self, size):
                self.size = size
                self.list = []

            def update(self, elm):
                if len(self.list) == self.size:
                    self.list.pop(0)
                self.list.append(elm)

            def first(self):
                return self.list[0]

            def last(self):
                return self.list[-1]

        def __init__(self, history_size=5):
            self.recent_list_dict = {
                    'detected': self.RecentList(history_size),
                    'coeffs':   self.RecentList(history_size),
                    'points':   self.RecentList(history_size),
                    }

        def update(self, detected, coeffs, points):
            self.recent_list_dict['detected'].update(detected)
            self.recent_list_dict['coeffs'].update(coeffs)
            self.recent_list_dict['points'].update(points)

    def __init__(self, n_windows, win_margin, reposition_thresh, x_m_per_px, y_m_per_px):
        self.n_windows = n_windows
        self.win_margin = win_margin
        self.reposition_thresh = reposition_thresh

        self.x_m_per_px = x_m_per_px
        self.y_m_per_px = y_m_per_px

        self.left_line_history  = self.LineHistory()
        self.right_line_history = self.LineHistory()

    def draw_lane(self, binary_image, raw_image, warper, batch=False, debug=False):
        height, width = binary_image.shape
        bottom_half   = binary_image[(height//2):,:]
        histogram     = np.sum(bottom_half, axis=0)

        hist_mid_idx    = np.int(histogram.shape[0]//2)
        left_win_x_mid  = np.argmax(histogram[:hist_mid_idx])
        right_win_x_mid = np.argmax(histogram[hist_mid_idx:]) + hist_mid_idx

        nonzero_idxs = binary_image.nonzero()
        nonzero_y_idxs = np.array(nonzero_idxs[0])
        nonzero_x_idxs = np.array(nonzero_idxs[1])

        def collect_line_xy_by_window(win_x_base, debug_image=None):
            nonlocal height, nonzero_y_idxs, nonzero_x_idxs

            win_height = np.int(height // self.n_windows)
            win_x_mid = win_x_base

            line_pt_idxs_list = []
            for win_idx in range(self.n_windows):
                win_y_hi  = height - win_idx * win_height
                win_y_low = win_y_hi - win_height
                win_x_hi   = win_x_mid + self.win_margin
                win_x_low  = win_x_mid - self.win_margin

                if debug_image is not None:
                    debug_image = cv2.rectangle(debug_image,
                            (win_x_low, win_y_low), (win_x_hi, win_y_hi),
                            (0,255,0), 3)

                line_pt_idxs = (
                        (nonzero_y_idxs >= win_y_low) &
                        (nonzero_y_idxs < win_y_hi) &
                        (nonzero_x_idxs >= win_x_low) &
                        (nonzero_x_idxs < win_x_hi)
                        ).nonzero()[0]

                line_pt_idxs_list.append(line_pt_idxs)

                if len(line_pt_idxs) > self.reposition_thresh:
                    win_x_mid = np.int(np.mean(nonzero_x_idxs[line_pt_idxs]))

            line_pt_idxs = np.concatenate(line_pt_idxs_list)
            xs = nonzero_x_idxs[line_pt_idxs]
            ys = nonzero_y_idxs[line_pt_idxs]
            return xs, ys

        def collect_line_xy_by_prev_area(prev_coeffs, debug_image=None):
            nonlocal height, nonzero_y_idxs, nonzero_x_idxs

            def f(coeffs, value):
                return coeffs[0] * value**2 + coeffs[1] * value + coeffs[2]

            prev_line_x_idxs = f(prev_coeffs, nonzero_y_idxs)
            line_pt_idxs = (
                    (nonzero_x_idxs >= prev_line_x_idxs - self.win_margin) &
                    (nonzero_x_idxs < prev_line_x_idxs + self.win_margin)
                    )

            if debug_image is not None:
                y = np.arange(height)
                x = f(prev_coeffs, y)
                def plot_line(img, x, y):
                    points = np.array([x, y], dtype='int32').T
                    cv2.polylines(debug_image, [points], False, (0,255,0), 3)
                plot_line(debug_image, x - self.win_margin, y)
                plot_line(debug_image, x + self.win_margin, y)

            xs = nonzero_x_idxs[line_pt_idxs]
            ys = nonzero_y_idxs[line_pt_idxs]
            return xs, ys

        debug_image = None
        fig = None
        if debug:
            debug_image = np.tile(binary_image.T, (3,1,1)).T
            fig, ax = plt.subplots(2, 2, figsize=(12,8))

        left_line_xs = None
        left_line_ys = None
        right_line_xs = None
        right_line_ys = None
        if batch and self.left_line_history.recent_list_dict['detected'].last():
            left_line_xs, left_line_ys = collect_line_xy_by_prev_area(
                    self.left_line_history.recent_list_dict['coeffs'].first(),
                    debug_image)
            right_line_xs, right_line_ys = collect_line_xy_by_prev_area(
                    self.right_line_history.recent_list_dict['coeffs'].last(),
                    debug_image)
        else:
            left_line_xs, left_line_ys = collect_line_xy_by_window(left_win_x_mid, debug_image)
            right_line_xs, right_line_ys = collect_line_xy_by_window(right_win_x_mid, debug_image)

        if debug:
            ax[0,0].imshow(debug_image)
            ax[0,0].scatter(left_line_xs, left_line_ys, s=1, c='red', marker='o')
            ax[0,0].scatter(right_line_xs, right_line_ys, s=1, c='blue', marker='o')

        def fit_quadratic_func(xs, ys):
            return np.polyfit(ys, xs, 2)

        left_line_coeffs  = fit_quadratic_func(left_line_xs, left_line_ys)
        right_line_coeffs = fit_quadratic_func(right_line_xs, right_line_ys)

        car_x_px = width // 2
        car_y_px = height - 1

        def calc_real_curvature(coeffs):
            nonlocal car_y_px

            # Scale coefficients
            # source: x = coeff[0] * y**2 + coeff[1] * y + coeff[2]
            # scaled: x = x_scale / y_scale**2 * coeff[0] * y**2 + x_scale / y_scale * coeff[1] * y + coeff[2]
            def scale_coeffs(coeff, indep_var_scale, dep_var_scale):
                return (coeff[0] * dep_var_scale / indep_var_scale**2,
                        coeff[1] * dep_var_scale / indep_var_scale,
                        coeff[2])
            scaled_coeffs = scale_coeffs(coeffs, self.y_m_per_px, self.x_m_per_px)

            def R(A, B, y):
                return (1 + (2 * A * y + B)**2)**(1.5) / (2 * np.abs(A))

            curvature_radius = R(scaled_coeffs[0], scaled_coeffs[1], car_y_px * self.y_m_per_px)
            return curvature_radius

        left_line_radius_m  = calc_real_curvature(left_line_coeffs)
        right_line_radius_m = calc_real_curvature(right_line_coeffs)

        def calc_car_offset_from_lane_center():
            nonlocal car_x_px, car_y_px, left_line_coeffs, right_line_coeffs

            def f(coeffs, value):
                return coeffs[0] * value**2 + coeffs[1] * value + coeffs[2]

            left_line_x_px = f(left_line_coeffs, car_y_px)
            right_line_x_px = f(right_line_coeffs, car_y_px)

            offset_px = (left_line_x_px + right_line_x_px) // 2 - car_x_px
            return offset_px * self.x_m_per_px

        car_offset_m = calc_car_offset_from_lane_center()

        def create_lane_image():
            nonlocal height, width, left_line_coeffs, right_line_coeffs

            lane_image = np.zeros((height, width, 3), dtype=np.uint8)

            def f(coeffs, value):
                return coeffs[0] * value**2 + coeffs[1] * value + coeffs[2]

            y = np.arange(height)
            left_x  = f(left_line_coeffs, y).astype('int32')
            right_x = f(right_line_coeffs, y).astype('int32')

            left_pts = np.array([left_x, y], dtype='int32').T
            right_pts = np.array([right_x, y], dtype='int32').T
            cv2.polylines(lane_image, [left_pts], False, (255,0,255), 30)
            cv2.polylines(lane_image, [right_pts], False, (255,0,255), 30)

            polygon = np.concatenate([left_pts, right_pts[::-1]])
            cv2.fillPoly(lane_image, [polygon], (0,0,255))

            return lane_image

        lane_warped_image = create_lane_image()
        lane_image = warper.inverse_warp(lane_warped_image)

        if debug:
            ax[0,1].imshow(lane_image)

        def weight_image(overlay_image, raw_image, α=0.8, β=1., γ=0.):
            """
            raw_image * α + overlay_image * β + γ
            NOTE: initial_img and img must be the same shape!
            """
            return cv2.addWeighted(raw_image, α, overlay_image, β, γ)

        result_image = weight_image(lane_image, raw_image)

        def embed_status(image, curvature_radius_m, car_offset_m):
            rad_text = 'Radius: {}m'.format(int(curvature_radius_m))
            pos_text = "Position: {:+.2f}m".format(car_offset_m)

            embedded_image = np.copy(image)

            def embed_text(image, text, pos):
                cv2.putText(embedded_image, text, pos, cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 6)

            embed_text(embedded_image, rad_text, (10,50))
            embed_text(embedded_image, pos_text, (10,100))

            return embedded_image

        result_image = embed_status(result_image, left_line_radius_m, car_offset_m)

        if debug:
            ax[1,0].imshow(result_image)
            plt.plot()

        return result_image


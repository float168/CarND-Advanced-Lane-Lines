from collections import defaultdict
import copy

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

        def extract_abs_sobel(gray, orient='x', ksize=3):
            sobel = None
            if orient == 'x':
                sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            elif orient == 'y':
                sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            else:
                raise ValueError("Unknown orient type: {}".format(orient))
            abs_sobel = np.absolute(sobel)
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            return scaled_sobel

        def extract_mag_sobel(gray, ksize=3):
            x_sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            y_sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag_sobel = np.sqrt(x_sobel**2 + y_sobel**2)
            scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
            return scaled_sobel

        def extract_dir_sobel(gray, ksize=3):
            x_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize))
            y_abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize))
            theta = np.arctan2(y_abs_sobel, x_abs_sobel)
            return theta

        def extract_satur(rgb):
            hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
            satur = hls[:,:,2]
            return satur

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        x_abs_sobel     = extract_abs_sobel(gray_image, orient='x', ksize=self.ksize_dict['abs_sobel'])
        x_abs_sobel_bin = apply_threshold(x_abs_sobel, thresh=self.thresh_dict['abs_sobel'])
        y_abs_sobel     = extract_abs_sobel(gray_image, orient='y', ksize=self.ksize_dict['abs_sobel'])
        y_abs_sobel_bin = apply_threshold(y_abs_sobel, thresh=self.thresh_dict['abs_sobel'])
        mag_sobel       = extract_mag_sobel(gray_image, ksize=self.ksize_dict['mag_sobel'])
        mag_sobel_bin   = apply_threshold(mag_sobel, thresh=self.thresh_dict['mag_sobel'])
        dir_sobel       = extract_dir_sobel(gray_image, ksize=self.ksize_dict['dir_sobel'])
        dir_sobel_bin   = apply_threshold(dir_sobel, thresh=self.thresh_dict['dir_sobel'])
        satur           = extract_satur(rgb_image)
        satur_bin       = apply_threshold(satur, thresh=self.thresh_dict['satur'])

        combined_binary = np.zeros_like(gray_image, dtype='uint8')
        combined_binary[((x_abs_sobel_bin== 255) & (y_abs_sobel_bin== 255))
                | ((mag_sobel_bin== 255) & (dir_sobel_bin== 255))
                | (satur_bin== 255)] = 255

        if debug:
            fig, ax = plt.subplots(5, 2, figsize=(12,20))
            fig.suptitle('LaneFeatureExtractor', fontsize=16)
            fig.subplots_adjust(top=0.95)

            ax[0,0].set_title("input")
            ax[0,0].imshow(rgb_image)

            def make_intersect_image(bin_a, bin_b):
                shape = (bin_a.shape[0], bin_a.shape[1], 3)
                rgb = np.zeros(shape, dtype='uint8')
                i = bin_a == 255
                j = bin_b == 255
                dark = 96
                rgb[i] = (0, dark, dark)
                rgb[j] = (dark, dark, 0)
                rgb[i & j] = (255, 0, 0)
                return rgb

            ax[0,1].set_title("x_abs_sobel")
            ax[0,1].imshow(x_abs_sobel, cmap='gray')

            ax[1,0].set_title("y_abs_sobel")
            ax[1,0].imshow(y_abs_sobel, cmap='gray')

            ax[1,1].set_title("x_abs_sobel_bin(cyan) / y_abs_sobel_bin(yellow)")
            rgb_debug = make_intersect_image(x_abs_sobel_bin, y_abs_sobel_bin)
            ax[1,1].imshow(rgb_debug)

            ax[2,0].set_title("mag_sobel")
            ax[2,0].imshow(mag_sobel, cmap='gray')

            ax[2,1].set_title("dir_sobel")
            ax[2,1].imshow(dir_sobel, cmap='gray')

            ax[3,0].set_title("mag_sobel_bin(cyan) / dir_sobel_bin(yellow)")
            rgb_debug = make_intersect_image(mag_sobel_bin, dir_sobel_bin)
            ax[3,0].imshow(rgb_debug)

            ax[3,1].set_title("saturation")
            ax[3,1].imshow(satur, cmap='gray')

            ax[4,0].set_title("saturation_bin")
            ax[4,0].imshow(satur_bin, cmap='gray')

            ax[4,1].set_title("output_bin")
            ax[4,1].imshow(combined_binary, cmap='gray')

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
# This class can be used as stream mode for stream processing
class LaneDetector:
    history_size = 5

    class Line:
        def __init__(self, detected, coeffs):
            self.detected = detected
            self.coeffs = coeffs

    class Lane:
        def __init__(self, left=None, right=None):
            self.detected = False
            if left is not None and right is not None:
                if left.detected and right.detected:
                    self.detected = True
            self.left  = left
            self.right = right

    class LaneHistory:
        def __init__(self, size):
            self.size = size
            dummy = LaneDetector.Lane()
            self.list = [dummy]
            self.prev = dummy
            self.averaged = dummy

        def update(self, lane):
            if len(self.list) == self.size:
                self.list.pop(0)
            self.list.append(lane)
            self.prev = self.list[-1]
            self.__update_averaged()

        def __update_averaged(self):
            # Collect detected lane elements
            detected_lanes = list(filter(lambda x: x.detected, self.list))
            if not detected_lanes:
                # Update averaged as dummy
                self.averaged = LaneDetector.Lane()
                return

            size = len(detected_lanes)
            left_coeffs = detected_lanes[0].left.coeffs
            right_coeffs = detected_lanes[0].right.coeffs
            for lane in detected_lanes[1:]:
                left_coeffs += lane.left.coeffs
                right_coeffs += lane.right.coeffs
            left_line  = LaneDetector.Line(True, left_coeffs / size)
            right_line = LaneDetector.Line(True, right_coeffs / size)
            self.averaged = LaneDetector.Lane(left_line, right_line)

    def __init__(self, n_windows, win_margin, reposition_thresh_rate, x_m_per_px, y_m_per_px, x_ignore_area=0):
        self.n_windows = n_windows
        self.win_margin = win_margin
        self.reposition_thresh_rate = reposition_thresh_rate
        self.x_ignore_area = x_ignore_area

        self.x_m_per_px = x_m_per_px
        self.y_m_per_px = y_m_per_px

        self.lane_history  = self.LaneHistory(self.history_size)

    def draw_lane(self, binary_image, raw_image, warper, stream=False, debug=False):
        # Create histogram and nonzero indexes for collecting line points
        height, width = binary_image.shape
        bottom_half   = binary_image[(height//2):,:]
        histogram     = np.sum(bottom_half, axis=0)

        hist_mid_idx    = np.int(histogram.shape[0]//2)
        left_win_x_mid  = self.x_ignore_area + np.argmax(histogram[self.x_ignore_area:hist_mid_idx])
        right_win_x_mid = hist_mid_idx + np.argmax(histogram[hist_mid_idx:width-self.x_ignore_area])

        nonzero_idxs = binary_image.nonzero()
        nonzero_y_idxs = np.array(nonzero_idxs[0])
        nonzero_x_idxs = np.array(nonzero_idxs[1])

        # Collect line points by window
        # Searching starts from win_x_base
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

                rate = len(line_pt_idxs) / win_height / self.win_margin / 2
                if rate > self.reposition_thresh_rate[0] and \
                        rate < self.reposition_thresh_rate[1]:
                    line_pt_idxs_list.append(line_pt_idxs)
                    win_x_mid = np.int(np.mean(nonzero_x_idxs[line_pt_idxs]))

            line_pt_idxs = np.concatenate(line_pt_idxs_list)
            xs = nonzero_x_idxs[line_pt_idxs]
            ys = nonzero_y_idxs[line_pt_idxs]
            return xs, ys

        # Collect line points considering previous lane area
        def collect_line_xy_by_prev_area(prev_coeffs, debug_image=None):
            nonlocal height, nonzero_y_idxs, nonzero_x_idxs

            prev_line_x_idxs = self.__line_func(prev_coeffs, nonzero_y_idxs)
            line_pt_idxs = (
                    (nonzero_x_idxs >= prev_line_x_idxs - self.win_margin) &
                    (nonzero_x_idxs < prev_line_x_idxs + self.win_margin)
                    )

            # DEBUG
            if debug_image is not None:
                fy = np.arange(height)
                fx = self.__line_func(prev_coeffs, y)
                def plot_line(img, x, y):
                    points = np.array([x, y], dtype='int32').T
                    cv2.polylines(debug_image, [points], False, (0,255,0), 3)
                plot_line(debug_image, fx - self.win_margin, fy)
                plot_line(debug_image, fx + self.win_margin, fy)

            xs = nonzero_x_idxs[line_pt_idxs]
            ys = nonzero_y_idxs[line_pt_idxs]
            return xs, ys

        # DEBUG
        debug_image = None
        ax = None
        if debug:
            debug_image = np.dstack((binary_image, binary_image, binary_image))
            fig, ax = plt.subplots(1, 2, figsize=(12,4))

        # Collect line points
        left_line_xs = None
        left_line_ys = None
        right_line_xs = None
        right_line_ys = None
        if stream and self.lane_history.prev.detected:
            left_line_xs, left_line_ys = collect_line_xy_by_prev_area(
                    self.lane_history.prev.left.coeffs, debug_image)
            right_line_xs, right_line_ys = collect_line_xy_by_prev_area(
                    self.lane_history.prev.right.coeffs, debug_image)
        else:
            left_line_xs, left_line_ys = collect_line_xy_by_window(
                    left_win_x_mid, debug_image)
            right_line_xs, right_line_ys = collect_line_xy_by_window(
                    right_win_x_mid, debug_image)

        # Fit coefficients using collected points
        left_line_coeffs  = self.__fit_line_coeffs(left_line_xs, left_line_ys)
        right_line_coeffs = self.__fit_line_coeffs(right_line_xs, right_line_ys)

        if stream:
            # TODO: judge whether line is detected
            left_line  = self.Line(True, left_line_coeffs)
            right_line = self.Line(True, right_line_coeffs)
            lane = self.Lane(left_line, right_line)
            self.lane_history.update(lane)

            if self.lane_history.averaged.detected:
                # Use averaged coeffs
                left_line_coeffs  = self.lane_history.averaged.left.coeffs
                right_line_coeffs = self.lane_history.averaged.right.coeffs

        # DEBUG
        if debug:
            ax[0].imshow(debug_image)
            ax[0].scatter(left_line_xs, left_line_ys,   s=1, c='red', marker='o')
            ax[0].scatter(right_line_xs, right_line_ys, s=1, c='green', marker='o')
            y = np.arange(height)
            ax[0].plot(self.__line_func(left_line_coeffs, y),  y, c='cyan', lw=3)
            ax[0].plot(self.__line_func(right_line_coeffs, y), y, c='cyan', lw=3)

        # Variables for calculating car position and curvature radius
        car_x_px = width // 2
        car_y_px = height - 1

        # Calculate curvature radius in meters
        def calc_real_curvature(coeffs):
            nonlocal car_y_px

            coeffs_m = self.__scale_line_coeffs(coeffs, self.y_m_per_px, self.x_m_per_px)
            y_m = car_y_px * self.y_m_per_px
            curvature_radius_m = self.__line_radius(coeffs_m, y_m)

            return curvature_radius_m

        left_line_radius_m  = calc_real_curvature(left_line_coeffs)
        #right_line_radius_m = calc_real_curvature(right_line_coeffs)

        # Calculate car offset from lane center in meters
        def calc_real_car_offset():
            nonlocal car_x_px, car_y_px, left_line_coeffs, right_line_coeffs

            left_line_x_px  = self.__line_func(left_line_coeffs, car_y_px)
            right_line_x_px = self.__line_func(right_line_coeffs, car_y_px)

            offset_px = (left_line_x_px + right_line_x_px) // 2 - car_x_px
            return offset_px * self.x_m_per_px

        car_offset_m = calc_real_car_offset()

        # Create lane indicator image
        def create_lane_image():
            nonlocal height, width, left_line_coeffs, right_line_coeffs

            lane_image = np.zeros((height, width, 3), dtype=np.uint8)

            y = np.arange(height)
            left_x  = self.__line_func(left_line_coeffs, y).astype('int32')
            right_x = self.__line_func(right_line_coeffs, y).astype('int32')

            left_pts = np.array([left_x, y], dtype='int32').T
            right_pts = np.array([right_x, y], dtype='int32').T
            cv2.polylines(lane_image, [left_pts], False, (255,0,0), 30)
            cv2.polylines(lane_image, [right_pts], False, (0,0,255), 30)

            lane_surface = np.concatenate([left_pts, right_pts[::-1]])
            cv2.fillPoly(lane_image, [lane_surface], (0,255,0))

            return lane_image

        lane_warped_image = create_lane_image()
        lane_image = warper.inverse_warp(lane_warped_image)

        # Mix images into single image
        # raw_image * α + overlay_image * β + γ
        # NOTE: overlay_image and raw_image must be the same shape!
        def mix_image(overlay_image, raw_image, α=0.8, β=1., γ=0.):
            return cv2.addWeighted(raw_image, α, overlay_image, β, γ)

        result_image = mix_image(lane_image, raw_image, α=1.0, β=0.3)

        # Embed radius and car offset info into image
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

        # DEBUG
        if debug:
            ax[1].imshow(result_image)
            plt.plot()

        return result_image

    # Fit line function
    def __fit_line_coeffs(self, xs, ys):
        return np.polyfit(ys, xs, 2)

    # Apply line function
    def __line_func(self, coeffs, y):
            return coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]

    # Calculate curvature radius of function
    def __line_radius(self, coeffs, y):
            A, B, _ = coeffs
            return (1 + (2 * A * y + B)**2)**(1.5) / (2 * np.abs(A))

    # Scale coefficients
    # source: x = coeff[0] * y**2 + coeff[1] * y + coeff[2]
    # scaled: x = x_scale / y_scale**2 * coeff[0] * y**2 + x_scale / y_scale * coeff[1] * y + coeff[2]
    def __scale_line_coeffs(self, coeffs, x_scale, y_scale):
        return (coeffs[0] * x_scale / y_scale**2,
                coeffs[1] * x_scale / y_scale,
                coeffs[2])



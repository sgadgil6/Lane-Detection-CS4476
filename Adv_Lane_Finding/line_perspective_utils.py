import cv2
import numpy as np


def perspectiveChange(inputImg):
    r, c = inputImg.shape
    source = np.array([[c, r - 10],
                       [0, r - 10],
                       [546, 460],
                       [732, 460]])
    dest = np.array([[c, r],
                     [0, r],
                     [400, 0],
                     [c, 0]])

    invPerspTransform = cv2.getPerspectiveTransform(dest, source)
    perspTranform = cv2.getPerspectiveTransform(source, dest)

    warpedImage = cv2.warpPerspective(inputImg, perspTranform, (r, c), flags=cv2.INTER_LINEAR)
    return warpedImage, invPerspTransform, perspTranform

class Line:
    def __init__(self, length_buffer=10):
        self.pixel_last_iteration = None
        self.meter_last_iteration = None

        self.curvature = None
        self.x_coords = None
        self.y_coords = None

    def set_new_line(self, pixel_new, meter_new):
        self.meter_last_iteration = meter_new
        self.pixel_last_iteration = pixel_new

    def return_lane(self, img, lane_type):
        lane_width = 30
        height, width, color = img.shape
        y_plot = np.linspace(0, height - 1, height)
        coefficients = self.pixel_last_iteration
        central = pow(coefficients[0] * y_plot, 2) + coefficients[1] * y_plot + coefficients[2]
        left_lane = central - lane_width // 2
        right_lane = central + lane_width // 2
        left_coords = np.array(list(zip(left_lane, y_plot)))
        right_coords = np.array(np.flipud(list(zip(right_lane, y_plot))))
        final_coords = np.vstack([left_coords, right_coords])
        if lane_type == 'right':
            filled_area = cv2.fillPoly(img, [np.int32(final_coords)], (255, 0, 0))
        else:
            filled_area = cv2.fillPoly(img, [np.int32(final_coords)], (0, 0, 255))
        return filled_area

    @property
    def radius_of_curvature(self):
        coefficients = self.meter_last_iteration
        curvature = ((1 + pow(pow(coefficients[1], 2), 1.5)) / abs(2 * coefficients[0]))
        return curvature

def detect_lanes_from_binary(binary_img, left_line, right_line):
    ht, wd = binary_img.shape
    hist = np.sum(binary_img[200:480, :], axis = 0)

    output_image = np.dstack((binary_img, binary_img, binary_img)) * 255
    midpt = len(hist) // 2
    start_point_left = np.argmax(hist[:midpt])
    start_point_right = np.argmax(hist[midpt:]) + midpt

    non_zero_points = binary_img.nonzero()
    y_coords = np.array(non_zero_points[0])
    x_coords = np.array(non_zero_points[1])

    currLeftCoord = start_point_left
    currRightCoord = start_point_right

    margin_width = 100
    recenterThresh = 50
    total_windows = 9
    singleWindowHt = int(ht / total_windows)

    leftLaneFinalCoords = []
    rightLaneFinalCoords = []
    for windowNum in range(total_windows):
        if windowNum < 3:
            continue
        y_window_down = ht - (windowNum + 1) * singleWindowHt
        y_window_up = ht - windowNum * singleWindowHt
        x_right_window_down = currRightCoord - margin_width
        x_right_window_up = currRightCoord + margin_width
        x_left_window_down = currLeftCoord - margin_width
        x_left_window_up = currRightCoord + margin_width

        cv2.rectangle(output_image, (x_left_window_down, y_window_down), (x_left_window_up, y_window_up), (0, 255, 0), 2)
        cv2.rectangle(output_image, (x_right_window_down, y_window_down), (x_right_window_up, y_window_up), (0, 255, 0), 2)
        leftLaneFinalCoords.append(((y_coords >= y_window_down) & (x_coords < x_left_window_up) & (y_coords < y_window_up)
                                   & (x_coords >= x_left_window_down)).nonzero()[0])
        rightLaneFinalCoords.append(((y_coords >= y_window_down) & (x_coords < x_right_window_up) & (y_coords < y_window_up)
             & (x_coords >= x_right_window_down)).nonzero()[0])


    rightLaneFinalCoords = np.hstack(rightLaneFinalCoords)
    leftLaneFinalCoords = np.hstack(leftLaneFinalCoords)
    rightLaneFinalCoords = rightLaneFinalCoords[::-1]
    leftLaneFinalCoords = leftLaneFinalCoords[::-1]

    left_line.x_coords = x_coords[leftLaneFinalCoords]
    left_line.y_coords = y_coords[leftLaneFinalCoords]
    right_line.x_coords = x_coords[rightLaneFinalCoords]
    right_line.y_coords = y_coords[leftLaneFinalCoords]

    pixel_new_left = np.polyfit(left_line.y_coords, left_line.x_coords, 2)
    pixel_new_right = np.polyfit(right_line.y_coords, right_line.x_coords, 2)

    meter_new_left = np.polyfit(left_line.y_coords * (30.0/720), left_line.x_coords * (3.7/700), 2)
    meter_new_right = np.polyfit(right_line.y_coords * (30.0/720), right_line.x_coords * (3.7/700), 2)

    left_line.set_new_line(pixel_new_left, meter_new_left)
    right_line.set_new_line(pixel_new_right, meter_new_right)

    y_plot = np.linspace(0, ht - 1, ht)
    fitted_left = pixel_new_left[0] * y_plot ** 2 +  pixel_new_left[1] * y_plot + pixel_new_left[2]
    fitted_right = pixel_new_right[0] * y_plot ** 2 +  pixel_new_right[1] * y_plot + pixel_new_right[2]

    output_image[y_coords[leftLaneFinalCoords], x_coords[leftLaneFinalCoords]] = [255, 0, 0]
    output_image[y_coords[rightLaneFinalCoords], x_coords[rightLaneFinalCoords]] = [0, 0, 255]

    return left_line, right_line, output_image

def draw_on_road(image, inverse_persp, left_line, right_line):
    ht, wd = image.shape

    warped = np.zeros_like(image, dtype='uint8')
    unwarped = cv2.warpPerspective(warped, inverse_persp, (wd, ht))
    onRoad = cv2.addWeighted(image, 1., unwarped, 0.3, 0)

    line = np.zeros_like(image)
    line = left_line.return_lane(line, 'left')
    line = right_line.return_lane(line, 'right')
    unwarped_line = cv2.warpPerspective(line, inverse_persp, (wd, ht))
    maskForLine = onRoad.copy()
    indices = np.any([unwarped_line != 0][0], axis = 2)
    maskForLine[indices] = unwarped_line[indices]
    newOnRoad = cv2.addWeighted(src1=maskForLine, alpha=0.8, src2=onRoad, beta=0.5, gamma=0.)
    return newOnRoad








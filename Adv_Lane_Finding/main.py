import cv2
import os
import matplotlib.pyplot as plt
from line_perspective_utils import perspectiveChange, detect_lanes_from_binary, Line, draw_on_road
from moviepy.editor import VideoFileClip
import numpy as np

processed_frames = 0  # counter of frames processed (when processing video)
line_lt = Line()  # line on the left of the lane
line_rt = Line()  # line on the right of the lane

def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames

    height, width = frame.shape[:2]
    binIm = np.zeros(shape=(height, width), dtype=np.uint8)

    yHSVmin = np.array([0, 70, 70])
    yHSVmax = np.array([50, 255, 255])

    hueV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    minThreshold = np.all(hueV > yHSVmin, axis=2)
    maxThreshold = np.all(hueV < yHSVmax, axis=2)

    binIm = np.logical_or(binIm, np.logical_and(minThreshold, maxThreshold))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, eq_white_mask = cv2.threshold(cv2.equalizeHist(gray), thresh=173, maxval=250, type=cv2.THRESH_BINARY)

    binIm = np.logical_or(binIm, eq_white_mask)

    kernel_size = 9
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobX = cv2.Sobel(grayScale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobY = cv2.Sobel(grayScale, cv2.CV_64F, 0, 1, ksize=kernel_size)

    magnitudeS = np.sqrt(sobX ** 2 + sobY ** 2)
    magnitudeS = np.uint8(magnitudeS / np.max(magnitudeS) * 255)

    _, magnitudeS = cv2.threshold(magnitudeS, 50, 1, cv2.THRESH_BINARY)
    binIm = np.logical_or(binIm, magnitudeS.astype(bool))

    kernel = np.ones((5, 5), np.uint8)
    img_binary = cv2.morphologyEx(binIm.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    plt.imshow(img_binary)
    plt.show()
    
    img_birdeye, M, Minv = perspectiveChange(img_binary)
    plt.imshow(img_birdeye)
    plt.show()
    
    line_lt, line_rt, img_fit = detect_lanes_from_binary(img_birdeye, line_lt, line_rt)

    # compute offset in meter from center of the lane
    line_lt_bottom = np.mean(line_lt.x_coords[line_lt.y_coords > 0.95 * line_lt.y_coords.max()])
    line_rt_bottom = np.mean(line_rt.x_coords[line_rt.y_coords > 0.95 * line_rt.y_coords.max()])
    lane_width = line_rt_bottom - line_lt_bottom
    midpoint = frame.shape[1] / 2
    offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
    offset_meter =  (3.7 / 700) * offset_pix

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_on_road(frame, Minv, line_lt, line_rt)
    
    ##
    blendShapeH, blendShapeW = blend_on_road.shape[:2]
   
    rati = 0.2
    thumb_h, thumb_w = int(rati * blendShapeH), int(rati * blendShapeW)
   
    off_x, off_y = 20, 15
   
    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)
   
    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h + off_y, off_x:off_x + thumb_w, :] = thumb_binary
   
    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h + off_y, 2 * off_x + thumb_w:2 * (off_x + thumb_w), :] = thumb_birdeye
   
    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h + off_y, 3 * off_x + 2 * thumb_w:3 * (off_x + thumb_w), :] = thumb_img_fit
   
    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.radius_of_curvature, line_rt.radius_of_curvature])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9,
               (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9,
               (255, 255, 255), 2, cv2.LINE_AA)
    ##
    
    processed_frames += 1

    return blend_on_road


if __name__ == '__main__':
    test_img_dir = 'test_images'
    for test_img in os.listdir(test_img_dir):
        frame = cv2.imread(os.path.join(test_img_dir, test_img))
        frame = cv2.resize(frame, (1280, 720))
        blend = process_pipeline(frame, keep_state=False)
        cv2.imwrite('output_images/{}'.format(test_img), blend)
        plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        plt.show()

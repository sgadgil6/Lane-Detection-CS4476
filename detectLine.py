import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
from os.path import join, basename
from collections import deque

class Line:
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[255, 0, 0], thickness=10):
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), color, thickness)


test_images_dir = join('data', 'test_images')
for name in os.listdir(test_images_dir):
    test_images = [join(test_images_dir, name)]

#for test_img in test_images:
test_img = 'driver_161_90frame\\06030822_0756.MP4\\00000.jpg'
ground_truth_file = 'driver_161_90frame\\06030822_0756.MP4\\00000.lines.txt'
array = []
with open(ground_truth_file) as f:
    for line in f:
        for x in line.split():
            array.append(float(x))
ground_truth = []
for i in range(len(array) - 1):
    ground_truth.append((array[i], array[i+1]))

#out_path = join('out', 'images', basename(test_img))
inputIm = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
img_h, img_w = inputIm.shape[0], inputIm.shape[1]
lane_lines = []

imGray = cv2.cvtColor(inputIm, cv2.COLOR_BGR2GRAY)
imBlur = cv2.GaussianBlur(imGray, (17,17), 0)
imEdg = cv2.Canny(imBlur, threshold1 = 50, threshold2=80)

lineDetect = cv2.HoughLinesP(imEdg, 2, np.pi / 180, 1, np.array([]), 15, 5)
lineDetect = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lineDetect]
candLine = []
for line in lineDetect:
    if 0.5 <= np.abs(line.slope) <= 2:
        candLine.append(line)
##
poSlop = [l for l in candLine if l.slope > 0]
negSlop = [l for l in candLine if l.slope < 0]

negBias = np.median([l.bias for l in negSlop]).astype(int)
negSlo = np.median([l.slope for l in negSlop])
x1, y1 = 0, negBias
x2, y2 = -np.int32(np.round(negBias / negSlo)), 0
lefLan = Line(x1, y1, x2, y2)

posBias = np.median([l.bias for l in poSlop]).astype(int)
posSlo = np.median([l.slope for l in poSlop])
x1, y1 = 0, posBias
x2, y2 = np.int32(np.round((imGray.shape[0] - posBias) / posSlo)), imGray.shape[0]
rigLan = Line(x1, y1, x2, y2)

inferLane = lefLan, rigLan

lane_lines.append(inferLane)
##
avgLinLeft = np.zeros((len(lane_lines), 4))
avgLinRigh = np.zeros((len(lane_lines), 4))

for t in range(0, len(lane_lines)):
    avgLinLeft[t] += lane_lines[t][0].get_coords()
    avgLinRigh[t] += lane_lines[t][1].get_coords()

lane_lines = Line(*np.mean(avgLinLeft, axis=0)), Line(*np.mean(avgLinRigh, axis=0))
##
line_img = np.zeros(shape=(img_h, img_w))

for lane in lane_lines:
    lane.draw(line_img)

vertices = np.array([[(50, img_h),
                  (450, 310),
                  (490, 310),
                  (img_w - 50, img_h)]],
                  dtype = np.int32)

##
mask = np.zeros_like(line_img)
if len(line_img.shape) > 2:
    chanCnt = line_img[2]
    ignoreMask = (255,) * chanCnt
else:
    ignoreMask = 255
cv2.fillPoly(mask, vertices, ignoreMask)

img_masked = cv2.bitwise_and(line_img, mask)
##

img_masked = np.uint8(img_masked)
if len(img_masked.shape) is 2:
    img_masked = np.dstack((img_masked, np.zeros_like(img_masked), np.zeros_like(img_masked)))
outIm = cv2.addWeighted(inputIm, 0.8, img_masked, 1, 0)

#Evaluation metrics
error = 0
for arr in vertices:
    for vertex in arr:
        minimum = float('inf')
        for coord in ground_truth:
            if (np.power(coord[0] - vertex[0], 2) + np.power(coord[1] - vertex[1], 2)) < minimum:
                minimum = np.power(coord[0] - vertex[0], 2) + np.power(coord[1] - vertex[1], 2)
        error = error + minimum
print(np.sqrt(error))



##
#cv2.imwrite(out_path, cv2.cvtColor(outIm, cv2.COLOR_RGB2BGR))
plt.imshow(outIm)
plt.waitforbuttonpress()

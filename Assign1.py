# OpenCV functions
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
# coordinates functions mous clicks
# https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/


# Find chestbord corners openCV: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# BUT DON'T use the function: cv.findChessboardCorners

# The sizes of the chestbord cells in realword are 1.8 cm X 1.8 cm

import numpy as np
import cv2 as cv

import glob

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
 
# images = glob.glob('*.jpeg')
 
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
 
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)
 
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)

#         cv.imshow('img', img)
#         cv.waitKey(0)
 
# cv.destroyAllWindows()


img = cv.imread('image1.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

screen_width = 1200  # adjust to your screen
screen_height = 800

h, w = img.shape[:2]

scale = min(screen_width / w, screen_height / h)

new_size = (int(w * scale), int(h * scale))
resized = cv.resize(img, new_size)

cv.imshow("Image", resized)

cv.waitKey(0)

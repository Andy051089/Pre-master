import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)
kernel1 = np.ones((5, 5), np.uint8)
img = cv2.imread('colorcolor.jpg')

img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img3 = cv2.cvtColor(img, 6)
img1 = cv2.GaussianBlur(img, (9, 9), 3)
img2 = cv2.Canny(img, 200, 250)
img4 = cv2.dilate(img2, kernel, iterations=1)
img5 = cv2.erode(img4, kernel1, iterations=1)
cv2.imshow('img', img)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.imshow('img5', img5)
cv2.waitKey(0)

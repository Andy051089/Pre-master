import cv2
import numpy as np


def empty(v):
    pass


img = cv2.imread('XiWinnie.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 640, 320)

cv2.createTrackbar('min hue', 'trackbar', 0, 179, empty)
cv2.createTrackbar('max hue', 'trackbar', 179, 179, empty)
cv2.createTrackbar('min sat', 'trackbar', 0, 255, empty)
cv2.createTrackbar('max sat', 'trackbar', 255, 255, empty)
cv2.createTrackbar('min val', 'trackbar', 0, 255, empty)
cv2.createTrackbar('max val', 'trackbar', 255, 255, empty)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    h_min = cv2.getTrackbarPos('min hue', 'trackbar')
    h_max = cv2.getTrackbarPos('max hue', 'trackbar')
    s_min = cv2.getTrackbarPos('min sat', 'trackbar')
    s_max = cv2.getTrackbarPos('max sat', 'trackbar')
    v_min = cv2.getTrackbarPos('min val', 'trackbar')
    v_max = cv2.getTrackbarPos('max val', 'trackbar')
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result= cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('img', img)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('result',result)
    cv2.waitKey(1)


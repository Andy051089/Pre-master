import cv2
import numpy as np
import random

img = np.empty((500, 500, 3), np.uint8)
# for row in range(300):
#     for col in range(300):
#         img[row][col]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
img[150:350, 150:350] = [0, 0, 225]
cv2.imwrite('img.jpg', img)
cv2.imshow('img', img)
cv2.waitKey(0)

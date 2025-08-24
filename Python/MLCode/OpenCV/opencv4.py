import cv2
import numpy as np

img = np.zeros((600, 600, 3), np.uint8)
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 2)
cv2.rectangle(img, (0, 0), (100, 200), (0, 255, 0), cv2.FILLED)
cv2.circle(img, (300, 300), 30, (0, 0, 255), cv2.FILLED)
cv2.putText(img, 'text', (500, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 150), 2)
# putText(哪個檔案,寫的文字,位子(左下角),字體,大小,顏色,粗度)
cv2.imshow('img', img)
cv2.waitKey(0)

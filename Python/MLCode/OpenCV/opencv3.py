import cv2

video = cv2.VideoCapture('thumb.mp4')
while True:
    ret, pic = video.read()

    if ret:
        pic = cv2.resize(pic, (0, 0), fx=0.5, fy=0.5)
        pic = cv2.cvtColor(pic, 6)
        cv2.imshow('video', pic)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break

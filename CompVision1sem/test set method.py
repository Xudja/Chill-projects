import cv2
import numpy as np
cap = cv2.VideoCapture(r'C:\Users\Admin\Downloads\Telegram Desktop\20231012_151744.mp4')
kernel = np.ones((11,11),np.uint8)
scale_percent = 30
object_detector = cv2.createBackgroundSubtractorMOG2(history=3000,varThreshold = 32) # create difference of frames
def nothing(x):
    pass

while True:
    ret,frame = cap.read()

    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame = cv2.resize(frame, dsize) # переразмер кадра
    frame = cv2.GaussianBlur(frame,(3,3),0)

    frame = cv2.flip(frame,1)
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    res = cv2.bitwise_and(frame, frame, mask=mask)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            print(contours,'ekfbvjsbd vjbsj,v j,sdv ')


    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('closing', closing)


    k = cv2.waitKey(16) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    edge = cv2.Canny(gray,70,500)
    print(edge)
    contours,h = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key = cv2.contourArea,reverse = True)
    print(contours)
    if len(contours)!=0:
        cv2.drawContours(frame,[contours[0]],-1,(0,255,0),5)
    cv2.imshow('frame', frame)

    cv2.imshow('EDGE',edge)
    k = cv2.waitKey(1) and 0xFF
    if k==27:
        break

cv2.destroyAllWindows()

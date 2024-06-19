import cv2
import numpy as np
import math
import time
import pandas as pd
cap = cv2.VideoCapture(0)
# cv2.namedWindow('frame')
kernel = np.ones((15,15),np.uint8)
scale_percent = 50

ret,frame = cap.read()
def nothing(x):
    pass
def ugol(x1,y1,x2,y2):
    fi = (y2-y1)/(((x2-x1)**2+(y2-y1)**2)**0.5)
    return math.acos(fi)*180/math.pi


cv2.namedWindow('tr')


cv2.createTrackbar('ЦВЕТ-НИЖ','tr',0,180,nothing) # создание трэкбаров

cv2.createTrackbar('НАСЫЩЕННОСТЬ-НИЖ','tr',0,255,nothing)
cv2.createTrackbar('ЗНАЧЕНИЕ-НИЖ','tr',0,255,nothing)
cv2.createTrackbar('ЦВЕТ-ВЕРХ','tr',0,180,nothing)
cv2.createTrackbar('НАСЫЩЕННОСТЬ-ВЕРХ','tr',0,255,nothing)
cv2.createTrackbar('ЗНАЧЕНИЕ-ВЕРХ','tr',0,255,nothing)

chet = 0
start_time = time.time()
df = pd.DataFrame([[0,0,0]])
degrees = pd.DataFrame(df,columns = ['Degree 1-2','Degree 2-3','Time'])

while True:
    _, frame = cap.read()
    if not _:
        break
    current_time = time.time()
    elapsed_time = current_time - start_time
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame = cv2.resize(frame, dsize)

    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #перевод в hsv формат

    h = cv2.getTrackbarPos('ЦВЕТ-НИЖ','tr')#снятие значение с трекбаров
    s = cv2.getTrackbarPos('НАСЫЩЕННОСТЬ-НИЖ', 'tr')
    v = cv2.getTrackbarPos('ЗНАЧЕНИЕ-НИЖ', 'tr')
    hl = cv2.getTrackbarPos('ЦВЕТ-ВЕРХ', 'tr')
    sl = cv2.getTrackbarPos('НАСЫЩЕННОСТЬ-ВЕРХ', 'tr')
    vl = cv2.getTrackbarPos('ЗНАЧЕНИЕ-ВЕРХ', 'tr')

    lower = np.array([h,s,v]) #установка границ в нумпай массив
    upper = np.array([hl, sl, vl])
    mask = cv2.inRange(hsv,lower,upper) #маска изображения по границам установленных значений

    res = cv2.bitwise_and(frame,frame,mask=mask) # логическое И
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    closing1 = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel) # морфологическое закрытие
    closing = cv2.morphologyEx(closing1, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key = cv2.contourArea,reverse=True)
    x_sum1 = 0
    y_sum1 = 0
    x_sum2 = 0
    y_sum2 = 0
    x_sum3 = 0
    y_sum3 = 0

    if len(contours) > 2:
        sel_countour1 = contours[0]  # самый большой контур
        sel_countour2 = contours[1]
        sel_countour3 = contours[2]
        arclen1 = cv2.arcLength(sel_countour1, True)
        arclen2 = cv2.arcLength(sel_countour2, True)
        arclen3 = cv2.arcLength(sel_countour3, True)

        eps = 0.1  # коэффицент для длины
        epsilon1 = arclen1 * eps
        epsilon2 = arclen2 * eps
        epsilon3 = arclen3 * eps

        approx1 = cv2.approxPolyDP(sel_countour1, epsilon1, True)  # аппроксимация точек по заданной длине
        approx2 = cv2.approxPolyDP(sel_countour2, epsilon2, True)  # аппроксимация точек по заданной длине
        approx3 = cv2.approxPolyDP(sel_countour3, epsilon2, True) # аппроксимация точек по заданной длине

        a = frame.shape
        height = a[0]
        width = a[1]
        sum_of_pixels = height * width
        zero = np.uint8(np.zeros((frame.shape[0], frame.shape[1], 3)))
        coordinate_count1 = 0
        for i in approx1:
            coordinate_count1 += 1
            x_sum1 += i[0][0]
            y_sum1 += i[0][1]
        x_geo_center = int(x_sum1 / coordinate_count1)
        y_geo_center = int(y_sum1 / coordinate_count1)
        geo_centaer1 = (x_geo_center, y_geo_center)
        cv2.circle(frame, geo_centaer1, 60, (0, 255, 0), 2)
        coordinate_count2 = 0

        for i in approx2:
            coordinate_count2 += 1
            x_sum2 += i[0][0]
            y_sum2 += i[0][1]
        x_geo_center = int(x_sum2 / coordinate_count2)
        y_geo_center = int(y_sum2 / coordinate_count2)
        geo_centaer2 = (x_geo_center, y_geo_center)
        cv2.circle(frame, geo_centaer2, 60, (0, 255, 0), 2)
        alpha12 = ugol(x1=geo_centaer1[0], y1=geo_centaer1[1], x2=geo_centaer2[0], y2=geo_centaer2[1])

        cv2.putText(frame, str(int(alpha12)), geo_centaer2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        coordinate_count3 = 0
        for i in approx3:
            coordinate_count3+=1
            x_sum3 += i[0][0]
            y_sum3 += i[0][1]
        x_geo_center = int(x_sum3/coordinate_count3)
        y_geo_center = int(y_sum3/coordinate_count3)
        geo_centaer3 = (x_geo_center,y_geo_center)
        cv2.circle(frame, geo_centaer3, 60, (0, 255, 0), 2)
        coordinate_count2 = 0
        alpha23 = ugol(x1 = geo_centaer2[0],y1 = geo_centaer2[1],x2 = geo_centaer3[0],y2 = geo_centaer3[1])
        cv2.putText(frame, str(int(alpha23)), geo_centaer3, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        degrees.loc[len(degrees.index)] = [alpha12,alpha23,elapsed_time]


        print((alpha12,alpha23,elapsed_time))


    cv2.imshow('zero', frame)
    cv2.imshow('mask', res)
    chet += 1

    cv2.imshow('mask',res)
    cv2.imshow('close', closing)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(200) and 0xFF

print(degrees)
degrees.to_excel('degrees_6.xlsx', index = False)

cv2.destroyAllWindows()


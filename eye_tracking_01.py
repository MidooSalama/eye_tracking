import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_righteye_2splits.xml')

#number signifies camera
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
        roi_color2 = img[ey:ey+eh, ex:ex+ew]
        roi_bw2 = bw[ey:ey+eh, ex:ex+ew]
        # circles = cv2.HoughCircles(roi_gray2,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        # circles = cv2.HoughCircles(roi_gray2, cv2.HOUGH_GRADIENT, 1, 200, param1=20, param2=10, minRadius=0,maxRadius=0)
        circles = cv2.HoughCircles(roi_bw2, cv2.HOUGH_GRADIENT, 1, 100, param1=10, param2=10, minRadius=4,maxRadius=15)
        # print(circles)
        # circles = np.uint16(np.around(circles))
        try:
            circles = circles[0]
            print(circles)
            for i in circles:
                print(i)
                # draw the outer circle
                cv2.circle(roi_color2,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),2)
                print("drawing circle")
                # draw the center of the circle
                cv2.circle(roi_color2,(int(i[0]),int(i[1])),2,(255,255,255),3)
                cv2.imshow('eye', roi_color2)
        except Exception as e:
            print(f"error : {e}")
    cv2.imshow('img',img)
    # cv2.imshow('gray', gray)
    cv2.imshow('bw', bw)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
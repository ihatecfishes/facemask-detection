import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, temp = cap.read()
    frame = cv2.flip(temp, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] #ycord_start, ycord_end(height) ... same for x (width)
        roi_color = frame[y:y+h, x:x+w]
        img_item = "clone.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0, 0, 255) #BGR
        stroke = 2
        height = y + h #height
        width = x + w  #width
        rec_w = (x+width)//2
        rec_h = (y+height)//2
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)
        cv2.line(frame, (rec_w,0), (rec_w,rec_h*2), color, stroke) #rec_linex
        cv2.line(frame, (0,rec_h), (rec_w*2,rec_h), color, stroke) #rec_liney
        
        if (rec_w <= 335) and (rec_w >= 305):
            cv2.rectangle(frame, (x, y), (width, height), (0,255,0), 5)
            cv2.line(frame, (rec_w,0), (rec_w,rec_h*2), (0,255,0), 5)

        if (rec_h <= 255) and (rec_h >= 225):
            cv2.rectangle(frame, (x, y), (width, height), (0,255,0), 5)
            cv2.line(frame, (0,rec_h), (rec_w*2,rec_h), (0,255,0), 5)


        #if (rec_w < 305):
        #  
        #if (rec_w > 335):
        #


        #if (rec_w <= 335) and (rec_w >= 305) and (rec_h <= 255) and (rec_h >= 225): #for both x and y
        #    cv2.rectangle(frame, (x, y), (width, height), (0,255,0), 5)
        #    cv2.line(frame, (rec_w,0), (rec_w,rec_h*2), (0,255,0), 5)
        #    cv2.line(frame, (0,rec_h), (rec_w*2,rec_h), (0,255,0), 5)
            

    cv2.line(frame, (320,480), (320,0), (255, 0, 0), 2) #framecenterlinex
    cv2.line(frame, (0,240), (640,240), (255, 0, 0), 2) #framecenterliney

    cv2.imshow('face_detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

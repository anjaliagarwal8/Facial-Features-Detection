import cv2
import sys

#importing the face classifier
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#importing the eye classifier
eyeCascade=cv2.CascadeClassifier('haarcascade_eye.xml')
#importing the smile classifier
smileCascade=cv2.CascadeClassifier('haarcascade_smile.xml')
#starting the webcam
cap = cv2.VideoCapture(0)
while(True):
    
    #capturing the frames in the real-time video
    ret, img = cap.read()
        #changing the images to grayscale for classifying
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #detecting the face in the real-time video
    faces=faceCascade.detectMultiScale(img_gray,1.5,2)
        #for each detection of face detecting the eyes and smile
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceROI_gray=img_gray[y:y+h,x:x+w]
        faceROI=img[y:y+h,x:x+w]
        #eye detection
        eye=eyeCascade.detectMultiScale(faceROI_gray,1.5,4)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(faceROI,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                
            #smile detection
    #        smile=smileCascade.detectMultiScale(faceROI_gray,1.5,10)
    #        for(sx,sy,sw,sh) in smile:
    #            cv2.rectangle(faceROI,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)
        cv2.imshow('img', img)
    if cv2.waitKey(25) & 0xFF == 27:
        cv2.destroyAllWindows()
#for closing the window
cap.release()



                
                
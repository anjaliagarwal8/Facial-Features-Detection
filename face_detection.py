import numpy as np
import cv2
import sys

if __name__ == '__main__':
    
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                                       
    faceNeighbours=10
    neighbourstep=1
    
    face=cv2.imread('input_image')
    faceg=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    
    #perform multi scale detection of faces
    for neigh in range(1,faceNeighbours,neighbourstep):
        faces=faceCascade.detectMultiScale(faceg,1.2,neigh)
        frameClone=np.copy(face)
        
        #display the image
        for (x,y,w,h) in faces:
            cv2.rectangle(frameClone,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('face detection',frameClone)
           
        if cv2.waitKey(500) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit()

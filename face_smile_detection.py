import cv2
import sys
import numpy as np

if __name__ == '__main__':
    faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smileCascade=cv2.CascadeClassifier('haarcascade_eye.xml')
    neighbours=100
    neighbourstep=2
    
    img=cv2.imread('input_image')
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #to detect the face
    faces=faceCascade.detectMultiScale(img_gray,1.4,5) 
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,0),2)
        faceROI_gray=img_gray[y:y+h,x:x+w]
        faceROI=img[y:y+h,x:x+w]
        
        #to detect the smile from the face detected area and display the image
        for neigh in range(1,neighbours,neighbourstep):
            smile=smileCascade.detectMultiScale(faceROI_gray,1.5,neigh)
            
            #to copy the contents of one image to another(Cloning)
            imgClone=np.copy(img)
            faceROI_clone=imgClone[y:y+h,x:x+w]
            
            for (sx,sy,sw,sh) in smile:
                cv2.rectangle(faceROI_clone,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            cv2.imshow("face and smile",imgClone)
            if cv2.waitKey(25) & 0xFF == 27:
                cv2.destroyAllWindows()
                sys.exit()
            
            
        
        
        
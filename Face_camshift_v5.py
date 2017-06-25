import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


face_cascade = cv2.CascadeClassifier('E:\Program Files (x86)\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml') #loads pre trained classifiers from OpenCv Libs
eye_cascade = cv2.CascadeClassifier('E:\Program Files (x86)\opencv\sources\data\haarcascades\haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

capWebcam = cv2.VideoCapture(0)         # declare a VideoCapture object and associate to webcam, 0 is default for in built-2/2/16 Ajinkya 


def main():
    flag=0
    
    if capWebcam.isOpened() == False:               # check if VideoCapture object was associated to webcam successfully
        print "error: capWebcam not accessed successfully\n\n"      # if not, print error message to std out
        os.system("pause")                                          # pause until user presses a key so user can see error message
        return                                                      # and exit function (which exits program)

    while cv2.waitKey(1) != 27 and capWebcam.isOpened():            # until the Esc key is pressed or webcam connection is lost
        ret,img = capWebcam.read()            # read next frame
        
        if not ret or img is None:     # if frame was not read successfully
            print "error: frame not read from webcam\n"             # print error message 
            os.system("pause")                                      # pause until user presses a key so user can see error message
            break                                                   
        
        height, width, channels = img.shape
        
        #Denoising the image
##        img_Array = [capWebcam.read()[1] for i in xrange(5)]
##        gray_Array = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img_Array]
##        gray_Array=[np.uint8(np.clip(i,0,255)) for i in gray_Array]
##        img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

        
##      Background subtractor
##        fgbg = cv2.createBackgroundSubtractorMOG2()
##        fgmask = fgbg.apply(img)

        
        

        while flag==0 :                                                   ##this is to make sure that the initial robust detector works only once.
            face_returned=initial_robust_detect()
                     
            print face_returned
            if face_returned is not None :
                flag=flag+1
##                ret, img = capWebcam.read()
##                img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
                for (x,y,w,h) in face_returned:

                    track_window=(x,y,w,h)
                    #set up ROI
                    roi = img[y:y+h , x:x+w]
                    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    cv2.imshow("hsv", hsv_roi)
                    mask = cv2.inRange(hsv_roi, np.array((0., 0.,50.)), np.array((50.,230.,170.)))
##
##                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
##                    mask=cv2.erode(mask, kernel, iterations = 2)
##                    mask=cv2.dilate(mask, kernel, iterations = 2)
                    
                    cv2.imshow("mask", mask)
                    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[255],[0,255])
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    roi_hist.reshape(-1) #very important
                    

                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    img =cv2.circle(img,(x+w/2,y+h/2),1,(255,0,0),2) #centre of face

                    face_center="("+str(x+w/2)+"," +str(y+h/2)+")"
            
                    img=cv2.putText(img,face_center,(x+w/2,y+h/2),font, 0.4 , (255,0,255),1 ,cv2.LINE_AA)


##                    bin_count = roi_hist.shape[0]
##                    bin_w = 24
##                    roi_hist = np.zeros((256, bin_count*bin_w, 3), np.uint8)
##                    
##                    for i in xrange(bin_count):
##                        h = int(roi_hist[i])
##                        cv2.rectangle(roi_hist, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
##                    roi_hist = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
##                    cv2.imshow('hist', roi_hist)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
##Now we have the rectangle locating the face position, so we can use it in camshift. Camshift came across better at following objects
            ## as you no longer have to detect a face- just follow the histogram it generated. Resistant to almost any angular shifts. 24/2 -Ajinkya           

##CAMSHIFT
                # create a mask
##                mask = np.zeros(img.shape[:2], np.uint8)
##                mask[y:y+h , x:x+w] = 255 #remember it is y first then x ! That Gotcha moment. 
##                masked_img = cv2.bitwise_and(img,img,mask = mask)
####                cv2.imshow('mask', mask)

##                #Histogram
##                hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
##                plt.plot(hist_mask)
##                plt.show()

                #track window

        if  ret == True :
            
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
            cv2.imshow("back",dst)
    
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img = cv2.polylines(img,[pts],True, 255,2)

        img =cv2.circle(img,(width/2,height/2) ,1,(255,0,0),2) #centre of camera frame
##        
##        frame_center="("+str(width/2)+"," +str(height/2)+")"
##        cv2.putText(img,frame_center,(width/2,height/2),font, 0.4 , (255,0,0),1 ,cv2.LINE_AA)
        cv2.imshow('img', img)

        
    capWebcam.release()    
    cv2.destroyAllWindows()

def initial_robust_detect() :

        ret, img = capWebcam.read()
        height, width, channels = img.shape
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        
        M = cv2.getRotationMatrix2D((width/2,height/2),22,1)
        img2 = cv2.warpAffine(img,M,(width,height))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

        M3 = cv2.getRotationMatrix2D((width/2,height/2),-22,1)
        img3 = cv2.warpAffine(img,M3,(width,height))
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        faces3 = face_cascade.detectMultiScale(gray3, 1.3, 5)
        
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            img =cv2.circle(img,(x+w/2,y+h/2),1,(255,0,0),2) #centre of face

            face_center="("+str(x+w/2)+"," +str(y+h/2)+")"
            
            img=cv2.line(img,(width/2,height/2),(x+w/2,y+h/2) ,(1,1,1), thickness=1, lineType=8, shift=0)
            img=cv2.putText(img,face_center,(x+w/2,y+h/2),font, 0.4 , (255,0,0),1 ,cv2.LINE_AA)

            roi_eye=img[y:y+h/3, x:x+w]
            roi_eye_gray = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
            
            eyes = eye_cascade.detectMultiScale(roi_eye_gray)

            if eyes is not None :
                cv2.imshow('face', img)
                return faces
            
        
##        for (x,y,w,h) in faces2:
##           img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
##           img =cv2.circle(img,(x+w/2,y+h/2),1,(0,255,255),2) #centre of face
##                         
##           face_center="("+str(x+w/2)+"," +str(y+h/2)+")"
##            
##           img=cv2.line(img,(width/2,height/2),(x+w/2,y+h/2) ,(1,1,1), thickness=1, lineType=8, shift=0)
##           img=cv2.putText(img,face_center,(x+w/2,y+h/2),font, 0.4 , (255,0,0),1 ,cv2.LINE_AA)
##
##           return faces2
##
##        for (x,y,w,h) in faces3:
##           img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
##           img =cv2.circle(img,(x+w/2,y+h/2),1,(255,0,255),2) #centre of face
##                         
##           face_center="("+str(x+w/2)+"," +str(y+h/2)+")"
##            
##           img=cv2.line(img,(width/2,height/2),(x+w/2,y+h/2) ,(1,1,1), thickness=1, lineType=8, shift=0)
##           img=cv2.putText(img,face_center,(x+w/2,y+h/2),font, 0.4 , (255,0,0),1 ,cv2.LINE_AA)
##
##           return faces3

        
if __name__ == "__main__":
    main()

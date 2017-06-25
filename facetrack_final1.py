import os
import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('E:\Program Files (x86)\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml') #loads pre trained classifiers from OpenCV Libs
eye_cascade = cv2.CascadeClassifier('E:\Program Files (x86)\opencv\sources\data\haarcascades\haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

capWebcam = cv2.VideoCapture(0)

reset_thresh=12 #no of minimum points before algorithm is reset

def main():
    flag=0

    
    if capWebcam.isOpened() == False:               # check if VideoCapture object was associated to webcam successfully
        print "error: capWebcam not accessed successfully\n\n"      # if not, print error message to std out
        os.system("pause")                                          # pause until user presses a key so user can see error message
        return                                                      # and exit function (which exits program)

    while cv2.waitKey(1) != 27 and capWebcam.isOpened():            # until the Esc key is pressed or webcam connection is lost
        blnFrameReadSuccessfully, img = capWebcam.read()            # read next frame

        
        if not blnFrameReadSuccessfully or img is None:     # if frame was not read successfully
            print "error: frame not read from webcam\n"             # print error message 
            os.system("pause")                                      # pause until user presses a key so user can see error message
            break                                                   
        


##        img =cv2.circle(img,(width/2,height/2) ,1,(255,0,0),2) #centre of camera frame
        
##        frame_center="("+str(width/2)+"," +str(height/2)+")"
##        cv2.putText(img,frame_center,(width/2,height/2),font, 0.4 , (255,0,0),1 ,cv2.LINE_AA)
        while flag==0 :
            face_returned=initial_robust_detect()
            if face_returned is not None :
                flag=flag+1
                ret, img = capWebcam.read()

                #brightness adjustment/shadow removal
                img=equalize_brightness1 (img,face_returned) 
                
                old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                for (x,y,w,h) in face_returned:
##                    face = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
##                    cv2.imshow("face",face)
        ##            img =cv2.circle(img,(x+w/2,y+h/2),1,(255,0,0),2) #centre of face

                    roi = img[y:y+h , x:x+w]
                    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

                    ##Shi_tomasi+ Lucas-Kanade flow
                    # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 35,qualityLevel = 0.03 ,minDistance = 12, blockSize = 11 )

        mask_use = np.zeros(old_gray.shape,np.uint8)
        for (x,y,w,h) in face_returned:
            mask_use[y-5:y+h-10,x+10:x+w-10] = old_gray[y-5:y+h-10,x+10:x+w-10] ##manual adjustment of ROI, as the x,y,w,h have a surrounding region where features could be detected
        cv2.imshow("mask", mask_use)
        
        lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
       
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_use, **feature_params)
##        p0=cv2.goodFeaturesToTrack(gray,25,0.01,10)
        p0 = np.float32(p0)
##        print p0
        
        for i in p0:
            x1,y1 = i.ravel()
            point_image=cv2.circle(img,(x1,y1),3,255,-1)


        # Create a mask image for drawing purposes
        mask = np.zeros_like(img)


        while(flag==1) :                              #activates after face_returned is not None
            cv2.imshow('points', point_image)
            
            # calculate optical flow
            ret,frame = capWebcam.read()
            
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None , **lk_params)


              # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            hull = cv2.convexHull(good_new)
## not required            ellipse = cv2.fitEllipse(good_new)
            
               # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
##                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2) # comment to disable tracks -3/3
                frame = cv2.circle(frame,(a,b),5,color[1].tolist(),-1)
##
##            #Deal with outliers

            for i,(new,old) in enumerate(zip(good_new,good_old)):
                p,q = new.ravel()
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    r,s = new.ravel()
                    dist=calc_distance(p,q,r,s)
####                    print dist
                    if dist > 250:
                        flag=0
                        face_returned=None #rests the algorithm if a point goes too far away from others (probably due to noise/tracking wrong feature)

            if good_new.size <reset_thresh :
                flag=0
                face_returned=None #resets the algorithm if most tracking points are lost; flag reset to break out of this loop;

                
            img = cv2.add(frame, mask)
            
            hull = cv2.convexHull(np.int32(good_new)) ##typcasting required to avoid error with polylines-6/3 Aj
            (cx,cy),radius = cv2.minEnclosingCircle(good_new)

            img=cv2.polylines(img,[np.int32(good_new)],1, (255,0,0))
            img=cv2.polylines(img,[hull],1, (255,0,0))
            img=cv2.circle(img,(int(cx),int(cy)),100,(0,255,0),2)
        

            if good_new.size >reset_thresh-1 :
                
                face_center="Face at:"+"("+str(int(cx))+"," +str(int(cy))+")"
                img=cv2.putText(img,face_center,(500,470),font, 0.4 , (255,118,72),1 ,cv2.LINE_AA)
                img=cv2.circle(img,(np.int32(cx),np.int32(cy)),1,(255,0,0),2)
            else :
                face_center="Face lost, Searching ..."
                img=cv2.putText(img,face_center,(500,470),font, 0.4 , (23,23,23),1 ,cv2.LINE_AA)
            
##                        cv2.imshow('old',old_gray)
            cv2.imshow('img',img)
##                        cv2.imshow('roi_flow',roi)
            k = cv2.waitKey(15) & 0xff
            if k == 27:
                break
        
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        
##        cv2.imshow('corners', corners)
##        cv2.imshow('img3',img3)
##        cv2.imshow('img2',img2)
##        cv2.imshow('gray',gray)
    cv2.imshow('Master',img)
    capWebcam.release()    
    cv2.destroyAllWindows()                 # triggered on long press ESC


def initial_robust_detect() :

        ret, img = capWebcam.read()
        height, width, channels = img.shape
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
        M = cv2.getRotationMatrix2D((width/2,height/2),10,1)
        img2 = cv2.warpAffine(img,M,(width,height))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

        M3 = cv2.getRotationMatrix2D((width/2,height/2),-10,1)
        img3 = cv2.warpAffine(img,M3,(width,height))
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        faces3 = face_cascade.detectMultiScale(gray3, 1.3, 5)

        faces=select_largest_face(faces)
        faces2=select_largest_face(faces2)
        faces3=select_largest_face(faces3)
##        print len(faces)
##        if len(faces) >0 :
##            print faces[0]
##            for (x,y,w,h) in faces :
##                print w,h
        if ((faces is not None) or (faces2 is not None) or (faces3 is not None)) :
            if faces is not None :
##               faces=select_largest_face(faces)
               for (x,y,w,h) in faces :
                    roi_eye=img[y:y+h/3, x:x+w]
                    roi_eye_gray = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
                    eyes = eye_cascade.detectMultiScale(roi_eye_gray)
                    
                    if eyes is not None :
                        return faces
                    elif faces2 is not None:
##                        faces2=select_largest_face(faces2)
                        for (x,y,w,h) in faces2:
                            roi_eye=img2[y:y+h/3, x:x+w]
                            roi_eye_gray = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
                            
                            eyes = eye_cascade.detectMultiScale(roi_eye_gray)
                            if eyes is not None :
                               return faces2
                            elif faces3 is not None :
##                                faces3=select_largest_face(faces3)
                                for (x,y,w,h) in faces3:
                                    roi_eye=img3[y:y+h/3, x:x+w]
                                    roi_eye_gray = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
                            
                                    eyes = eye_cascade.detectMultiScale(roi_eye_gray)
                                    if eyes is not None:
                                      return faces3
                                    else:
                                        initial_robust_detect()
                                        face_center="Face lost, Searching ..."
                                        img=cv2.putText(img,face_center,(500,470),font, 0.4 , (72,118,250),1 ,cv2.LINE_AA)
                    elif faces3 is not None :
                        for (x,y,w,h) in faces3:
                            roi_eye=img3[y:y+h/3, x:x+w]
                            roi_eye_gray = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
                            eyes = eye_cascade.detectMultiScale(roi_eye_gray)
                            if eyes is not None:
                                 return faces3
                            else:
                                 initial_robust_detect()
                                 face_center="Face lost, Searching ..."
                                 img=cv2.putText(img,face_center,(500,470),font, 0.4 , (72,118,250),1 ,cv2.LINE_AA)
        else:
             initial_robust_detect()
             face_center="Face lost, Searching ..."
             img=cv2.putText(img,face_center,(500,470),font, 0.4 , (72,118,250),1 ,cv2.LINE_AA)

def calc_distance(a,b,c,d) :
    dist=((c-a)**2+(d-b)**2)**0.5
    return int(dist)

def select_largest_face(faces) : #works - dont change
    if len(faces)==1 :
        print "Only one face detected"
        return faces

    elif len(faces)>1 :
        largest=faces[0]
        print ( str(len(faces))+" faces detected")
        for i in range (1,len(faces)-1) :
              check_face=faces[i]
              for (x,y,w,h) in check_face :
                  hcheck=h
              for  (x,y,w,h) in largest :
                  largesth=h
              if hcheck > largesth :
                  largest=faces[i]
        print "Largest->" + str(largest)
        return [largest]  ##send back as a matrix and not as an element 

def equalize_brightness1 (image,face_region) :
    for (x,y,w,h) in face_region :
        pt1x=int(x+w/2-5)
        pt1y=int(y)
        pt2x=int(x+w/2+5)
        pt2y=int(y)
        print x,y,w,h

            
        #convert to HSV and split into channels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
##        hsv=hsv[y:y+h , x:+w]
        hue,sat,val =cv2.split(hsv)
##        cv2.imshow("val",val)
##        print val [pt1x][pt1y], val[pt2x][pt2y]
        #compare brightness and adjust
        print "Removing shadows...."
##        print (y+h-1)

##Removing the shadows        
        ht=h
        yline=y

        if val[yline][pt2x]>val[yline][pt1x] : #helps prevent underflow
            adj=(val[yline][pt2x] - val[yline][pt1x])
        else :
            adj=(val[yline][pt1x] - val[yline][pt2x])
        if adj >12:
            adj=12 #capped adjustment
        
        while yline < yline+ht-1:
            while pt2x < x+w-1:   ## inverted y and x in the val matrix
                if val[yline][pt1x] < val[yline][pt2x] :
                    if val[yline][pt1x]+adj <255:            ##prevents overflow
                        val[yline][pt1x] = val[yline][pt1x]+adj
                else :
                    if val[yline][pt2x]+adj <255:
                        val[yline][pt2x] = val[yline][pt2x]+adj
                pt1x=pt1x-1
                pt2x=pt2x+1
                
            pt1x=int(x+w/2-1)
            pt2x=int(x+w/2+1)
            yline=yline+1
            if yline >y+h-1 :
                break
##        for i  in range (y,y+h-1) :
##            for j in range (x,x+w-1) :
##                    pt1x=pt1x+1
##                    pt2x=pt2x-1
##                else :
##                    val[pt2x][pt2y] = val[pt1x][pt1y]
##                    pt1x=pt1x+1
##                    pt2x=pt2x-1
        
        #reconstruct
        print "Reconstructing image... "
        merged=cv2.merge((hue,sat,val))
        merged=cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
##        cv2.imshow("Reconstructed", merged)
        return merged
       
if __name__ == "__main__":
    main()

import cv2

face_detect = cv2.CascadeClassifier("../haarcascade/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("../haarcascade/haarcascade_eye.xml")

smile_detect = cv2.CascadeClassifier("../haarcascade/haarcascade_smile.xml")
stream = cv2.VideoCapture(0)

while True:
    st,frame = stream.read()
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    
####### Face Detect
    faces= face_detect.detectMultiScale(gray_frame , 1.3 , 12)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y), (x+w,y+h) ,(0,0,255),3  )
        face_only = frame[y:y+h , x:x+w]


######### eye detect
        eyes= eye_detect.detectMultiScale(face_only )
        for (ex , ey , ew,  eh) in eyes:
            #cv2.rectangle(face_only , (ex,ey) , (ex+ew ,ey+eh) , (0,255,0) , 2)
            eye_x = int((ex+(ew/2)))-10
            eye_y = int((ey+(eh/2))) +10
            cv2.putText(face_only , "x" , (eye_x,eye_y) , cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),5)
##### smile detect
                                                        
        smiles = smile_detect.detectMultiScale(face_only,1.3,15)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(face_only , (sx,sy) , (sx+sw , sy+sh) ,(255,0,0) , 2)
####### Show   
   # cv2.imshow("face only"    , face_only )
    cv2.imshow("live stream" , frame)
    
    if cv2.waitKey(1) & 0xff == ord("x"):
        break

stream.release()

cv2.destroyAllWindows()

import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# enter unique number id for face
face_id = input('\n enter face id :   ')

print("\n Now camera is opening ...")

count = 0

while(True):

    ret, img = cam.read()
    img=cv2.flip(img,+1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 1)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # directory for DataSet where will be store all face sample
        cv2.imwrite("DataSet/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('dataSetCreater', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if count >= 100: # it will take 100 face image for the DataSet .
         break

print("Exiting Program")
cam.release()
cv2.destroyAllWindows()



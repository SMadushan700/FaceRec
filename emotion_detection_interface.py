import cv2
import numpy as np
from keras.models import load_model
# import tensorflow as tf
# from time import sleep
# from tensorflow.keras.preprocessing.image import img_to_array
# from keras.preprocessing import image

model=load_model('emotion_detection_model_file.h5')

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

video=cv2.VideoCapture(0)
while True:
    ret,frame=video.read()
   
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        print(labels_dict[label])
            
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k == ord('q'):  #Press q to exit
        break
video.release()
cv2.destroyAllWindows()


import os
import cv2
import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
import json
from keras.initializers import glorot_uniform
#Reading the model from JSON file
with open('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/fer.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
#model_j.summary()
model_j.load_weights('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/fer.h5')
# with open('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/model.json','r') as f:
#     model_json = json.load(f)

# model = model_from_json(model_json)
# model.load_weights('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/model_weights.h5')

# keras.models.load_model('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/model_weights.h5')
#model.load_weights('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/model_weights.h5')
face_haar_cascade = cv2.CascadeClassifier('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/haarcascade_frontalface_default.xml')

# cap  =  cv2.VideoCapture('/Users/kandagadlaashokkumar/Desktop/Facial_Expression_Recognition/videos/facial_exp.mkv')
cap  = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img,1.32,5)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray = gray_img[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels,axis = 0)
        img_pixels/=255
        predictions = model_j.predict(img_pixels)
        print(predictions)
        max_index = np.argmax(predictions[0])
        emotions = ('angry','disgust','fear','happy','sad','surprise','anger')
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)
        cv2.putText(img,predicted_emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        #cv2.putText(img,predicted_emotion,(75,75),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    resized = cv2.resize(img,(1000,700))
    cv2.imshow('Facial emotion analysis',resized)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
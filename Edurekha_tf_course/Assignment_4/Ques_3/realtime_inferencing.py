import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

image_size= (64,64)
classifier = MobileNetV2(include_top=True, weights=None, input_tensor=None, input_shape=image_size + (3,), pooling=None, classes=2)
# classifier.summary()
model_weights= r"model_weights.h5"
haarcascade_model= r"haarcascade_frontalface_default.xml"

classifier.load_weights(model_weights)
face_haar_cascade = cv2.CascadeClassifier(haarcascade_model)


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    # cv2.imshow('Facial emotion analysis ',test_img)
    if not ret:
        continue
    # gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(test_img, 1.32, 5)
    print(faces_detected)
    

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=test_img[y:y+w,x:x+h,:]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(64,64))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = classifier.predict(img_pixels)

        #find max indexed array
        max_index = np.argsort(predictions)[0][::-1][0]

        CLASSES= {"0":"Male","1":"Female"}
        gender = CLASSES[str(max_index)]

        cv2.putText(test_img, gender, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Gender detector ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
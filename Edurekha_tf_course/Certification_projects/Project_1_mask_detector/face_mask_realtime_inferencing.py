import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D,BatchNormalization,Dense

image_size= (64,64)
classifier = MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=image_size + (3,), pooling=None, classes=2)
# Finetuning the MobileNetV2 model
layer_list = [layers.name for layers in classifier.layers]
print('no of layers:> ',len(classifier.layers))

for layers in classifier.layers:
    layers.trainable = False

last_layer = classifier.get_layer('out_relu') #final layer
last_output = last_layer.output

x = GlobalMaxPooling2D()(last_output)
x = BatchNormalization()(x)

x = Dense(1, activation="sigmoid", name="pred")(x)
classifier = Model(classifier.input, x)

# classifier.summary()
model_weights= r"mask_detector_model_weights1.h5"
haarcascade_model= r"D:\Exploring-Tensorflow\Edurekha_tf_course\Assignment_4\Ques_3\haarcascade_frontalface_default.xml"

classifier.load_weights(model_weights) #loading teh weights
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
        print(predictions)

        #find max indexed array
        max_index = np.argsort(predictions)[0][::-1][0]

        CLASSES= {"0": "with_mask", "1": "without_mask"}
        gender = CLASSES[str(max_index)]

        cv2.putText(test_img, gender, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Gender detector ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
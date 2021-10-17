# Exploring-Tensorflow

## Table Of Contents
---------------------
- [Importing library snippets](#importing-library-snippets)
- [Plotting snippet](#plotting-snippet)
- [Display image snippet](#display-image-snippet)
- [Using pre-trained model snippets](#using-pre-trained-model-snippets)
- [Compiling and fitting model snippets](#compiling-and-fitting-model-snippets)
- [Prediction snippets](#prediction-snippets)
- [Augmentation snippets pre-processing layers for augmentation](#augmentation-snippets-pre-processing-layers-for-augmentation)
- [Awesome ways of using callbacks](#awesome-ways-of-using-callbacks)
- [Pre-trained model with grayscale images](#pre-trained-model-with-grayscale-images)
- [Tensorflow memory issues with colab](#Tensorflow-memory-issues-with-colab)
- [cv2.imshow in google colab](#cv2imshow-in-google-colab)
- [Disable eagar execution in tensorflow](#Disable-eagar-execution-in-tensorflow)
- [Color channel swapping for images](#Color-channel-swapping-for-images)
- [Multi-Label Classification](#Multi-Label-Classification)

## Useful code snippets

#### Importing library snippets

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation,GlobalMaxPooling2D,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image #PIL image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as keras_backend

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

```

#### Plotting snippet
```python
import matplotlib.pyplot as plt
%matplotlib inline
# list all data in training
print(training.history.keys())
# summarize training for accuracy
plt.plot(training.history['accuracy'])   # training is the variable from the fit method
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize traning for loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
#### Display image snippet
```python
CLASSES=  {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

plt.figure(figsize=(12, 12))
for i in range(0,9):
    plt.subplot(5, 3, i+1)
    plt.imshow(image_list[i], cmap="gray") #'image_list' is the list of images
    plt.xlabel(CLASSES[labels[i]]) # 'labels' is the list of labels
plt.tight_layout()
plt.show()
```

#### Using pre-trained model snippets
```python
from tensorflow.keras.applications import MobileNetV2  #name of the model to be used
from tensorflow.keras.models import Model #API for the wrapping

# different operations on the pre-trained model
pretrained_model= MobileNetV2(include_top=False,weights='imagenet',input_shape=input_shape)

# List of layers in the pretrained_model
layer_list = [layers.name for layers in pretrained_model.layers]

# make all the layers non-trainable
for layers in pretrained_model.layers:
    layers.trainable = False

# make the layers(by api names) as non-trainable 
for layers in pretrained_model.layers:
     if layers._keras_api_names[0] == 'keras.layers.BatchNormalization':
         layers.trainable = False

#to find the methods available
dir(pretrained_model)

```

#### Compiling and fitting model snippets
```python

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

Mobilenet_v2_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

training = Mobilenet_v2_model.fit(train_generator,
                   steps_per_epoch=100,epochs=50,
                   validation_data=validation_generator,
                       validation_steps=100,
                       callbacks=callbacks)
```

#### Prediction snippets
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/
```python
from tensorflow.keras.preprocessing import image
import numpy as np
img_pred = image.load_img("test_set/test_set/dogs/dog.4003.jpg",target_size=(150,150))

img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred, axis=0)

result = Efficientnet_model.predict(img_pred)
```

#### Augmentation snippets pre-processing layers for augmentation
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)
# applying augmentations on the image
for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image, axis=0)) #augmentation sequential layer
        plt.imshow(aug_img[0].numpy().astype("uint8")) #imshow need numpy array with 'unsigned-int8' precision
        plt.title("{}".format(format_label(label)))
        plt.axis("off")
```

#### Awesome ways of using callbacks
https://blog.paperspace.com/tensorflow-callbacks/


#### Pre-trained model with grayscale images
https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images

#### Tensorflow memory issues with colab
https://github.com/tensorflow/tensorflow/issues/31312

#### cv2.imshow in google colab
```python
from google.colab.patches import cv2_imshow
cv2_imshow("output.png")
```
https://stackoverflow.com/questions/55288657/image-is-not-displaying-in-google-colab-while-using-imshow


#### Disable eagar execution in tensorflow
```python
tf.compat.v1.disable_eager_execution()
```

#### Color channel swapping for images
https://www.scivision.dev/numpy-image-bgr-to-rgb/
```python
b,g,r = cv2.split("img.jpg")
data_rgb= cv2.merge([r,g,b])
```

#### Multi-Label Classification <a href='https://leetcode.com/novoforce/' target='_blank'><img src='https://img.shields.io/badge/-Tutorial-yellow?style=for-the-badge' border='0' alt='Leetcode'/></a>
https://github.com/novoforce/Exploring-Tensorflow/blob/main/multi_label_classification.ipynb

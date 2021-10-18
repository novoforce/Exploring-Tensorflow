# Exploring-Tensorflow

## Table Of Contents
---------------------
- <a href='https://github.com/novoforce/Exploring-Tensorflow#importing-library-snippets' target='_blank'><img src='https://img.shields.io/static/v1?label=Importing%20library&message=Snippet&color=blue&style=flat-square' border='0' alt='Multi-Label Classification'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#plotting-snippet' target='_blank'><img src='https://img.shields.io/static/v1?label=Plotting%20snippet&message=Snippet&color=blue&style=flat-square' border='0' alt='Plotting snippet'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Display-image-snippet' target='_blank'><img src='https://img.shields.io/static/v1?label=Display%20image%20snippet&message=Snippet&color=blue&style=flat-square' border='0' alt='Display image snippet'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Using-pre-trained-model-snippets' target='_blank'><img src='https://img.shields.io/static/v1?label=Using%20pre-trained%20model%20snippets&message=Snippet&color=blue&style=flat-square' border='0' alt='Using pre-trained model snippets'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Compiling-and-fitting-model-snippets' target='_blank'><img src='https://img.shields.io/static/v1?label=Compiling%20and%20fitting%20model%20snippets&message=Snippet&color=blue&style=flat-square' border='0' alt='Compiling and fitting model snippets'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Prediction-snippets' target='_blank'><img src='https://img.shields.io/static/v1?label=Prediction%20snippets&message=Snippet&color=blue&style=flat-square' border='0' alt='Prediction snippets'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Augmentation-snippets-pre-processing-layers-for-augmentation' target='_blank'><img src='https://img.shields.io/static/v1?label=Augmentation%20snippets%20pre-processing%20layers%20for%20augmentation&message=Snippet&color=blue&style=flat-square' border='0' alt='Augmentation snippets pre-processing layers for augmentation'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#Awesome-ways-of-using-callbacks' target='_blank'><img src='https://img.shields.io/static/v1?label=Awesome%20ways%20of%20using%20callbacks&message=Tutorial&color=yellow&style=flat-square' border='0' alt='Awesome ways of using callbacks'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#Pre-trained-model-with-grayscale-images' target='_blank'><img src='https://img.shields.io/static/v1?label=Pre-trained%20model%20with%20grayscale%20images&message=Tutorial&color=yellow&style=flat-square' border='0' alt='Pre-trained model with grayscale images'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#Tensorflow-memory-issues-with-colab' target='_blank'><img src='https://img.shields.io/static/v1?label=Tensorflow%20memory%20issues%20with%20colab&message=Issue&color=red&style=flat-square' border='0' alt='Tensorflow memory issues with colab'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#cv2imshow-in-google-colab' target='_blank'><img src='https://img.shields.io/static/v1?label=cv2imshow%20in%20google%20colab&message=Issue&color=red&style=flat-square' border='0' alt='cv2.imshow in google colab'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Disable-eagar-execution-in-tensorflow' target='_blank'><img src='https://img.shields.io/static/v1?label=Disable%20eagar%20execution%20in%20tensorflow&message=Snippet&color=blue&style=flat-square' border='0' alt='Disable eagar execution in tensorflow'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#Color-channel-swapping-for-images' target='_blank'><img src='https://img.shields.io/static/v1?label=Color%20channel%20swapping%20for%20images&message=Tutorial&color=yellow&style=flat-square' border='0' alt='Color channel swapping for images'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#multi-label-classification' target='_blank'><img src='https://img.shields.io/static/v1?label=Multi-Label%20Classification&message=Tutorial&color=yellow&style=flat-square' border='0' alt='Multi-Label Classification'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Download-files-to-colab' target='_blank'><img src='https://img.shields.io/static/v1?label=Download%20files%20to%20colab&message=Snippet&color=blue&style=flat-square' border='0' alt='Download files to colab'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Download-files-from-colab-to-local-system' target='_blank'><img src='https://img.shields.io/static/v1?label=Download%20files%20from%20colab%20to%20local%20system&message=Snippet&color=blue&style=flat-square' border='0' alt='Download files from colab to local system'/></a>
- <a href='https://github.com/novoforce/Exploring-Tensorflow#Import-python-files-from-different-location' target='_blank'><img src='https://img.shields.io/static/v1?label=Import%20python%20files%20from%20different%20location&message=Snippet&color=blue&style=flat-square' border='0' alt='Import python files from different location'/></a>
- - <a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/yolo_v4.ipynb' target='_blank'><img src='https://img.shields.io/static/v1?label=YoloV4&message=Tutorial&color=yellow&style=flat-square' border='0' alt='YoloV4'/></a>

## Useful code snippets

### Importing library snippets

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

### Plotting snippet
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
### Display image snippet
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

### Using pre-trained model snippets
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

### Compiling and fitting model snippets
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

### Prediction snippets
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/
```python
from tensorflow.keras.preprocessing import image
import numpy as np
img_pred = image.load_img("test_set/test_set/dogs/dog.4003.jpg",target_size=(150,150))

img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred, axis=0)

result = Efficientnet_model.predict(img_pred)
```

### Augmentation snippets pre-processing layers for augmentation
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

### Awesome ways of using callbacks
https://blog.paperspace.com/tensorflow-callbacks/


### Pre-trained model with grayscale images
https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images

### Tensorflow memory issues with colab
https://github.com/tensorflow/tensorflow/issues/31312

### cv2.imshow in google colab
```python
from google.colab.patches import cv2_imshow
cv2_imshow("output.png")
```
https://stackoverflow.com/questions/55288657/image-is-not-displaying-in-google-colab-while-using-imshow


### Disable eagar execution in tensorflow
```python
tf.compat.v1.disable_eager_execution()
```

### Color channel swapping for images
https://www.scivision.dev/numpy-image-bgr-to-rgb/
```python
b,g,r = cv2.split("img.jpg")
data_rgb= cv2.merge([r,g,b])
```

### Multi-Label Classification
https://github.com/novoforce/Exploring-Tensorflow/blob/main/multi_label_classification.ipynb

### Download files to colab
```python
!wget -O <output file with extension> --no-check-certificate "<download link>"

! curl <download link without quotes> > <output_file>

# If public sharing link is available
!gdown --id 1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
```
If shared link is only for you:

    https://stackoverflow.com/questions/62759748/downloading-data-from-a-shared-google-drive-link-in-google-colab
Extra links:

    https://stackoverflow.com/questions/48735600/file-download-from-google-drive-to-colaboratory
       
### Download files from colab to local system
```python
!zip -r /content/file.zip /content/Folder_To_Zip

from google.colab import files
files.download("/content/file.zip")
```




### Import python files from different location
```python
import sys 
import os
sys.path.append(os.path.abspath("/home/el/foo4/stuff")) # attach the path of the directory to be included
from riaa import *
watchout()
```
https://stackoverflow.com/questions/2349991/how-to-import-other-python-files

















# References for the badges
Tutorial:

https://img.shields.io/static/v1?label=put-custom-label&message=Tutorial&color=yellow&style=flat-square
<a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#' target='_blank'><img src='https://img.shields.io/static/v1?label=put%20label&message=Tutorial&color=yellow&style=flat-square' border='0' alt='put alternate text'/></a>


Snippet:

https://img.shields.io/static/v1?label=put-custom-label&message=Snippet&color=blue&style=flat-square
<a href='https://github.com/novoforce/Exploring-Tensorflow#' target='_blank'><img src='https://img.shields.io/static/v1?label=put%20label&message=Snippet&color=blue&style=flat-square' border='0' alt='put alternate text'/></a>

Issue:

https://img.shields.io/static/v1?label=put-custom-label&message=Issue&color=red&style=flat-square
<a href='https://github.com/novoforce/Exploring-Tensorflow/blob/main/README.md#' target='_blank'><img src='https://img.shields.io/static/v1?label=put%20label&message=Issue&color=red&style=flat-square' border='0' alt='put alternate text'/></a>

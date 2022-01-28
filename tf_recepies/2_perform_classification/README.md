# Perform Classification
--------------------------------------------------------

## Binary Classification
--------------------------------------------------------
### API Snippets Used

#### Loading image
```python
image = load_img(image_path, target_size=(32, 32),color_mode='grayscale')
image = img_to_array(image)
```
#### Generic Image loader functions
Problem name:
Detect whether a face in the photo is a smiling face or not ?
* positive class: Image with smile
* negative class: Image without smile

Dataset format
Datasets are subdivided into different folders namely "positives" and "negatives"
* X: png/jpg image format
* Y: Path of the image.
    * For eg:
        * Positive class:>
            "SMILEsmileD/SMILEs/positives/positives7/9039.jpg"
        * Negative class:>
            "SMILEsmileD/SMILEs/negatives/negatives7/5472.jpg"

How to process the data for the model ?

For training the model we need data in the array format so,
the below load_images_and_labels() will load the images and extract the labels from
the path of the image, and finally convert to list array of images and labels.

```python
def load_images_and_labels(image_paths):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=(32, 32),
                         color_mode='grayscale')
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]
        label = 'positive' in label
        label = float(label)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)
```

#### for enumerating files inside list
```python
files_pattern = "SMILEsmileD/SMILEs/*/*/*.jpg"

dataset_paths = [*glob.glob(files_pattern)]
```

#### Stratify means to rearrange between subgroups
```python

(X_train, X_val,
 y_train, y_val) = train_test_split(X_train, y_train,
                                    test_size=0.2,
                                    stratify=y_train,
                                    random_state=999)

```

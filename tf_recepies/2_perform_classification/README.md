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
--------------------------------------------------------
## Multi Class Classification
--------------------------------------------------------
### API Snippets Used
Problem name:
* Classify whether the image is a "stone", "paper", "scissor"
* No of classes: 3

Dataset format:
* The directories containing the class images with the names of the folder as corresponding class names
* for eg:
    * "stone"
        * "img1.png"
        * "img2.png"
        * "img3.png"
    * "paper"
        * "img1.png"
        * "img2.png"
        * "img3.png"
    * "scissor"
        * "img1.png"
        * "img2.png"
        * "img3.png"

#### Loading the image

```python
image = tf.io.read_file(image_path) #read image and return tensor of contents
image = tf.image.decode_jpeg(image, channels=3) #jpeg encoded tensor is converted to uint8 tensor with RGB values and return it
image = tf.image.rgb_to_grayscale(image)
image = tf.image.convert_image_dtype(image, np.float32) #change the datatype of the image to float32
image = tf.image.resize(image, target_size)
```
ratio parameter can be added to the "tf.image.decode_jpeg()" which will downscale the image based on the value.
Alternatively "tf.io.decode_image()" can be used which is more cleaner
default value is 1.
Allowed values are: 1,2,4 and 8.

for more info on other options available:
https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg
#### Generic Image loader function using TF
* Good data-pipleine TF API blog:> https://cs230.stanford.edu/blog/datapipeline/
* Good snippets from official TF blog:> https://www.tensorflow.org/api_docs/python/tf/data/Dataset
```python
def load_image_and_label(image_path, target_size=(32, 32)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES)  # One-hot encode. #This will be run in looping fashion using map functionality
    label = tf.dtypes.cast(label, tf.float32)

    return image, label

def prepare_dataset(dataset_path,buffer_size,batch_size,shuffle=True):
    """
    dataset_path: list of image paths
    buffer_size: memory buffering size
    batch_size: batching size
    shuffle: true/false
    """
    dataset = (tf.data.Dataset
               .from_tensor_slices(dataset_path) #
               .map(load_image_and_label,num_parallel_calls=AUTOTUNE))

    if shuffle:
        dataset.shuffle(buffer_size=buffer_size)
    dataset = (dataset.batch(batch_size=batch_size).prefetch(buffer_size=buffer_size))
    return dataset
```
The "tf.data.Dataset.from_tensor_slices(dataset_path)" will iterate through the list of images and
apply using .map() this function "load_image_and_label()" to each of the image iteratively using parallel execution
and return tensors.

for more info on dataset.shuffle:
here "buffer_size" should be "len(dataset_path)" which will make sure shuffle is performed properly
https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

--------------------------------------------------------
## Multi Label Classification
--------------------------------------------------------
Problem name: Predict labels for a product in the image.

For eg: For a image of red shirt

The labels are: "Red" "Shirt"

* Dataset format: 
    * Image folder containing images of the products
    * A csv file containing the following attributes/columns:

id | gender | masterCategory | subCategory | articleType | baseColour | season | year | usage | productDetails
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
59263 | Women | Accessories | Watches | Watches | Silver | Winter | 2016 | Casual | Titan Women Silver Watch
15970 | Men | Apparel | Topwear | Shirts | Navy Blue | Fall | 2011 | Casual | Turtle Check Men Navy Blue Shirt

**Out of the above attributes we only need "Gender" and "Usage" attributes for our model prediction**


### API Snippets Used
#### Data loading snippet

input:> CSV file with 10 columns/attributes and "n" rows
The following snippet will convert the csv data to key value pairs,
where "key" is "id" and values are other attributes

At line 191, it is a good usecase of dictionary comprehension.
```python
with open(styles_path, 'r') as f:
    dict_reader = DictReader(f)
    STYLES = [*dict_reader] #list of ordered dicts as shown below
    """ 
    OrderedDict([('id', '15970'), ('gender', 'Men'), ('masterCategory', 'Apparel'), ('subCategory', 'Topwear'), 
    ('articleType', 'Shirts'), ('baseColour', 'Navy Blue'), ('season', 'Fall'), ('year', '2011'), ('usage', 'Casual'),
    ('productDisplayName', 'Turtle Check Men Navy Blue Shirt')])
    """
    article_type = 'Watches'
    genders = {'Men', 'Women'}
    usages = {'Casual', 'Smart Casual', 'Formal'}
    STYLES = {style['id']: style
              for style in STYLES
              if (style['articleType'] == article_type and
                  style['gender'] in genders and
                  style['usage'] in usages)}
#final output:
# key:> 59263 value:> OrderedDict([('id', '59263'), ('gender', 'Women'), ('masterCategory', 'Accessories'), ('subCategory', 'Watches'), ('articleType', 'Watches'), ('baseColour', 'Silver'), ('season', 'Winter'), ('year', '2016'), ('usage', 'Casual'), ('productDisplayName', 'Titan Women Silver Watch')])
```
#### Snippet for filtering data using functional() python methods
```python
image_paths = [*filter(lambda p: p.split(os.path.sep)[-1][:-4] in STYLES.keys(),image_paths)]
```

#### Multi-label binarization API
A good link for Multi-label binarizatio:> https://www.kaggle.com/questions-and-answers/66693
A good link from the official docs:> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

```python
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
"""
mlb.classes_ : This will return the classes names which were encoded.
mlb.inverse_transform(output_from_the_model) #since model will predict labels(not human understandable) and inverse_transform will convert back to the actual labels.
"""
```




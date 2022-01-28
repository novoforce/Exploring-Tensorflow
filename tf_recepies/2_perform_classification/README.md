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
#### for enumerating files inside list
```python
files_pattern = "SMILEsmileD/SMILEs/*/*/*.jpg"

dataset_paths = [*glob.glob(files_pattern)]
```

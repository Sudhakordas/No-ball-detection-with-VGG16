# Legal-balls-and-No-balls-Image-Classification

# Objective
Objective of this project is to create a model that can classify a Legal ball or No ball.


## Intoduction
Cricket is one of the most popular game in the recent world. There are some actionswhich create controversy in a cricket match. Signalling a delivary as legal and No is 
one of the major task of the umpire. Any wrong decision can totally change the scenario of a cricket match. There is a need of automate system which can roughly provide the probability of bowled being delevered is legal or not. I create this end  to end web app which can classify a image properly. Where i used manually created CNN model and transfer learning model(VGG16).

### Legal ball

![Crick](https://github.com/Sudhakordas/No-ball-detection-with-VGG16/blob/master/static/image/Crick-1.JPG)

### No ball

![Crick-2](https://github.com/Sudhakordas/No-ball-detection-with-VGG16/blob/master/static/image/Cricket-2.JPG)


 


### Workflow

First challenges in this project is to collect tthe dataset. As there was no available dataset i have to create tthe dataset. That's i gather data from google search and many live matches. As collected dataset was not too large to make the model robust i used Data augmentation technique to make the dataet of almost 3500 imgages. 

```python
from keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img

Datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range = 30,
        width_shift_range = .1,
        height_shift_range = .1,
        shear_range = .3,
        #zoom_range = .0001,
        horizontal_flip = True,
       # vertical_flip=True,
        fill_mode = 'nearest'
        
)
```
Then i import the VGG16 model and adjust according to our dataset and fit the model with imagesize of [100,100]. which provides a training accuracy of almost 100%(seems to be overfitted) and test accuracy  of 80%. And to make this project end to end i used the flask frmaework to build the backend and basic html to create the frontend. 

```python

#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator ,array_to_img, img_to_array, load_img

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

 
# Model saved with Keras model.save()
MODEL_PATH = 'No_ball_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
 
print('@@ Model loaded')
 
 
def pred_mode(img):
  test_image = load_img(img, target_size = (100, 100)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict class horse or human
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Legal ball" 
  else:
    return "No ball"
 
     
 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
   
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user-upload', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred = pred_mode(file_path)
               
        return render_template('predict.html', pred_output = pred, user_image = file_path)
     
#Fo local system
if __name__ == "__main__":
    app.run(threaded=False,) 

```

## How to run this project in your system.
1. Down load or clone the repository
```python
git clone https://github.com/Sudhakordas/No-ball-detection-with-VGG16.git
```
2. Create a new environment.
3. Activate that environment 
 ```python
conda activate environment_name
```
4. Install all the denpendencies.
```python
pip install -r requirements.txt
```
5. Now run the project.
 ### To run the web app.
 Go to the directory where you have clone the repository.
 Type 
 ```python
 python app.py
  ```

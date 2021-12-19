
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
  #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
     
# #Fo AWS cloud
# if __name__ == "__main__":
#     app.run(host='0.0.0.0.0', post='8080',threaded=False,) 
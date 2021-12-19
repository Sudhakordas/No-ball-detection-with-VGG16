# Legal-balls-and-No-balls-Image-Classification

# bjective
Objective of this project is to create a model that can classify a Legal ball or No ball.


## Intoduction
Cricket is one of the most popular game in the recent world. There are some actionswhich create controversy in a cricket match. Signalling a delivary as legal and No is 
one of the major task of the umpire. Any wrong decision can totally change the scenario of a cricket match. There is a need of automate system which can roughly provide the probability of bowled being delevered is legal or not. I create this end  to end web app which can classify a image properly. Where i used manually created CNN model and transfer learning model(VGG16).


 


### Methods / Algorithms 
 
We have deployed a Convolution Neural Network (CNN) based classification method with VGG19 to automatically detect and differentiate foot overstepping no balls from fair balls.
We have used Transfer learning algorithms which uses the knowledge gained from solving one problem and applying it to another related problem. Transfer learning aims to transfer knowledge from a large dataset known as source domain to a smaller dataset named target domain. 
In our model, we have used 5674 images of size 100 x 100 x 3  as input. Our input dataset contains images collected from google image search and various video clips from live matches.
Some of the techniques used to increase our image dataset are:
*Randomized Cropping
*Changing contrast in various proportions
*Changing brightness
*Horizontal flipping

The images are manually annotated and contains two classes:
No-ball
Legal-ball

We have used Keras and Tensorflow2.0 to build our model and generate results. Our model produces a score for both possible outcomes then each of them is converted to a probability by Sigmoid activation function.


### How to work with the Model?

Upload the test data set on google drive.
Give the path of the dataset folder on the drive to variable path.
Give  ‘y’ (correct output of images to be tested) as text file y.txt.
The model will print accuracy score, precision, recall and F1 score for the test data

### Model used for transfer learning
VGG 19

### Number of hiddel layers
20

### Number of epochs
30

### Optimizer
adam

### Metrics for evaluation
Accuracy

### Model train accuracy
94.88 %

### Model test accuracy
89.45 %

### Precision
0.7722

### Recall
0.9950

### F1 score
0.8696
 


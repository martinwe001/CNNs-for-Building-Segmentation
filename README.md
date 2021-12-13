# Specialized project

Created by [Katrine Nguyen](https://github.com/katrineng) and [Martin Wangen-Eriksen](https://github.com/martinwe001) as a part of our specialized project at Norwegian University of Science and Technology [(NTNU)](https://www.ntnu.edu/).

<<<<<<< HEAD
## Models

Most of our code and the U-net model is significantly inspired by this project [Unet-for-Person-Segmentation](https://github.com/nikhilroxtomar/Unet-for-Person-Segmentation). The SegNet model we created on our own based on other implementations of SegNet in Tensorflow.

## Data

The model is trained and tested on Massachusetts Buildings Dataset from [Kaggle](https://www.kaggle.com/balraj98/massachusetts-buildings-dataset). The original images where 1500X1500 pixels each over an area of 1500x1500 meters (1mx1m resolution). The original 137 images were cropped into 64x64 pixels and images without building were filtered out. 

Images of data set

To make the masks compatible with our model the masks was changed from white (255,255,255) labels to greyscale with value 1. 

## Running the project

### Requirements

### Training

### Testing

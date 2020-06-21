from evalbody_singleposemodel import *
import cv2
import tensorflow
from tensorflow.keras.preprocessing.image import load_img


imagePath = 'SideShotMe2.jpg'
modelPath = 'bodypix_resnet50_float_model-stride16/model.json'

# read in a single frame of the image
img = load_img(imagePath)

imageProcess(img, modelPath, plotting=True)

# plotIDImages(imagePath)


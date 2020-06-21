from evalbody_singleposemodel import *
import cv2
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageSequence

# paths to load
vidPath = 'kneeMov.MOV'
modelPath = 'bodypix_resnet50_float_model-stride16/model.json'

# read in a single frame of the image as a PIL object
cap = cv2.VideoCapture(vidPath)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# count the frames
n = 0
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    n += 1

    print("Frame " + str(n))

    # process every 10th loaded frame
    if (ret == True) & (n%50 == 0):

        # Load the image, rotate the image for the correct orientation and convert to rgb channels
        frame = cv2.cvtColor(np.rot90(np.array(frame), 3), cv2.COLOR_BGR2RGB) 

        # Display the resulting frame
        plt.imshow(frame); plt.show()

        imageProcess(frame, modelPath, plotting=True)



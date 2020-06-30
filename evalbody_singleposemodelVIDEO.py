from evalbody_singleposemodel import *
import cv2
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageSequence

# paths to load
vidPath = 'kneeMov2.MOV'
modelPath = 'bodypix_resnet50_float_model-stride16.nosync/model.json'
modelPath = 'bodypix_mobilenet_float_050_model-stride8.nosync/model.json'
# read in a single frame of the image as a PIL object
cap = cv2.VideoCapture(vidPath)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# count the frames
n = 0

outputs = list()
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()

    print("Frame " + str(n))

    # process every 10th loaded frame
    if (ret == True) & (n%50 == 0) & (n <= 100):

        # Load the image, rotate the image for the correct orientation and convert to rgb channels
        frame = cv2.cvtColor(np.rot90(np.array(frame), 3), cv2.COLOR_BGR2RGB) 

        x, y, a = frame.shape

        s = x/y
        xn = 300
        frame = cv2.resize(frame, (xn, int(xn*s)))

        # Display the resulting frame
        # plt.imshow(frame); plt.show()

        # save the dictionary outputs into a list
        outputs.append(imageProcess(frame, modelPath, plotting=True))

    elif n > 100:
        break

    n += 1


for o in outputs:
    bg = o['background']
    fg = o['foreground']

    kernel = np.ones((3, 3),np.uint8)
    fgd = cv2.dilate(fg, kernel, iterations=10)
    plt.imshow(fgd); plt.show()

pass

# Get a time stamp for data
# get the x and y positions of the joint positions of some description
# CURRENTLY SET UP FOR 720 footage
# send to julia the raw bodypix co-ordinates for knee, hip and ankle


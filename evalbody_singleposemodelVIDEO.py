from evalbody_singleposemodel import *
import cv2
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageSequence
from glob import glob

# paths to load
vidDir = 'TestingVids/'
modelPath = 'bodypix_resnet50_float_model-stride16.nosync/model.json'
# modelPath = 'bodypix_mobilenet_float_050_model-stride8.nosync/model.json'
# read in a single frame of the image as a PIL object

vidPaths = glob(vidDir + "20200526_125332*.mp4")
for vidPath in vidPaths:

    print("\nVideo: " + vidPath + "\n")

    cap = cv2.VideoCapture(vidPath)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # count the frames
    n = 0
    outputs = list()
    jointMovement = {}
    jointMovement['leftKnee'] = list()
    jointMovement['leftAnkle'] = list()
    jointMovement['leftHip'] = list()
    

    f = open(vidPath + 'jointmovement.txt', 'w')
    f.write('frame,knee_x,knee_y,ankle_x,ankle_y,hip_x,hip_y\n')

    maxFrames = 40
    print("\n ----- Mask making -----")
    while(cap.isOpened()):

        # NOTE if I go and save frames in a list or something then I could parallelise
        # this whole thing by processing each from, returning the mask and adding it 
        # all together at the end

        # Capture frame-by-frame
        ret, frame = cap.read()

        # create a mask from the first 4 seconds of footage
        if (n%5 == 0) & (n < maxFrames):
                
            print("Frame " + str(n))
            # Load the image, rotate the image for the correct orientation and convert to rgb channels
            # frame = cv2.cvtColor(np.rot90(np.array(frame), 3), cv2.COLOR_BGR2RGB) 
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB) 

            x, y, a = frame.shape

            s = x/y
            xn = 500
            frame = cv2.resize(frame, (xn, int(xn*s)))

            # get image data
            o = imageProcess(frame, modelPath, plotting=False)

            
            knee_x, knee_y = np.round(o['allocatedKeypoints']['leftKnee'], 2)
            ankle_x, ankle_y = np.round(o['allocatedKeypoints']['leftAnkle'], 2)
            hip_x, hip_y = np.round(o['allocatedKeypoints']['leftHip'], 2)
            f.write(str(n) + "," + str(knee_x) + "," + str(knee_y) + "," + str(ankle_x) + "," + str(ankle_y) + "," + str(hip_x) + "," + str(hip_y) + "\n")

            '''
            jointMovement['leftKnee'].append(np.array([knee_x, knee_y]))
            jointMovement['leftAnkle'].append(np.array([ankle_x, ankle_y]))
            jointMovement['leftHip'].append(np.array([hip_x, hip_y]))
            '''


            # for only the initial frame 
            if n == 0:
                x, y = o['mask'].shape      
                maskAccumulateWB = np.zeros([x, y])     # whole body
                maskAccumulateUB = np.zeros([x, y])     # upper body

            # store the masks
            maskAccumulateWB += o['mask']  
            maskAccumulateUB += o['upperBodyMask']   

            # SUM ALL THE MASKS FROM BODYPIX ACROSS ALL FRAMES AND THEN RE-APPLY 
            # ONE BIG MASK TO THE WHOLE VIDEO WHICH CONTAINS ONLY THE BODY AND WHERE
            # THE LEG GOES THROUGHOUT THE VIDEO

        elif (n >= maxFrames) or (ret == False):

            # use the accumulated mask
            maskcWB = np.array(maskAccumulateWB).copy()         # create a copy to modify 
            maskcUB = np.array(maskAccumulateUB).copy()

            # perform dilation of the mask to smooth out edges and create a larger bouding area of the mask. reshape for the frame
            kernel = np.ones((3, 3),np.uint8)

            maskdiWB = cv2.resize(cv2.dilate(maskcWB, kernel, iterations=1), (frame.shape[1], frame.shape[0]))
            maskdiUB = cv2.resize(cv2.dilate(maskcUB, kernel, iterations=2), (frame.shape[1], frame.shape[0]))

            # turn the mask into a binary
            maskdiWBm = (maskdiWB > 1) * 1
            maskdiUBm = (maskdiUB < 10) * 1

            # combine the whole body and upper body to identify only the lower body
            masktarget = (maskdiWBm * maskdiUBm).astype(np.uint8)
            # plt.imshow(masktarget); plt.show()

            cap.release()
            f.close()
            break

        # once processing complete for max frames, create a full ROM mask
        n += 1
    
    
    print("\n ----- Video making -----")
    cap = cv2.VideoCapture(vidPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vidPath + "segmented.avi", fourcc, 30, (frame.shape[1], frame.shape[0]))  # use previous frame size
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    n = 0
    while(cap.isOpened()):

        if n%10 == 0:
            print("Frame " + str(n))

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        frameM = frame*np.expand_dims(masktarget, -1)

        if cv2.waitKey(1) == ord('q'):
            break
        out.write(frameM)

        n += 1

        
    out.release()
    cap.release()
    cv2.destroyAllWindows()



# Get a time stamp for data
# get the x and y positions of the joint positions of some description
# CURRENTLY SET UP FOR 720 footage
# send to julia the raw bodypix co-ordinates for knee, hip and ankle


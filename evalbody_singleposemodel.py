import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import cv2
from utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# perform inferencing on a single frame
def imageProcess(img, modelPath, plotting = False):

    # processes a 3D numpy array on the body pix algorithm and extracts the key points
    # Inputs:   (img), numpy array of the image (RGB colour and MUST be rotated to be vertical person position)
    #           (modelPath), bodypix model for inferencing
    #           (plotting), boolean controlling if there is visualisation of the inference
    # Outputs:  (output), dictionary containing all the inferenced information 

    # CONSTANTS
    OutputStride = 16

    KEYPOINT_NAMES = [
        "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
        "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
        "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    ]


    KEYPOINT_IDS = {name: id for id, name in enumerate(KEYPOINT_NAMES)}

    CONNECTED_KEYPOINTS_NAMES = [
        ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
        ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
        ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
        ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
        ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
        ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
    ]

    CONNECTED_KEYPOINT_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b])
                                for a, b in CONNECTED_KEYPOINTS_NAMES]

    PART_CHANNELS = [
        'left_face',
        'right_face',
        'left_upper_arm_front',
        'left_upper_arm_back',
        'right_upper_arm_front',
        'right_upper_arm_back',
        'left_lower_arm_front',
        'left_lower_arm_back',
        'right_lower_arm_front',
        'right_lower_arm_back',
        'left_hand',
        'right_hand',
        'torso_front',
        'torso_back',
        'left_upper_leg_front',
        'left_upper_leg_back',
        'right_upper_leg_front',
        'right_upper_leg_back',
        'left_lower_leg_front',
        'left_lower_leg_back',
        'right_lower_leg_front',
        'right_lower_leg_back',
        'left_feet',
        'right_feet'
    ]

    print("Loading model...", end="")
    graph = load_graph_model(modelPath)  # downloaded from the link above
    print("done.\nLoading sample image...", end="")

    # load sample image into numpy array
    imgHeight, imgWidth, imgDim = img.shape

    targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
    targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1

    print(imgHeight, imgWidth, targetHeight, targetWidth)
    # img = img.resize((targetWidth, targetHeight))

    img = cv2.resize(img, (targetWidth, targetHeight))
    x = img.copy()      # this is just to minimise changes to the code from the original version

    # x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    InputImageShape = x.shape
    print("Input Image Shape in hwc", InputImageShape)


    widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
    heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1
    print('Resolution', widthResolution, heightResolution)

    # Get input and output tensors
    input_tensor_names = get_input_tensors(graph)
    print(input_tensor_names)
    output_tensor_names = get_output_tensors(graph)
    print(output_tensor_names)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

    # Preprocessing Image
    # For Resnet
    if any('resnet_v1' in name for name in output_tensor_names):
        # add imagenet mean - extracted from body-pix source
        m = np.array([-123.15, -115.90, -103.06])
        x = np.add(x, m)
    # For Mobilenet
    elif any('MobilenetV1' in name for name in output_tensor_names):
        x = (x/127.5)-1
    else:
        print('Unknown Model')
    sample_image = x[tf.newaxis, ...]
    print("done.\nRunning inference...", end="")

    # NOTE THESE ARE THE RESULTS
    # evaluate the loaded model directly
    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_tensor_names, feed_dict={
                        input_tensor: sample_image})
    print("done. {} outputs received".format(len(results)))  # should be 8 outputs

    output = {}

    for idx, name in enumerate(output_tensor_names):
        if 'displacement_bwd' in name:
            print('displacement_bwd', results[idx].shape)
        elif 'displacement_fwd' in name:
            print('displacement_fwd', results[idx].shape)
        elif 'float_heatmaps' in name:
            heatmaps = np.squeeze(results[idx], 0)
            output['heatmaps'] = heatmaps
            print('heatmaps', heatmaps.shape)
        elif 'float_long_offsets' in name:
            longoffsets = np.squeeze(results[idx], 0)
            output['longoffsets'] = longoffsets
            print('longoffsets', longoffsets.shape)
        elif 'float_short_offsets' in name:
            offsets = np.squeeze(results[idx], 0)
            output['offsets'] = offsets
            print('offests', offsets.shape)
        elif 'float_part_heatmaps' in name:
            partHeatmaps = np.squeeze(results[idx], 0)
            output['partHeatmaps'] = partHeatmaps
            print('partHeatmaps', partHeatmaps.shape)
        elif 'float_segments' in name:
            segments = np.squeeze(results[idx], 0)
            output['segments'] = segments
            print('segments', segments.shape)
        elif 'float_part_offsets' in name:
            partOffsets = np.squeeze(results[idx], 0)
            output['partOffsets'] = partOffsets
            print('partOffsets', partOffsets.shape)
        else:
            print('Unknown Output Tensor', name, idx)

    # Segmentation mask
    segmentation_threshold = 0.7
    segmentScores = tf.sigmoid(segments)
    mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
    print('maskshape', mask.shape)

    # -------------- PLOTTING RESULTS --------------

    if plotting:

        # partOffsetVector, partHeatmapPositions, partPositions, partScores, partMasks = pltSegmentation(img, segments, OutputStride, segmentation_threshold, mask, targetWidth, targetHeight)
       
        fg, bg = pltSegmentation(img, segments, OutputStride, mask, plotting)

        returnHeat = HeatMap(mask, partHeatmaps, partOffsets, offsets, heatmaps, PART_CHANNELS, OutputStride, plotting = False)

        pltPoints(img, CONNECTED_KEYPOINT_INDICES, KEYPOINT_NAMES, returnHeat['keypointPositions'], plotting = False)

        # PRINT KEYPOINT CONFIDENCE SCORES
        print("Keypoint Confidence Score")
        for i, score in enumerate(returnHeat['keyScores']):
            print(KEYPOINT_NAMES[i], score)

        # PRINT POSE CONFIDENCE SCORE
        print("\nPose Confidence Score", np.mean(np.asarray(returnHeat['keyScores'])))

    return(output)

# plotting the masks
def pltSegmentation(img, segments, OutputStride, mask, plotting):

    # plots the masks from segmentation which identify the foreground and background
    # Inputs:   (img), numpy array of the frame
    #           (segments), inference output
    #           (OutputStride), size of the filter moving across the image
    #           (mask), ??? TBC but its function is to differentiate the fg and bg
    #           (plotting), boolean whether to plot the results 
    # Outputs:  (fg), the foreground identifying the outline of the person
    #           (bg), the background identifying everything non-person

    targetHeight, targetWidth , imgDim = img.shape

    segmentationMask = tf.dtypes.cast(mask, tf.int32)
    segmentationMask = np.reshape(
        segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))
    print('maskValue', segmentationMask[:][:])

    # Draw Segmented Output
    mask_img = Image.fromarray(segmentationMask * 255)
    mask_img = mask_img.resize(
        (targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    fg = np.bitwise_and(img, np.array(
    mask_img))

    segmentationMask_inv = np.bitwise_not(mask_img)
    bg = np.bitwise_and(img, np.array(
            segmentationMask_inv))

    if plotting:
        plt.clf()
        plt.title('Segmentation Mask')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(segmentationMask * OutputStride)
        plt.show()
        plt.title('Foreground Segmentation')
        plt.imshow(fg)
        plt.show()
        plt.title('Background Segmentation')
        plt.imshow(bg)
        plt.show()
    
    return(fg, bg)

# plotting the heat maps
def HeatMap(mask, partHeatmaps, partOffsets, offsets, heatmaps, PART_CHANNELS, OutputStride, plotting):

    # this function calculates some of the key scores such as detected positions and 
    # scores of the detected points and optionally produces heat maps visually representing
    # these results
    # Input:    (img), numpy array of the loaded image 
    #           (partHeatMaps), inference output
    #           (partOffSets), inference output
    #           (offsets), inference output
    #           (heatmaps), inference output
    #           (PART_CHANNELS), body locations being located
    #           (OutputStride), size of the filter moving across the image
    #           (plotting), boolean whether to plot the results
    # Output:   (returns), dictionary containing the key information:
    #               offsetVector - ?
    #               heatmapPositions - ?
    #               keypointPositions - ?
    #               keyScores - probability of match

    # BODYPART SEGMENTATION
    partOffsetVector = []
    partHeatmapPositions = []
    partPositions = []
    partScores = []
    partMasks = []

    # Part Heatmaps, PartOffsets,
    for i in range(partHeatmaps.shape[2]):

        heatmap = partHeatmaps[:, :, i]  # First Heat map
        heatmap[np.logical_not(tf.math.reduce_any(mask, axis=-1).numpy())] = -1
        # Set portions of heatmap where person is not present in segmentation mask, set value to -1

        heatmap_sigmoid = tf.sigmoid(heatmap)
        y_heat, x_heat = np.unravel_index(
            np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

        partHeatmapPositions.append([x_heat, y_heat])
        partScores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
        # Offset Corresponding to heatmap x and y
        x_offset = partOffsets[y_heat, x_heat, i]
        y_offset = partOffsets[y_heat, x_heat, partHeatmaps.shape[2]+i]
        partOffsetVector.append([x_offset, y_offset])

        key_x = x_heat * OutputStride + x_offset
        key_y = y_heat * OutputStride + y_offset
        partPositions.append([key_x, key_y])

        # SHOW HEATMAPS
        if plotting:
            plt.clf()
            plt.title('Heatmap: ' + PART_CHANNELS[i])
            plt.ylabel('y')
            plt.xlabel('x')
            plt.imshow(heatmap * OutputStride)
            plt.show()

            print('partheatmapPositions', np.asarray(partHeatmapPositions).shape)
            print('partoffsetVector', np.asarray(partOffsetVector).shape)
            print('partkeypointPositions', np.asarray(partPositions).shape)
            print('partkeyScores', np.asarray(partScores).shape)


    # POSE ESTIMATION
    offsetVector = []
    heatmapPositions = []
    keypointPositions = []
    keyScores = []
    for i in range(heatmaps.shape[2]):
        heatmap = heatmaps[:, :, i]  # First Heat map
        # SHOW HEATMAPS
        '''
        plt.clf()
        plt.title('Heatmap' + str(i) + KEYPOINT_NAMES[i])
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(heatmap * OutputStride)
        plt.show()
        '''

        heatmap_sigmoid = tf.sigmoid(heatmap)
        y_heat, x_heat = np.unravel_index(
            np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

        heatmapPositions.append([x_heat, y_heat])
        keyScores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
        # Offset Corresponding to heatmap x and y
        x_offset = offsets[y_heat, x_heat, i]
        y_offset = offsets[y_heat, x_heat, heatmaps.shape[2]+i]

        offsetVector.append([x_offset, y_offset])
        key_x = x_heat * OutputStride + x_offset
        key_y = y_heat * OutputStride + y_offset
        keypointPositions.append([key_x, key_y])

    returns = {
    'offsetVector': offsetVector,
    'heatmapPositions': heatmapPositions,
    'keypointPositions': keypointPositions,
    'keyScores': keyScores
    }

    return(returns)

# plotting the processed points
def pltPoints(img, CONNECTED_KEYPOINT_INDICES, KEYPOINT_NAMES, keypointPositions, plotting):

    # plots where the key locations points are on the identified person in frame
    # Inputs:
    # Outputs:  (), none but if called will produce images showing the detected points

    # function for pltPoints which get co-ordinates of boxed areas
    def getBoundingBox(keypointPositions, offset=(10, 10, 10, 10)):
        minX = math.inf
        minY = math.inf
        maxX = - math.inf
        maxY = -math.inf
        for x, y in keypointPositions:
            if (x < minX):
                minX = x
            if(y < minY):
                minY = y
            if(x > maxX):
                maxX = x
            if (y > maxY):
                maxY = y
        return (minX - offset[0], minY-offset[1]), (maxX+offset[2], maxY + offset[3])

    if plotting:

        # Get Bounding BOX
        (xmin, ymin), (xmax, ymax) = getBoundingBox(
        keypointPositions, offset=(0, 0, 0, 0))

        # Show Bounding BOX
        implot = plt.imshow(img)
        # Get the current reference
        ax = plt.gca()
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                linewidth=1, edgecolor='r', facecolor='none', fill=False)

        # Add the patch
        ax.add_patch(rect)
        plt.show()

        # Show all keypoints
        plt.figure(0)
        plt.imshow(img)

        x_points = []
        y_points = []
        for i, [x, y] in enumerate(keypointPositions):
            x_points.append(x)
            y_points.append(y)
        plt.scatter(x=x_points, y=y_points, c='r', s=40)
        plt.show()


        # DEBUG KEYPOINTS
        #  Show Each Keypoint and it's name
        
        for i, [x, y] in enumerate(keypointPositions):
            plt.figure(i)
            plt.title('keypoint' + str(i) + KEYPOINT_NAMES[i])
        #    img = plt.imread(imagePath)
            implot = plt.imshow(img)

            plt.scatter(x=[x], y=[y], c='r', s=40)
            plt.show()

        # SHOW CONNECTED KEYPOINTS
        plt.figure(20)
        for pt1, pt2 in CONNECTED_KEYPOINT_INDICES:
            plt.title('connection points')
            implot = plt.imshow(img)
            plt.plot((keypointPositions[pt1][0], keypointPositions[pt2][0]), (
                keypointPositions[pt1][1], keypointPositions[pt2][1]), 'ro-', linewidth=2, markersize=5)
        plt.show()



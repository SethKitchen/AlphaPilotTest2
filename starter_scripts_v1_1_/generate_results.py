# This script is to be filled by the team members.
# Import necessary libraries
# Load libraries
import json
import numpy as np
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from scipy import ndimage
import cv2
import math

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs","gate20190304T2306","mask_rcnn_gate_0030.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score. 

class gateConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "gate"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + gate

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(gateConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class GenerateFinalDetections():
    def __init__(self):
        self.config = InferenceConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                  model_dir=DEFAULT_LOGS_DIR)
        # Find last trained weights
        self.weights_path = WEIGHTS_PATH
        self.model.load_weights(self.weights_path, by_name=True)
        #np.set_printoptions(threshold=np.inf)


    def predict(self,img,imgname):
        # Read image
        image = img
        height,width,channels=img.shape
        # Detect objects
        r = self.model.detect([image], verbose=0)[0]
        mask=r['masks']
        x1=0
        x2=0
        x3=0
        x4=0
        y1=0
        y2=0
        y3=0
        y4=0
        minDistanceTopLeft=9999999
        minDistanceTopRight=9999999
        minDistanceBottomLeft=9999999
        minDistanceBottomRight=9999999
        for x in range(0, len(mask)):
            for y in range(0, len(mask[x])):
                try:
                    if(mask[x][y][0]):
                        distToTopLeft=(x-0)*(x-0)+(y-0)*(y-0)
                        if(distToTopLeft<minDistanceTopLeft):
                            minDistanceTopLeft=distToTopLeft
                            x1=x
                            y1=y
                        distToTopRight=(x-width)*(x-width)+(y-0)*(y-0)
                        if(distToTopRight<minDistanceTopRight):
                            minDistanceTopRight=distToTopRight
                            x2=x
                            y2=y
                        distToBottomLeft=(x-0)*(x-0)+(y-height)*(y-height)
                        if(distToBottomLeft<minDistanceBottomLeft):
                            minDistanceBottomLeft=distToBottomLeft
                            x4=x
                            y4=y
                        distToBottomRight=(x-width)*(x-width)+(y-height)*(y-height)
                        if(distToBottomRight<minDistanceBottomRight):
                            minDistanceBottomRight=distToBottomRight
                            x3=x
                            y3=y
                except:
                  pass
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, [255,0,0], gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        splash[x1][y1]=[0,255,0]
        splash[x2][y2]=[0,255,0]
        splash[x3][y3]=[0,255,0]
        splash[x4][y4]=[0,255,0]
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        toReturn=np.array([x1, y1, x2, y2, x3, y3, x4, y4, 1])
        return [toReturn.tolist()]


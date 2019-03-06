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
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs","gate20190305T0706","mask_rcnn_gate_0100.h5")

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
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                  model_dir=DEFAULT_LOGS_DIR)
        # Find last trained weights
        self.weights_path = WEIGHTS_PATH
        self.model.load_weights(self.weights_path, by_name=True)


    def predict(self,img):
        try:
            height,width,channels=img.shape
            # Detect objects
            r = self.model.detect([img], verbose=0)[0]
            mask=r['masks']
            x1=0
            x2=0
            x3=0
            x4=0
            y1=0
            y2=0
            y3=0
            y4=0
            # Copy color pixels from the original color image where mask is set
            if mask.shape[-1] > 0:
                # We're treating all instances as one, so collapse the mask into one layer
                mask = (np.sum(mask, -1, keepdims=True) >= 1)
                splash = np.where(mask, [255,0,0], img).astype(np.uint8)
            else:
                splash = img.astype(np.uint8)
            # threshold image
            thresh = cv2.inRange(splash, np.array([255,0,0]), np.array([255,0,0]))
            # dilate thresholded image - merges top/bottom 
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            # find contours
            contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # simplify contours
            if len(contours)>0:
                epsilon = 0.1*cv2.arcLength(contours[0],True)
                approx = cv2.approxPolyDP(contours[0],epsilon,True)
                cv2.drawContours(img, [approx], 0, (255,255,255), 3)
                if len(approx)==4:
                    minDistanceTopLeft=9999999
                    minDistanceTopRight=9999999
                    minDistanceBottomLeft=9999999
                    minDistanceBottomRight=9999999
                    for i in range(0, len(approx)):
                        x=approx[i][0][0]
                        y=approx[i][0][1]
                        distToTopLeft=x*x+y*y
                        if(distToTopLeft<minDistanceTopLeft):
                            minDistanceTopLeft=distToTopLeft
                            x1=x
                            y1=y
                        distToTopRight=(x-width)*(x-width)+y*y
                        if(distToTopRight<minDistanceTopRight):
                            minDistanceTopRight=distToTopRight
                            x2=x
                            y2=y
                        distToBottomLeft=x*x+(y-height)*(y-height)
                        if(distToBottomLeft<minDistanceBottomLeft):
                            minDistanceBottomLeft=distToBottomLeft
                            x4=x
                            y4=y
                        distToBottomRight=(x-width)*(x-width)+(y-height)*(y-height)
                        if(distToBottomRight<minDistanceBottomRight):
                            minDistanceBottomRight=distToBottomRight
                            x3=x
                            y3=y
                    toReturn=np.array([x1, y1, x2, y2, x3, y3, x4, y4, 1])
                    return [toReturn.tolist()]
            minDistanceTopLeft=9999999
            minDistanceTopRight=9999999
            minDistanceBottomLeft=9999999
            minDistanceBottomRight=9999999
            firstRange=0
            secondRange=0
            try:
                firstRange=range(0, len(mask))
                secondRange=range(0, len(mask[0]))
            except:
                pass
            for x in firstRange:
                for y in secondRange:
                    try:
                        if(mask[x][y][0]):
                            distToTopLeft=x*x+y*y
                            if(distToTopLeft<minDistanceTopLeft):
                                minDistanceTopLeft=distToTopLeft
                                x1=x
                                y1=y
                            distToTopRight=(x-width)*(x-width)+y*y
                            if(distToTopRight<minDistanceTopRight):
                                minDistanceTopRight=distToTopRight
                                x2=x
                                y2=y
                            distToBottomLeft=x*x+(y-height)*(y-height)
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
            if x1==0:
                if x2==0:
                    return [[]]                        
            toReturn=np.array([y1, x1, y4, x4, y3, x3, y2, x2, 1])
            return [toReturn.tolist()]
        except:
            return [[]]
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
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs","gate20190227T0051","mask_rcnn_gate_0007.h5")

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
        
def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr

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

        
    def predict(self,img):
        # Read image
        image = cv2.imread(os.path.join(ROOT_DIR, "Data_LeaderboardTesting", img))
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        mask=r['masks']
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2RGB)*255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, [248,24,148], [0,0,0]).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)

        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave(file_name, splash)
        # read image
        #img = cv2.imread(file_name)

        # thresholding
        thresh=cv2.inRange(splash, (5, 5, 5), (255, 255, 255))
        #cv2.imwrite("thresh.png", thresh)

        gray = np.float32(thresh)
        dst = cv2.cornerHarris(gray,5,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        x1=0
        x2=0
        x3=0
        x4=0
        y1=0
        y2=0
        y3=0
        y4=0
        xAverage=0.0
        yAverage=0.0
        if len(corners) > 4:
            for i in range(1, len(corners)):
                xAverage+=corners[i][0]
                yAverage+=corners[i][1]
            xAverage=xAverage/(len(corners)-1)
            yAverage=yAverage/(len(corners)-1)
            for i in range(1, len(corners)):
                if corners[i][0] < xAverage and corners[i][1]<yAverage and x1==0.0 and y1==0.0:
                    x1=corners[i][0]
                    y1=corners[i][1]
                elif corners[i][0] < xAverage and corners[i][1]>yAverage:
                    x4=corners[i][0]
                    y4=corners[i][1]
                elif corners[i][0] > xAverage and corners[i][1]<yAverage:
                    x2=corners[i][0]
                    y2=corners[i][1]
                elif corners[i][0] > xAverage and corners[i][1]>yAverage and x3==0.0 and y3==0.0:
                    x3=corners[i][0]
                    y3=corners[i][1]    
        print(x1,y1,x2,y2,x3,y3,x4,y4)            
        #img[dst>0.1*dst.max()]=[0,255,0]
        #cv2.imwrite('Corners.png', img)
        toReturn=np.array([x1, y1, x2, y2, x3, y3, x4, y4, 1])
        return [toReturn.tolist()]
        

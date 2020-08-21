import os
import tensorflow as tf

import math
import numpy as np
import skimage.io


import sys  
sys.path.insert(0, 'C:/Users/Saumya Sah/Mask_RCNN/build/lib')

import mrcnn
# Root directory of the project
ROOT_DIR = os.path.abspath("Saumya Sah/")

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.insert(0,'C:/Users/Saumya Sah/Mask_RCNN/samples/coco')  # To find local version
import coco
global graph
graph = tf.get_default_graph()


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = ('C:/Users/Saumya Sah/Downloads/mask_rcnn_coco.h5')

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=COCO_MODEL_PATH, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
#model.keras_model._make_predict_function()

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



#model = modellib.MaskRCNN(mode="inference", model_dir='./', config=TestConfig())
#model.load_weights('C:/Users/Saumya Sah/Downloads/mask_rcnn_coco.h5', by_name=True)
#model.keras_model._make_predict_function()


import numpy as np

def processing_image_m(path_to_image):
    image = skimage.io.imread(path_to_image)
    #img = load_img(path_to_image)
    #img = img_to_array(img)
    
    with graph.as_default():
        results = model.detect([image], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    for i in range(r['class_ids'].shape[0]):
        if (r['class_ids'][i]==1): 
            h=i
            break
    mask=r['masks'][:,:,h]
    box= r['rois'][h]
    [y1,x1,y2,x2]=box
    th=y2-y1
    ychest= int(0.26*th)
    ybutt= int(0.4*th)
    mask=mask[y1:y2,x1:x2]
    a=np.zeros((y2-y1))
    for i in range(y2-y1):
        for j in range(x2-x1):
            if(mask[i,j]): a[i]=a[i]+1
    max1=0
    min1=mask.shape[1]
    imin1=0
    imax1=0
    mask1= mask[ychest:ybutt,:]
    a1=a[ychest:ybutt]
    for i in range(ybutt-ychest):
        if a1[i]>max1:
            max1=a1[i]
            imax1=i
        if a1[i]<min1:
            min1=a1[i]
            imin1=i
    imax1=imax1+ychest+y1
    imin1=imin1+ychest+y1
    param = [max1,min1,imax1,imin1]
    min_mxbymn_1 = 1.1
    min_mxbyth_1 = 0.116
    max_mxbyth_2 = 0.13867
    max_mxbyth_3 = 0.169928
    max_mxbyth_4 = 0.206721
    
    # for case 1
    if imax1>imin1:
        if (max1/min1)>=min_mxbymn_1:
            if(max1/th)>=min_mxbymn_1: return 1
    #for case 2
    if (max1/th)<= max_mxbyth_2:
        return 2
    #case 3
    if ((max1/min1)<1.08)|((max1/min1)>0.92):
        if (max1/th)<= max_mxbyth_3 : return 3
    if (imax1<imin1)&(max1/th<=max_mxbyth_3) : return 3
    r=max1/th
    if(r<=max_mxbyth_4) : return 4
    if(r>max_mxbyth_4) : return 5
print(processing_image_m("C:/Users/Saumya Sah/Desktop/leanguysideview.jpg"))
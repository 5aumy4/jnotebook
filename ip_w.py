import sys  
sys.path.insert(0, 'C:/Users/Saumya Sah/Mask_RCNN/build/lib')

import mrcnn
import tensorflow as tf
import numpy as np
#print(tf.__version__)

# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# define 81 classes that the coco model knowns about
class_names =         ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

               

               
               
path = 'C:/Users/Saumya Sah/Desktop/Album 2/SquarePic_20200821_00351647.jpg'
# define the test configuration
class TestConfig(Config):
    
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('C:/Users/Saumya Sah/Downloads/mask_rcnn_coco.h5', by_name=True)

def processing_image_w(path):
    img = load_img(path)
    img = img_to_array(img)
# make prediction
    results = rcnn.detect([img], verbose=0)
# get dictionary for first prediction
    r = results[0]
# show photo with bounding boxes, masks, class labels and scores
    #display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    for i in range(r['class_ids'].shape[0]):
        if (r['class_ids'][i]==1): 
            h=i
            break
    mask=r['masks'][:,:,h]
    box= r['rois'][h]
    [y1,x1,y2,x2]=box
    th=y2-y1+40
    r['scores'][0]

    ybutt_w= int(0.58*th)
    ystomach_w= int(0.426*th)
    ybrst_w= int(0.31*th)
    yneck_w = int(0.17*th)
    
    mask=mask[y1:y2,x1:x2]
    a=np.zeros((y2-y1))
    for i in range(y2-y1):
        for j in range(x2-x1):
            if(mask[i,j]): a[i]=a[i]+1
           
    
    m1 = np.max(a[yneck_w:ybrst_w])
    m2= np.max(a[ybrst_w: ystomach_w])
    m3 = np.max(a[ystomach_w:ybutt_w])

    m = (m1+m2+m3)/th
     
    if m< 0.43456 :
        return 1
    elif m< 0.47578 :
        return 2
    elif m< 0.580379 :
        return 3
    elif m< 0.72965 :
        return 4
    else: return 5
print(processing_image_w('C:/Users/Saumya Sah/Desktop/Album 2/SquarePic_20200821_00223449.jpg'))
    
    
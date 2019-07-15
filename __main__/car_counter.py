from matplotlib import pyplot as plt 
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import numpy as np
import cv2
import dlib
import imutils

def counting_vehicle(input_path, output_path, model_cfg, name_list
                    model_weight, skip_frame, thresh_prop):
    '''
    Parameters:
        + video_path: 
            . None: take frame from camera
            . path: take frame from video
        + output_path: 
            . None: show the result
            . path: save into new video
        + name_list: list labels
        + model_cfg: model config 
        + model_weight: model weight
        + skip_frame: number of frame that system take sample 1 time
        + thresh_prop: the minimum score (propability) accepted to recognize an object 
    Output: None
    '''

    if input_path == None:
        print("Starting streaming")
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)

    else:
        print("Opening " + input_path)
        cap = cv2.VideoCapture(input_path)

    if output_path != None:
        writer = None
    config = model_cfg
    weight = model_weight
    name   = name_list

    with open(name, 'r') as f:
        # generate all classes of COCO, bicycle idx = 1, car idx = 2 and motorbike idx = 3
        classes = [line.strip() for line in f.readlines()] 
    np.random.seed(11)
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # Read the model
    net = cv2.dnn.readNet(weight, config)
    # take shape of image in order to scale it to 416x416, first layer of Yolo CNN

    scale = 0.00392

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=20, maxDistance=30)
    # trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0

    while True:
        
        ret,frame = cap.read()
        # Resize width to 500 and scale height propability equal  
        # 500/old_width (because the smaller pics the faster our model run) 
        frame = imutils.resize(frame, width = 500)
        # convert to rgb for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if W is None or H is None:
            # take original for scale the output of YOLO
            (H,W) = frame.shape[:2]
        status = "Waiting"
        rects = []
    # release memory
    if totalFrames == skip_frame**3:
        totalFrames = 1
    if totalFrames % skip_frame == 0:
        status = "Detecting"
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0,0,0), True, crop =False)
        net.setInput(blob)
        outs = net.foward(get_output_layers(net))

        for out in outs:
            




        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
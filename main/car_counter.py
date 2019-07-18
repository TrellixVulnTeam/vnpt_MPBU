from matplotlib import pyplot as plt 
from main.centroidtracker import CentroidTracker
from main.trackableobject import TrackableObject
import numpy as np
import cv2
import dlib
import imutils

def get_output_layers(net):
    '''
    get all output layer names: with yolov3 is yolo_82, 94 and 106
    ''' 
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def counting_vehicle(input_path, output_path, model_cfg, name_list,
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
        + horizon = the number decide line system counting on
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

    ct = CentroidTracker(maxDisappeared=10, maxDistance=30)
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
            outs = net.forward(get_output_layers(net))

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    # get the highest score to determine its label
                    class_id = np.argmax(scores)
                    # make sure we only choose vehicle
                    if class_id not in [0,1,2,3,7]:
                        continue
                    else:
                        # score of that object, make sure more than 50% correct label
                        confidence = scores[class_id]
                        if confidence > thresh_prop:
                            center_x = int(detection[0] * W)
                            center_y = int(detection[1] * H)
                            w = int(detection[2] * W)
                            h = int(detection[3] * H)
                            # remember it return x_center and y_center, not x,y, so we need to find x,y
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                        elif confidence <= thresh_prop:
                            continue
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(x, y, x+w, y+h)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))
        cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 255), 2)
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]       
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if direction > 0.001:
                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True
                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True
            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
        totalFrames += 1
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break
    cap.release()
    cv2.destroyAllWindows()
counting_vehicle(input_path="C:/Users/TobyCurtis/Desktop/front.avi",output_path=None, name_list="coco.names",
                model_cfg="YOLOv3-320/yolov3.cfg",model_weight="YOLOv3-320/yolov3.weights", 
                skip_frame=15, thresh_prop= 0.3)
import cv2
import numpy as np
import time
import os
from client.Client import *
#os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
#import cv2 as cv
#load Yolo model
net = cv2.dnn.readNet("yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

print(0)

client = CClient('XXX.XXX.X.XX', 1500)
client.EXT_ConnectToServer()

#cap = cv2.VideoCapture(0) #insert 0 if using PC camera
#cap = cv2.VideoCapture('GOPR0295.MP4')

#cap.set(3, 1500)
#cap.set(4, 1500)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

#measure_time_start, measure_time_end
#while(cap.isOpened()):
#   ret, frame = cap.read()
##   if ret:
#      assert not isinstance(frame, type(None)), 'frame not found'
while True:
    show = client.EXT_GetImg(False)
    show = show[170:790, 190:900]
    has_frame = True

    #   _, frame = show.read()
    if has_frame:
      measure_time_start = time.time()
      frame_id += 1
      height, width, channels = show.shape
    # Detecting objects
      blob = cv2.dnn.blobFromImage(show, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #extract 416x416 images
      net.setInput(blob)
      outs = net.forward(output_layers)
    # Showing informations on the screen
      class_ids = []
      confidences = []
      boxes = []
      for out in outs:
         for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) #gives max value in the scores array in a particular axis
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
      indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3) #to remove multiple boxes for same object
      for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(show, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(show, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(show, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
            if label == 'person':
                cv.putText(show, 'WARNING: finding people..', (20, 90), cv.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 255), 2)

      measure_time_end = time.time()

      elapsed_time = time.time() - starting_time
      fps = frame_id / elapsed_time
      cv2.putText(show, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
      #cv2.resizeWindow("camera", 1800, 1800)
      cv2.imshow("first step differernt scenarios  find people", show)
      measure_detection = measure_time_end - measure_time_start
      print(measure_detection)
      key = cv2.waitKey(1)
      if key == 27:
         break
# cap.release()
#cv2.destroyAllWindows()

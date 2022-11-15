# -*- coding: UTF-8 -*-
import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
import glob
import sys
from client.Client import *

parser = argparse.ArgumentParser(description='Action Recognition by Open-Pose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# load models:  mobilenet_thin (trained in 432x368)
estimator = load_pretrain_model('mobilenet_thin')
#action recognition
action_classifier = load_action_premodel('model.h5')

# parameters initialized
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0


#cap = choose_run_mode(args)
#video_writer = set_video_writer(cap, write_fps=int(7.0))

# # save data to txt file for training
#f= open('data.txt', 'a+')

print(0)

client = CClient('XXX.XXX.X.XX', 1500)
client.EXT_ConnectToServer()
show = client.EXT_GetImg(False)

show = client.EXT_GetImg(False)
#video_writer = set_video_writer(show, write_fps=int(7.0))
#f = open('1103lcdatacancel.txt', 'a+')

while cv.waitKey(1) < 0:
   # has_frame, show = cap.read()
    show = client.EXT_GetImg(False)
    #show = show[y:y+h, x:x+w]

    show = show[170:790, 190:900]

#cv2.imshow("cropped", crop_img)

    has_frame = True
    if has_frame:
        measure_time_start = time.time()

        fps_count += 1
        frame_count += 1
        # pose estimation
        humans = estimator.inference(show)
        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier)


        height, width = show.shape[:2]
        # show FPS
        if (time.time() - start_time) > fps_interval:
          
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width - 160, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # show the num of human
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height - 45), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # determine the group
        if len(humans) == 0:
            label2 = "Nobody"
            cv.putText(show, label2, (width - 360, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif len(humans) == 1:
            label2 = "Individual"
            cv.putText(show, label2, (width -360, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            label = "Multiple"
            cv.putText(show, label, (width - 360, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        measure_time_end = time.time()
        measure_detection = measure_time_end - measure_time_start
        print(measure_detection)

        # Shows the current runtime and total frame rate
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow('Human Action Recognition', show)

        #video_writer.write(show)
        #joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        #f.write(' '.join(joints_norm_per_frame))
        #f.write('\n')


#video_writer.release()
#f.close()




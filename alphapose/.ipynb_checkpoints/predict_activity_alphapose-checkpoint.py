#!/usr/bin/env python
# coding: utf-8

# In[8]:


import time

import argparse
import cv2
import glob
import json
import numpy as np
import os
import subprocess
from keras.models import model_from_json
from pprint import pprint
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ## Implementation

# This notebook predicts activities in a video using AlphaPose,
# YOLOv4-tiny and created deep learning model.
# Note that you need to arrange below paths for your system / directory.

# In[2]:


# 1. Feed input video to AlphaPose
# 1.1. Save video frame JSON outputs
# 2. Combine JSON outputs and convert to txt file (preprocessing)
# 3. create blocks (20 frames per block) from TXT file to feed the LSTM model
# 4. feed blocks to the final LSTM model
# 5. Visualize (annotate) predicted class on video with OpenCV as well as YOLOv4 + DeepSORT

# In[3]:

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video file")
args = vars(ap.parse_args())


# In[4]:

alphapose_path = "/home/enes/alpha_conda/AlphaPose"
bpose_path = "/home/enes/Downloads/BabyPose/"
root_path = "/home/enes/Downloads/BabyPose/alphapose"
data_path = "/home/enes/Downloads/BabyPose/alphapose/json_output_alphapose"
yolo_path = "/home/enes/Downloads/BabyPose/yolov4-deepsort"

#input_video = 'demo\\baby_stretching.mp4'
input_video = args["input"] if args["input"] else 0

print('input_vid: ', input_video)

# #### Method to read JSON files

# In[5]:

def read_json_file(json_file):    
    os.chdir(data_path + '/sep-json')

    print("Reading JSON: " + str(json_file))
    with open(json_file) as data_file:
        data = json.load(data_file)

        frame_kps = []

        # holds x and y values of the current JSON frame.
        try:
            pose_keypoints = data['people'][0]['pose_keypoints_2d']
            # print(data['people'][0]['pose_keypoints_2d'])
        except:
            print("JSON file cannot read: " + file)

        # print(data['people'][0]['pose_keypoints_2d'])
        # loop through 25 pose keypoints (total = 75, 25x3 (x, y and accuracy))
        # we will loop through for 19 keypoints - ignoring foots.
        # total = 19*2 = 38
        j = 0
        for i in range(36):  # 50
            frame_kps.append(pose_keypoints[j])
            j += 1
            if ((j+1) % 3 == 0):
                j += 1

        return frame_kps


# #### Method to write combined JSON files

# In[6]:

# Now we have kps, a list of lists, that includes the x and y positions of all 19 keypoints, for all frames in the video.
# So a list of length frameset.length, with each element being a 38 element long list.

def write_frames_txt(kps):
    os.chdir(data_path)

    # create filename for combined frames
    output_file = str("all_combined_jsons.txt")

    with open(output_file, "w") as text_file:
        for i in range(len(kps)):
            for j in range(36): 
                text_file.write('{}'.format(kps[i][j]))
                if j < 35: 
                    text_file.write(',')
            text_file.write('\n')


# #### Method to clear OpenPose JSON output directory

# In[7]:


def clear_json_outputs():
    os.chdir(data_path)

    files = glob.glob('*')
    for f in files:
        os.remove(f)


# #### Method for getting string labels for predictions

# In[8]:


def get_label_from_model(predicted_label):
    if predicted_label == 0:
        return "arching_back"

    if predicted_label == 1:
        return "head_banging"

    if predicted_label == 2:
        return "kicking_legs"

    if predicted_label == 3:
        return "rubbing_eye"

    if predicted_label == 4:
        return "stretching"

    if predicted_label == 5:
        return "sucking_fingers"
    
def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    return int(int_part)


# ### 1. Feed input video to openpose

# In[9]:


# 0. clear openpose_json_output folder for new JSON files.
clear_json_outputs()

# 1. we are giving kicking_legs1_short2.mp4 file for testing.
#os.chdir(bpose_path)

## print(Path(bpose_path + input_video).name)

#input_file_path = Path(bpose_path + input_video).name

print("Rendering input video with AlphaPose...")

os.chdir(alphapose_path)

print('goster: ', bpose_path + input_video)

cmd_string = ('python video_demo.py'
              ' --video ' + bpose_path + input_video +
              ' --outdir  ' + data_path +
              ' --sp'
              ' --vis_fast'
              ' --format open' 
              ' --fast_inference False'
              ' --save_video'
              )

# 1.a. JSON files created in alphapose\\json_output_alphapose
# subprocess.Popen(cmd_string)

subprocess.call(cmd_string, shell=True)
print("Input video rendered. All JSON files saved.")


# ### 2. Combine JSON outputs for input video

# In[10]:

# 2. Now combine those JSON outputs.
os.chdir(data_path + '/sep-json')


# kps is a list of pose keypoints in each frame,
# where kps[0] is the x position of kp0, kps[1] is the y position of kp0 etc
kps = []

json_file_count = len([name for name in os.listdir('.') if os.path.isfile(name)])


#sorted(glob.glob(f'{os.getcwd()}/*.txt'), key=len)
#filelist = sorted(glob.glob("*.json"))
filelist = sorted(glob.glob(f'{os.getcwd()}/*.json'), key=get_key)
for json_file in filelist:
    current_frame_json = read_json_file(json_file)

    # append frame to frame list of current video 
    kps.append(current_frame_json)

# Trying to adjust block sizes (20)
# json_file_count mod 20 = A:
mod = json_file_count % 20

# too many frames to be ignored.
# add last frame to kps list n-times (mod) to complete block size.
if mod > 5:
    #print('mod > 5 . frame to be added.')
    frames_to_be_added = 20 - mod
    
    # get last frame from kps
    kps_last_frame = kps[-1]
    
    for _ in range(frames_to_be_added):
        #print('appending last frame...')
        kps.append(kps_last_frame)

else:
    # just deleting < 5 frame info to match block size.
    del kps[-mod:]

# write combined JSON files to TXT file.
write_frames_txt(kps)

# ### 3. Create blocks for LSTM model input (3-dimensional)

# In[11]:

# 3. create blocks (20 frames per block) from TXT file to feed the LSTM model

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()

    return X_


X_pred = load_X(data_path + '/all_combined_jsons.txt')

# normalize X_pred values before feeding them to LSTM network.
scaler = MinMaxScaler().fit(X_pred)
X_pred = scaler.transform(X_pred)

# In[12]:


print ('\nProcessed frame count: ', len(kps))


# In[13]:


# creating blocks
blocks = int(len(X_pred) / 20)
X_pred = np.array(np.split(X_pred, blocks))
X_pred.shape


# In[14]:


# reshape x_pred with (block_number, 20, 38)


# ### 4. Feed blocks to the final LSTM model
# #### 4.1. Load final LSTM model

# In[15]:


# os.chdir(root_path)
# tf.keras.backend.clear_session()
#model = tf.keras.models.load_model('best_model_new', compile=False)


# In[16]:


# Loading model from JSON - much more faster loading time.
os.chdir(root_path + "/best_model_alphapose")

with open("alphapose_best_model.json", 'r') as json_file:
    loaded_json = json_file.read()

model = model_from_json(loaded_json)
model.load_weights("alphapose_best_weights.h5")


# In[17]:


# model.summary()


# In[18]:


#model.predict_classes(X_pred)


# In[19]:


#np.argmax(model.predict(X_pred), axis=-1)


# In[20]:


pred_array = np.argmax(model.predict(X_pred), axis=-1)


# In[21]:


#(model.predict(X_pred) > 0.5).astype("int32")


# ### Visualize (annotate) predicted class on video with OpenCV

# In[23]:


os.chdir(bpose_path)

cap = cv2.VideoCapture(bpose_path + input_video)

label_counter = 0
predicted_label = ""

# Define the codec and create VideoWriter object
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_yolo/output.avi', fourcc, fps, (width, height))

while True:
    success, frame = cap.read()
    
    #out = cv2.VideoWriter('output_video/output.avi', codec, fps, (width, height))

    if frame is None:
        break
        
    

    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 20 == 0 & label_counter < blocks:
        predicted_label = get_label_from_model(pred_array[label_counter])
        label_counter += 1

        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, predicted_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, predicted_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    out.write(frame)
    # cv2.imshow("BabyPose", frame)
    
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cap.release()
time.sleep(3)
cv2.destroyAllWindows()

print ('\nRunning YOLOv4 with DeepSORT on output video...\n')

os.chdir(yolo_path)

subprocess.call('python object_tracker_.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ../output_yolo/output.avi --tiny', shell=True)

# In[ ]:





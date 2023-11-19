import cv2
import xarm
import numpy as np
import time
from utils import VideoCapture, XarmIK
import os
from datetime import datetime

current_date = datetime.now().strftime("%Y-%m-%d")
dataset_dir = f'saved_data/{current_date}'

os.makedirs(dataset_dir, exist_ok=True)




arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, 'xarm.URDF')  
cap = VideoCapture(3)

control_frequency = 15 # Hz
t0 = time.time()
next_time = t0 + 1/control_frequency
frames = []
states = []
actions = []
positions = []
start_time = time.time()

while(True):
    frame = cap.read()
    cv2.imshow('frame',frame)
    t = time.time()

    if t >= next_time:
        location_3d = xarm_ik.get_location()
        # position = xarm_ik.get_positions()
        gripper_open = xarm_ik.gripper_open_check()
        print("gripper_open: ", gripper_open)
        location_3d = location_3d + [gripper_open]
        frames.append(frame)
        states.append(location_3d)
        next_time += 1/control_frequency

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('==================')
print('==================')
print('HZ: ', len(states)/(time.time()-start_time))

yn = input('wanna save:?')

des_num = 100
if len(states) < des_num:
    states = states + (des_num-len(states)) * [states[-1]]
    frames = frames + (des_num-len(frames)) * [frames[-1]]


for i in range(1, len(states)):
    # converting to cm
    actions.append((states[i] - states[i-1])*100)
frames = frames[:-1]

if yn == 'y':
    print('saving...')
    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/actions_{filename}.npy', actions)
    np.save(f'{dataset_dir}/frames_{filename}.npy', frames)
    np.save(f'{dataset_dir}/positions_{filename}.npy', positions)

cap.stop()  # Stop the video capture thread
cv2.destroyAllWindows()
import cv2
import xarm
import numpy as np
import time
from utils import VideoCapture, XarmIK, Controller
import os
from datetime import datetime
import random


current_date = datetime.now().strftime("%Y-%m-%d")
dataset_dir = f'saved_data/{current_date}'
os.makedirs(dataset_dir, exist_ok=True)


def read_times(cap, duration):
    start_time = time.time()

    while True:
        frame = cap.read()
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            break  # Break the loop after 0.2 seconds



arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, 'xarm.URDF')  
cap = VideoCapture(2)


# original: [-0.1519, 0.0000, 0.03]
# very close: [-0.07, -0.0001, 0.04]
# very far: [-0.20, -0.0001, 0.04]
# right [-0.14, -0.08, 0.04]
# left [-0.13, 0.08, 0.03]
controller = Controller([-0.13, 0.08, 0.03]) 


#x 0.05 -0.2
#y -0.1 0.1
#z 0.1 0.25

in_x = random.uniform(-0.2, 0.05)
in_y = random.uniform(-0.1, 0.1)
in_z = random.uniform(0.1, 0.25)

print(in_x, in_y, in_z)
xarm_ik.set_location([in_x, in_y, in_z], duration=3000)


frames = []
states = []
actions = []
gripper_state = 1
time.sleep(3)


while(True):
    frame = cap.read()
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    current_location = xarm_ik.get_location()
    action = list(current_location) + [gripper_state]
    # print(current_location, gripper_state, action)
    # print('====================')
    frames.append(frame)
    states.append(action)

    direction_v, gripper_state = controller.act(current_location)
    if not direction_v:
        break
    next_position = list(np.array(current_location)+np.array(direction_v))
    xarm_ik.set_location(next_position, gripper_state)

print(len(frames), len(states))


yn = input('wanna save:?')

des_num = 50
if len(states) < des_num:
    states = states + (des_num-len(states)) * [states[-1]]
    frames = frames + (des_num-len(frames)) * [frames[-1]]


for i in range(1, len(states)):
    action_3d = list(np.array(states[i][:3]) - np.array(states[i-1][:3]))
    gripper = states[i][-1]
    final_action = action_3d + [gripper]
    actions.append(final_action)

frames = frames[:-1]

if yn == 'y':
    print('saving...')
    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/actions_{filename}.npy', actions)
    np.save(f'{dataset_dir}/frames_{filename}.npy', frames)
    np.save(f'{dataset_dir}/states_{filename}.npy', states)

cap.stop()  # Stop the video capture thread
cv2.destroyAllWindows()
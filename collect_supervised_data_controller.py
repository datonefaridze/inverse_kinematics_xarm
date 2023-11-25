import cv2
import xarm
import numpy as np
import time
from utils import VideoCapture, XarmIK, Controller
import os
from datetime import datetime

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

# cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


controller = Controller([-0.1519, 0.0000, 0.03]) 
xarm_ik.set_location([-0.1182, -0.0001, 0.1917])


frames = []
states = []
actions = []
gripper_state = 1

while(True):
    frame = cap.read()
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    current_location = xarm_ik.get_location()
    action = current_location + [gripper_state]
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
    action_3d = states[i][:3] - states[i-1][:3]
    gripper = states[i][-1]
    actions.append([action_3d, gripper])
frames = frames[:-1]

if yn == 'y':
    print('saving...')
    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/actions_{filename}.npy', states)
    np.save(f'{dataset_dir}/frames_{filename}.npy', frames)
    np.save(f'{dataset_dir}/states_{filename}.npy', states)

cap.stop()  # Stop the video capture thread
cv2.destroyAllWindows()
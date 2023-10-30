import cv2
import numpy as np
import xarm
import time
max_episodes = 200

arm = xarm.Controller('USB',)

# cap0 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(2)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)



def get_positions(arm):
    positions = []
    for i in range(0, 6):
        position = arm.getPosition(i+1, True)
        positions.append(position)
    return positions
current_pos = get_positions(arm)


def frames(cap, n):
    for _ in range(n):
        cap.read()

actions_arr = []
states_arr = []
positions_arr = []
print("set up")
print('daiwyoooooooooooooo')
time.sleep(0.5)
print('nagdad daiwyo')
# frames(cap, 5)
avg = 0
n = 0

while(True):
    ret, frame = cap.read()
    start = time.time()
    print(frame.shape)
    if ret == True: 
        cv2.imshow('frame',frame)
        time.sleep(0.08)
        temp_pos = get_positions(arm)        
        action = np.array(temp_pos) - np.array(current_pos)
        if np.linalg.norm(action) >= 5:
            actions_arr.append(action)
            states_arr.append(frame)
            positions_arr.append(temp_pos)            
        current_pos = temp_pos
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break 

yn = input('wanna save:?')
dataset_dir = '/home/dato/src/my_projects/robotics/datasets/xarm/my_test_dataset'
if yn == 'y':
    print('saving...')
    import time
    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/actions_{filename}.npy', actions_arr)
    np.save(f'{dataset_dir}/states_{filename}.npy', states_arr)
    np.save(f'{dataset_dir}/positions_{filename}.npy', positions_arr)
cap.release()

cv2.destroyAllWindows()
#0.20008201468480777
#0.1973280131816864
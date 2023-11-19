import cv2
import xarm
import numpy as np
import time
from utils import VideoCapture, XarmIK
import os
from datetime import datetime
from multiprocessing import Process, Queue, Event

current_date = datetime.now().strftime("%Y-%m-%d")
dataset_dir = f'saved_data/{current_date}'

os.makedirs(dataset_dir, exist_ok=True)



def move_robot(xarm_ik, datapoints, end_event):
    """
    Function to move the robot to each datapoint.
    """
    for point in datapoints:
        xarm_ik.set_location(point)
    end_event.set()



def collect_data(cap, xarm_ik, data_queue, end_event):
    """
    Function to collect frames and datapoints.
    """
    control_frequency = 15  # Hz
    next_time = time.time() + 1 / control_frequency

    while not end_event.is_set():  # Check if moving process has ended
        frame = cap.read()
        location_3d = xarm_ik.get_location()
        gripper_open = xarm_ik.gripper_open_check()
        print("gripper_open: ", gripper_open)
        location_3d = location_3d + [gripper_open]
        data_queue.put((frame, location_3d))
        next_time += 1/control_frequency

    # data_queue.put((frames, states))  # Send any remaining data

def main():
    end_event = Event()
    datapoints = [[-0.1519, 0.0000, 0.0078], [-0.1182, -0.0001, 0.1917], [-0.0437, -0.1098, 0.1911]]
    arm = xarm.Controller('USB',)
    xarm_ik = XarmIK(arm, 'xarm.URDF')  
    cap = VideoCapture(3)

    # Create and start the processes
    data_queue = Queue()
    move_process = Process(target=move_robot, args=(xarm_ik, datapoints, end_event))
    data_process = Process(target=collect_data, args=(cap, xarm_ik, data_queue, end_event))

    move_process.start()
    time.sleep(0.5)  # Optional delay to ensure moving starts first
    data_process.start()

    # Wait for processes to complete
    move_process.join()
    data_process.join()

    collected_frames, collected_states = data_queue.get()

    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/actions_{filename}.npy', collected_frames)
    np.save(f'{dataset_dir}/frames_{filename}.npy', collected_states)

    cap.release()
    cv2.destroyAllWindows()
    # Process collected data
    # ...

if __name__ == "__main__":
    main()


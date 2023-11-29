# from uitls import XarmIK

import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import xarm
import time
import math
from utils import Controller, XarmIK, VideoCapture, ParabolicController
import cv2

np.set_printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format})
#p1 [-0.13, 0.08, 0.03]
#p2 [-0.13, -0.15, 0.03]

arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, 'xarm.URDF') 
loc = [-0.10132686,  0.04980654,  0.22125744]

print(xarm_ik.get_location())
xarm_ik.set_location(loc, duration=2000)
# controller = ParabolicController(loc, [-0.13, 0.08, 0.06], [-0.13, -0.15, 0.06]) 
# print(len(controller.sequence))

# while True:
#     cur_loc = xarm_ik.get_location()
#     dir_v, gripper = controller.act(cur_loc)
#     next_loc = np.array(cur_loc) + np.array(dir_v)
#     print(next_loc, dir_v, gripper)
#     xarm_ik.set_location(next_loc, gripper_action=gripper)
#     if not dir_v:
#         break


loaded_np = np.load('/home/dato/src/my_projects/robotics/inverse_kinematics_xarm/saved_data/2023-11-28/actions_1701207234580959.npy')
current_loc = xarm_ik.get_location()

for action in loaded_np:
    print(action)
    next_location = current_loc + action[:3]
    gripper = action[3]
    print('next loc', next_location, gripper)
    xarm_ik.set_location(next_location, gripper_action=gripper)
    current_loc = next_location

    # time.sleep(1)
# cv2.destroyAllWindows()
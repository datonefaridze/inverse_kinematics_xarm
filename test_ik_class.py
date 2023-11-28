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
loc = [-0.2154, -0.0028, 0.0701]

print(xarm_ik.get_location())
xarm_ik.set_location(loc, duration=2000)
controller = ParabolicController(loc, [-0.13, 0.08, 0.06], [-0.13, -0.15, 0.06]) 


while True:
    cur_loc = xarm_ik.get_location()
    dir_v, gripper = controller.act(cur_loc)
    next_loc = np.array(cur_loc) + np.array(dir_v)
    print(next_loc, dir_v, gripper)
    xarm_ik.set_location(next_loc, gripper_action=gripper)
    if not dir_v:
        break



# cv2.destroyAllWindows()
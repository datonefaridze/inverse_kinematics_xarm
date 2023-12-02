
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import numpy as np
import xarm
from utils import XarmIK

# np.set_printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format})

# arm = xarm.Controller('USB',)
# xarm_ik = XarmIK(arm, 'xarm.URDF') 
# print(xarm_ik.get_location())

# loc = [-0.1324, 0.0524, 0.04]
# print(xarm_ik.get_location())
# xarm_ik.set_location(loc, duration=2000)

# x 0, -0.24
# y -0.03 0.1


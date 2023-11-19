# from uitls import XarmIK

import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import xarm
import time
import math

np.set_printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format})

class XarmIK:
    def __init__(self, arm, chain_description):
        self.my_chain = ikpy.chain.Chain.from_urdf_file(chain_description)
        self.arm = arm
        self.open = -90.0
        self.close = 10.0
        self.gripper_open()

    def set_location(self, target_position):
        target_angles = self.my_chain.inverse_kinematics(target_position)
        # print("target_angles: ", target_angles)
        target_angles_degrees = np.array([math.degrees(radian) for radian in target_angles])
        target_angles_degrees =  np.flip(target_angles_degrees[1:-1])
        desired_pos = [[x+3, float(target_angles_degrees[x])] for x in range(len(target_angles_degrees))]
        self.arm.setPosition(desired_pos, duration=5000, wait=True)
        return True
        
    def get_location(self):
        ''''returns 3d positions of gripper in meters'''
        
        current_position = [float(math.radians(x)) for x in self.get_positions()][::-1]
        current_position = [0.0] + current_position + [0.0]
        coordinates_3d = self.my_chain.forward_kinematics(current_position)
        return coordinates_3d[:3, 3]
    
    def get_positions(self):
        """ Returns poisitions in degrees"""
        positions = []
        for i in range(2, 6):
            position = self.arm.getPosition(i+1, True)
            positions.append(position)
        return positions
    

    def gripper_open_check(self):
        position = self.arm.getPosition(1, True)
        if position <= -50:
            return 1
        else:
            return -1

    def gripper_close(self):
        position = self.arm.setPosition([[1, self.close]], duration=500, wait=True)
        return True

    def gripper_open(self):
        position = self.arm.setPosition([[1, self.open]], duration=500, wait=True)
        return True

arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, 'xarm.URDF')  
print(xarm_ik.get_location())


xarm_ik.set_location([-0.1519, 0.0000, 0.0078])
xarm_ik.gripper_close()


xarm_ik.set_location([-0.1182, -0.0001, 0.1917])
print(xarm_ik.get_location())

xarm_ik.set_location([-0.0437, -0.1098, 0.1911])
print(xarm_ik.get_location())



# -90.0 open
# 10.0 close
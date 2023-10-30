
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import xarm
import time
import math

arm = xarm.Controller('USB')

def move():
    # x, y, z
    np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:0.2f}'.format})

    target_position = [ 0.0, 0.0, 31.0]
    my_chain = ikpy.chain.Chain.from_urdf_file("xarm.URDF")
    target_angles = my_chain.inverse_kinematics(target_position)

    print("target_angles: ", target_angles)
    target_angles = np.insert(target_angles, 0, 0., axis=0)
    target_angles = np.array([math.degrees(radian) for radian in target_angles])
    print("The angles of each joints are : ", target_angles)


    # real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position[1:]))
    # print("Computed vector : %s, original position vector : %s" % (real_frame[:3, 3], target_position))

    desired_pos = [[x+1, float(target_angles[x])] for x in range(len(target_angles))]
    arm.setPosition(desired_pos, duration=1000, wait=True)

if __name__ == '__main__':
    move()

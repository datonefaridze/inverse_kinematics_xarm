
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import xarm
import time
import math

arm = xarm.Controller('USB')

def move():
    np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:0.2f}'.format})

    # x, y, z
    # ik ითვლის დალაგებულად: 6,5,4,3,2,1
    target_positions = [[-0.1, 0.1, 0.3]]
    for target_position in target_positions:
        my_chain = ikpy.chain.Chain.from_urdf_file("xarm.URDF")
        print('my_chain: ', my_chain)
        target_angles = my_chain.inverse_kinematics(target_position)
        print("target_angles: ", target_angles)


        target_angles_degrees = np.array([math.degrees(radian) for radian in target_angles])
        print("The angles of each joints are : ", target_angles_degrees)
        target_angles_degrees =  np.flip(target_angles_degrees[1:-1])
        print(target_angles_degrees)

        real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position))
        print("Computed vector : %s, original position vector : %s" % (real_frame[:3, 3], target_position))

        desired_pos = [[x+3, float(target_angles_degrees[x])] for x in range(len(target_angles_degrees))]
        print('dato')
        arm.setPosition(desired_pos, duration=5000, wait=True)
        print('dato')
if __name__ == '__main__':
    move()

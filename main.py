
import xarm
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils


def move():
    target_position = [ 0.1, -0.2, 0.1]
    my_chain = ikpy.chain.Chain.from_urdf_file("xarm.URDF")
    print("The angles of each joints are : ", my_chain.inverse_kinematics(target_position))

    real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position))
    print("Computed          vector : %s, original position vector : %s" % (real_frame[:3, 3], target_position))



if __name__ == '__main__':
    move()

# from uitls import XarmIK

import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import xarm
import time
import math
from utils import Controller, XarmIK, VideoCapture
import cv2

np.set_printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format})


arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, 'xarm.URDF') 
controller = Controller([-0.1519, 0.0000, 0.03]) 
xarm_ik.set_location([-0.1182, -0.0001, 0.1917])
print(xarm_ik.get_location())

array = []

# while True:
#     cur_loc = xarm_ik.get_location()
#     print('current loc', cur_loc)
#     array.append(cur_loc)
#     vec, gr = controller.act(xarm_ik.get_location())
#     if not vec:
#         break
#     final = list(np.array(xarm_ik.get_location())+np.array(vec))
#     xarm_ik.set_location(final, gr)


# np.save('travelled_distance.npy', np.array(array))


cap = VideoCapture(2)

loaded_array = np.load('travelled_distance.npy')    
for ar in loaded_array:
    xarm_ik.set_location(ar)
    frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(3)




cap.stop()  # Stop the video capture thread
cv2.destroyAllWindows()
import xarm
import time

arm = xarm.Controller('USB')
reset = [0.0, -88.0, -120.0, -2.5, -6.75, -7.5, -120.75]

for i in range(1, 7):
    arm.setPosition(i, float(reset[i]))
    # position = arm.getPosition(i, True)
    # print(i,position)
    print(i, float(reset[i]))


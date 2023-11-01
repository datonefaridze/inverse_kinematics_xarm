import xarm
import time

arm = xarm.Controller('USB')
reset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0]

for i in range(1, 7):
    position = arm.getPosition(i, True)
    print(f'position {i}', position)


import xarm
import time


arm = xarm.Controller('USB')

reset = [-90, 0.25, -87.25, 59.75, 40.5, 0.25]
reset_id = [[x+1, float(reset[x])] for x in range(len(reset))]
print(reset_id)

arm.setPosition(reset_id, duration=2000, wait=True)

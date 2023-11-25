from utils import *


controller = Controller([-0.1519, 0.0000, 0.03]) 
for i in range(100):
    print(controller.act([-0.1519, 0.0000, 0.03]))




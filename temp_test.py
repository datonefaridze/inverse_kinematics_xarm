import cv2, queue, threading, time
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import time
import math

class Controller:
    def __init__(self, location_point):
        self.location_point = location_point
        self.up_point = [-0.1182, -0.0001, 0.1917]
        self.up_right_point = [-0.06, -0.17, 0.1911]
        self.down_point = [-0.06, -0.17, 0.05]
        self.sequence = [self.location_point, self.up_point, self.up_right_point, self.down_point, self.up_right_point, self.up_point]
        self.gripper = [-1, -1, -1, -1, 1, 1]
        self.chasing_idx = 0
        self.step_size = 0.03
        self.gripper_open = 1

    def act(self, current_location):
        if self.chasing_idx >= len(self.sequence):
            return False, False
        dist = np.linalg.norm(current_location-self.sequence[self.chasing_idx])
        direction_v = self.sequence[self.chasing_idx]-current_location
        if dist <= self.step_size * 1.5:
            act = list(direction_v)
            self.chasing_idx += 1
        else:
            act = list((direction_v/np.linalg.norm(direction_v)) * self.step_size)
        dist_prev = np.linalg.norm(current_location-self.sequence[self.chasing_idx-1]) if self.chasing_idx > 0 else 1000
        if min(dist_prev, dist) <= 0.01:
            self.gripper_open = self.gripper[self.chasing_idx]
        return (act, self.gripper_open)


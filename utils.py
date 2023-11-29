
import cv2, queue, threading, time
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import time
import math


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    self.q = queue.Queue(maxsize=3)
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

  def stop(self):
      self.running = False
      self.cap.release()

class XarmIK:
    def __init__(self, arm, chain_description):
        self.my_chain = ikpy.chain.Chain.from_urdf_file(chain_description)
        self.arm = arm
        self.open = -90.0
        self.close = 10.0
        self.gripper_act(1)

    def set_location(self, target_position, gripper_action=False, duration=1000):
        target_angles = self.my_chain.inverse_kinematics(target_position)
        target_angles_degrees = np.array([math.degrees(radian) for radian in target_angles])
        target_angles_degrees =  np.flip(target_angles_degrees[1:-1])
        desired_pos = [[x+3, float(target_angles_degrees[x])] for x in range(len(target_angles_degrees))]
        self.arm.setPosition(desired_pos, duration=duration, wait=True)
        if gripper_action:
            self.gripper_act(gripper_action)
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
    
    def gripper_act(self, action):
        if action == 1:
            self.arm.setPosition([[1, self.open]], duration=1000, wait=True)
        else:
            self.arm.setPosition([[1, self.close]], duration=1000, wait=True)
    


class Controller:
    def __init__(self, location_point):
        self.location_point = location_point
        self.up_point = [-0.1182, -0.0001, 0.1917]
        self.up_right_point = [-0.06, -0.17, 0.1911]
        self.down_point = [-0.06, -0.24, 0.05]
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
        
        print('dist: ', dist)
        print('dist_prev: ', dist_prev) 
        if min(dist_prev, dist) <= 0.028:
            self.gripper_open = self.gripper[self.chasing_idx]
        return (act, self.gripper_open)
    

class ParabolicController:
    def __init__(self, current_location, p1, p2, h=0.2, t=0.1):
        # 1 ღიაა
        # -1 დახურულია

        self.current_location = current_location
        self.p1 = p1
        self.p2 = p2
        assert p1[-1]==p2[-1]
        self.Z_coordinate = p1[-1]
        self.h = h
        self.t = t
        self.sequence, self.gripper = self.get_sequence()
        print(self.sequence)
        self.chasing_idx = 0
        self.gripper_open = 1

    def get_sequence(self):
        stop_height = 0.1
        sequence = []
        gripper = []
        for x in np.arange(0, 1, self.t):
            sequence.append(self.cl_pt(self.current_location, self.p1, x, h=self.h/2))
            gripper.append(1)
            if x + self.t * 1.5 >= 1:
                break
        
        for x in np.arange(0, 1+self.t, self.t):
            sequence.append(self.cl_pt(self.p1, self.p2, x))
            if x >= 1:
                gripper.append(1)
            else:
                gripper.append(-1)

        root = (1 + math.sqrt(1-stop_height/self.h)) / 2
        for x in np.arange(self.t, 1+self.t, self.t):
            sequence.append(self.cl_pt(self.p2, self.p1, x))
            gripper.append(1)
            if x >= root:
                break
        return sequence, gripper
    

    def cl_pt(self, p1, p2, t, h=None):
        if not h:
            h = self.h
        x_t = p1[0] + (p2[0]-p1[0])*t
        y_t = p1[1] + (p2[1]-p1[1])*t
        # z_t = 4*self.h*t*(1-t) + self.Z_coordinate 
        z1 = p1[-1]
        z2 = p2[-1]

        z_t = (-4 * h + 2 * z1 + 2 * z2)*t*t + (4 * h - 3 * z1 - z2)*t +z1
        return [x_t, y_t, z_t]

    def act(self, current_location):
        self.chasing_idx+=1
        if len(self.sequence) <= self.chasing_idx:
            return False, False
        act = np.array(self.sequence[self.chasing_idx]) - np.array(current_location)
        gripper = self.gripper[self.chasing_idx]
        return (list(act), gripper)



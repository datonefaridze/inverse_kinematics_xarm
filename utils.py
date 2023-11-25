
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
import cv2, queue, threading, time
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import time
import math


def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.uint8)
  return image

def terminate_bool_to_act(terminate_episode: tf.Tensor) -> tf.Tensor:
  return tf.cond(
      terminate_episode == tf.constant(1.0),
      lambda: tf.constant([1, 0, 0], dtype=tf.int32),
      lambda: tf.constant([0, 1, 0], dtype=tf.int32),
  )

def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
  """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
  resc_actions = (actions - low) / (high - low) * (
      post_scaling_max - post_scaling_min
  ) + post_scaling_min
  return tf.clip_by_value(
      resc_actions,
      post_scaling_min + safety_margin,
      post_scaling_max - safety_margin,
  )

def rescale_action(action):
  """Rescales action."""

  action['world_vector'] = rescale_action_with_bound(
      action['world_vector'],
      low=-0.05,
      high=0.05,
      safety_margin=0.01,
      post_scaling_max=1.75,
      post_scaling_min=-1.75,
  )
  action['rotation_delta'] = rescale_action_with_bound(
      action['rotation_delta'],
      low=-0.25,
      high=0.25,
      safety_margin=0.01,
      post_scaling_max=1.4,
      post_scaling_min=-1.4,
  )

  return action

def to_model_action(from_step):
  """Convert dataset action to model action. This function is specific for the Bridge dataset."""

  model_action = {}

  model_action['world_vector'] = from_step['action']['world_vector']
  model_action['terminate_episode'] = terminate_bool_to_act(
      from_step['action']['terminate_episode']
  )

  model_action['rotation_delta'] = from_step['action']['rotation_delta']

  open_gripper = from_step['action']['open_gripper']

  possible_values = tf.constant([True, False], dtype=tf.bool)
  eq = tf.equal(possible_values, open_gripper)

  assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

  with tf.control_dependencies([assert_op]):
    model_action['gripper_closedness_action'] = tf.cond(
        # for open_gripper in bridge dataset,
        # 0 is fully closed and 1 is fully open
        open_gripper,
        # for Fractal data,
        # gripper_closedness_action = -1 means opening the gripper and
        # gripper_closedness_action = 1 means closing the gripper.
        lambda: tf.constant([-1.0], dtype=tf.float32),
        lambda: tf.constant([1.0], dtype=tf.float32),
    )

  model_action = rescale_action(model_action)

  return model_action


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

    def set_location(self, target_position, gripper_action=False):
        if gripper_action:
            self.gripper_act(gripper_action)
        target_angles = self.my_chain.inverse_kinematics(target_position)
        target_angles_degrees = np.array([math.degrees(radian) for radian in target_angles])
        target_angles_degrees =  np.flip(target_angles_degrees[1:-1])
        desired_pos = [[x+3, float(target_angles_degrees[x])] for x in range(len(target_angles_degrees))]
        self.arm.setPosition(desired_pos, duration=1000, wait=True)
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
        
        # print('dist: ', dist)
        # print('dist_prev: ', dist_prev) 
        if min(dist_prev, dist) <= 0.01:
            self.gripper_open = self.gripper[self.chasing_idx]
        return (act, self.gripper_open)

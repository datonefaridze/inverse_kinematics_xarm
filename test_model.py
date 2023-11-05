import numpy as np
import xarm
from typing import Iterator, Tuple, Any
from tf_agents.policies import py_tf_eager_policy
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import tf_agents
from tf_agents.trajectories import time_step as ts
from utils import *
import cv2
import ikpy.chain
import numpy as np
import xarm
import math
import time
import cv2, queue, threading, time

# 3, 4, 5, 6

def calculate_angles(my_chain, pos):
    target_angles = my_chain.inverse_kinematics(pos)
    target_angles_degrees = np.array([math.degrees(radian) for radian in target_angles])
    target_angles_degrees =  np.flip(target_angles_degrees[1:-1])
    desired_pos = [[x+3, float(target_angles_degrees[x])] for x in range(len(target_angles_degrees))]
    return desired_pos


arm = xarm.Controller('USB',)

current_position = [0.2, 0,  0.2]
my_chain = ikpy.chain.Chain.from_urdf_file("xarm.URDF")
desired_pos = calculate_angles(my_chain, current_position)
arm.setPosition(desired_pos, duration=1000, wait=True)
arm.setPosition([[1, -90.]], duration=2000, wait=True)

print("initial_position: ", current_position)



_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

text = "Lift pack of matches in front of you"
# "Take a object next to you and place it in the black box next to you" 
language_embedding = _embed([text])[0].numpy()
saved_model_path = 'robotics_open_x_embodiment_and_rt_x_oss_rt_1_x_tf_trained_for_002272480_step/rt_1_x_tf_trained_for_002272480_step'
tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_model_path,
    load_specs_from_pbtxt=True,
    use_tf_function=True)
observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))
policy_state = tfa_policy.get_initial_state(batch_size=1)


class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
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

cap = VideoCapture(2)

idx=0
start = time.time()
while True:
    frame = cap.read()
    ret =True
    if ret == True:
        image = resize(frame)
        observation['image'] = image
        observation['natural_language_instruction'] = text
        observation['natural_language_embedding'] = language_embedding
        tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
        policy_step = tfa_policy.action(tfa_time_step, policy_state)
        action = policy_step.action
        policy_state = policy_step.state

        temp_pos = np.array(current_position) + action['world_vector']
        desired_pos = calculate_angles(my_chain, temp_pos)
        # idx=idx+1
        # np.save(f'saved_data/frame_{idx}', frame)
        # np.save(f'saved_data/desired_pos_{idx}', temp_pos)
        print('time', time.time()-start)
        # arm.setPosition(desired_pos, duration=1000, wait=True)
        if np.argmax(action['terminate_episode'])==0:
            arm.setPosition([[1, 0]], duration=1000, wait=True)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("you suck")
        break 

cap.release()
cv2.destroyAllWindows()
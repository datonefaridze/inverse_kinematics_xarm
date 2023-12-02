import cv2
import xarm
import numpy as np
import time
from utils import VideoCapture, XarmIK, Controller
from mujoco_vc.model_loading import (
    load_pretrained_model,
    fuse_embeddings_flare,
)
import numpy as np, time as pickle, torch
import torchvision.transforms as T
from collections import deque



def fuse_embeddings_flare(embeddings: list):
    if type(embeddings[0]) == np.ndarray:
        import pdb; pdb.set_trace()

        history_window = len(embeddings)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1].copy())
        return np.array(delta).ravel()
    elif type(embeddings[0]) == torch.Tensor:
        history_window = len(embeddings)
        # each embedding will be (Batch, Dim)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1])
        return torch.cat(delta, dim=1)
    else:
        print("Unsupported embedding format in fuse_embeddings_flare.")
        print("Provide either numpy.ndarray or torch.Tensor.")
        quit()



arm = xarm.Controller('USB',)
xarm_ik = XarmIK(arm, '/home/dato/src/my_projects/robotics/inverse_kinematics_xarm/xarm.URDF')  
cap = VideoCapture(2)
embedding_name = 'vc1_vitl'
device = 'cuda:0'
controller = Controller([-0.1519, 0.0000, 0.03]) 

model, embedding_dim, transforms, metadata = load_pretrained_model(
    embedding_name=embedding_name
)
model.to(device)
print('=========================')
print('=========================')
print('setting a location ')

xarm_ik.set_location([-0.1182, -0.0001, 0.1917])
gripper_state = 1
current_location = xarm_ik.get_location()

fixed_queue = deque(maxlen=3)


model_path = 'best_policy_xarm.pickle'
policy = pickle.load(open(model_path, "rb"))
policy.model.eval()

model.to(device)
frames = []

while(True):
    frame = cap.read()
    
    cv2.imshow('frame',frame)
    frames.append(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    current_location = xarm_ik.get_location()

    emb = model(transforms(frame).to(device))
    fixed_queue.append(emb)
    while len(fixed_queue) < 3:
        fixed_queue.append(emb)

    feat_t = fuse_embeddings_flare(list(fixed_queue))

    with torch.no_grad():
        pred = policy.model(feat_t)

    next_location = current_location + pred[0][:3].cpu().numpy()/100
    gripper_state = 1 if pred[0][-1] >=0 else -1
    xarm_ik.set_location(next_location, gripper_state)


yn = input('wanna save:?')

dataset_dir = 'saved_frames'
if yn == 'y':
    print('saving...')
    filename = str(time.time()).replace(".", "")
    np.save(f'{dataset_dir}/frames_{filename}.npy', frames)


cap.stop()  # Stop the video capture thread
cv2.destroyAllWindows()
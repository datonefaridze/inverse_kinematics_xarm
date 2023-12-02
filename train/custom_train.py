
from train_loop import bc_pvr_train_loop

"/home/dato/src/my_projects/robotics/eai-vc/cortexbench/mujoco_vc/visual_imitation/data/datasets/xarm/xarm.pickle"

config = {'demo_paths_loc': "/home/dato/src/my_projects/robotics/eai-vc/cortexbench/mujoco_vc/visual_imitation/data/datasets/xarm/xarm.pickle",
'wandb': {'project': 'cortexbench', 'entity': 'cortexbench', 'mode': 'offline'},
'env': 'relocate-v0', 'algorithm': 'BC', 'pixel_based': True, 'embedding': 'vc1_vitl',
 'camera': 'vil_camera', 'device': 'cuda', 'data_dir': 'data/datasets/adroit-expert-v1.0/', 'data_parallel': True,
'seed': 100, 'epochs': 400, 'eval_frequency': 5, 'save_frequency': 10, 'eval_num_traj': 25, 'num_cpu': 1, 'num_demos': 100,
'exp_notes': 'Add experiment notes here to help organize results down the road.',
'env_kwargs': {'env_name': 'relocate-v0', 'suite': 'adroit', 'device': 'cuda', 'image_width': 256, 'image_height': 256,
 'camera_name': 'vil_camera', 'embedding_name': 'vc1_vitl', 'pixel_based': True, 'render_gpu_id': 0,
  'seed': 100, 'history_window': 3, 'add_proprio': True, 'proprio_key': 'proprio'},
 'bc_kwargs': {'hidden_sizes': '(256, 256, 256)', 'nonlinearity': 'relu', 'loss_type': 'MSE', 'batch_size': 256, 'lr': 0.001, 'dropout': 0},
   'job_name': 'adroit_cortex_vil'}


bc_pvr_train_loop(config)
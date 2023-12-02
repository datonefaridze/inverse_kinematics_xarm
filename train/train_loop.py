#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP, BatchNormMLP
from mujoco_vc.gym_wrapper import env_constructor
from mujoco_vc.model_loading import (
    load_pretrained_model,
    fuse_embeddings_concat,
    fuse_embeddings_flare,
)
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from vc_models.utils.wandb import setup_wandb
import numpy as np, time as pickle, os, torch, gc


def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


def bc_pvr_train_loop(config: dict) -> None:
    class env_spec:
        observation_dim = 3072
        action_dim = 4

    set_seed(config["seed"])

    demo_paths_loc = demo_paths_loc['demo_paths_loc'] 
    try:
        demo_paths = pickle.load(open(demo_paths_loc, "rb"))
    except:
        print("Unable to load the data. Check the data path.")
        print(demo_paths_loc)
        quit()

    demo_paths = demo_paths[: config["num_demos"]]
    print("Number of demonstrations used : %i" % len(demo_paths))
        
    policy = BatchNormMLP(
        env_spec=env_spec,
        hidden_sizes=eval(config["bc_kwargs"]["hidden_sizes"]),
        seed=config["seed"],
        nonlinearity=config["bc_kwargs"]["nonlinearity"],
        dropout=config["bc_kwargs"]["dropout"],
    )

    print("===================================================================")
    print(">>>>>>>>> Precomputing frozen embedding dataset >>>>>>>>>>>>>>>>>>>")
    demo_paths = compute_embeddings(
        demo_paths,
        device=config["device"],
        embedding_name=config["env_kwargs"]["embedding_name"],
    )
    demo_paths = precompute_features(
        demo_paths,
        history_window=config["env_kwargs"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
        proprio_key=config["env_kwargs"]["proprio_key"],
    )
    gc.collect()  # garbage collection to free up RAM
    dataset = FrozenEmbeddingDataset(
        demo_paths,
        history_window=config["env_kwargs"]["history_window"],
        fuse_embeddings=fuse_embeddings_flare,
    )
    # Dataset in this case is pre-loaded and on the RAM (CPU) and not on the disk
    dataloader = DataLoader(
        dataset,
        batch_size=config["bc_kwargs"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(
        list(policy.model.parameters()), lr=config["bc_kwargs"]["lr"]
    )
    loss_func = torch.nn.MSELoss()


    os.chdir(config["job_name"])  # important! we are now in the directory to save data
    if os.path.isdir("iterations") == False:
        os.mkdir("iterations")
    if os.path.isdir("logs") == False:
        os.mkdir("logs")


    for epoch in tqdm(range(config["epochs"])):
        # move the policy to correct device
        policy.model.to(config["device"])
        policy.model.train()
        # update policy for one BC epoch
        running_loss = 0.0
        for mb_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            feat = batch["features"].float().to(config["device"])
            tar = batch["actions"].float().to(config["device"])

            pred = policy.model(feat)
            loss = loss_func(pred, tar.detach())
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.to("cpu").data.numpy().ravel()[0]
        print({"epoch_loss": running_loss / (mb_idx + 1)}, epoch + 1)
    pickle.dump(policy, open("best_policy_xarm.pickle", "wb"))


class FrozenEmbeddingDataset(Dataset):
    def __init__(
        self,
        paths: list,
        history_window: int = 1,
        fuse_embeddings: callable = None,
        device: str = "cuda",
    ):
        self.paths = paths
        assert "embeddings" in self.paths[0].keys()
        # assume equal length trajectories
        # code will work even otherwise but may have some edge cases
        self.path_length = max([p["actions"].shape[0] for p in paths])
        self.num_paths = len(self.paths)
        self.history_window = history_window
        self.fuse_embeddings = fuse_embeddings
        self.device = device

    def __len__(self):
        return self.path_length * self.num_paths

    def __getitem__(self, index):
        traj_idx = int(index // self.path_length)
        timestep = int(index - traj_idx * self.path_length)
        timestep = min(timestep, self.paths[traj_idx]["actions"].shape[0])
        if "features" in self.paths[traj_idx].keys():
            features = self.paths[traj_idx]["features"][timestep]
            action = self.paths[traj_idx]["actions"][timestep]
        else:
            embeddings = [
                self.paths[traj_idx]["embeddings"][max(timestep - k, 0)]
                for k in range(self.history_window)
            ]
            embeddings = embeddings[
                ::-1
            ]  # embeddings[-1] should be most recent embedding
            features = self.fuse_embeddings(embeddings)
            # features = torch.from_numpy(features).float().to(self.device)
            action = self.paths[traj_idx]["actions"][timestep]
            # action   = torch.from_numpy(action).float().to(self.device)
        return {"features": features, "actions": action}


def compute_embeddings(
    paths: list, embedding_name: str, device: str = "cpu", chunk_size: int = 20
):
    model, embedding_dim, transforms, metadata = load_pretrained_model(
        embedding_name=embedding_name
    )
    model.to(device)
    for path in tqdm(paths):
        inp = path["images"]  # shape (B, H, W, 3)
        path["embeddings"] = np.zeros((inp.shape[0], embedding_dim))
        path_len = inp.shape[0]

        preprocessed_inp = torch.cat(
            [transforms(frame) for frame in inp]
        )  # shape (B, 3, H, W)
        for chunk in range(path_len // chunk_size + 1):
            if chunk_size * chunk < path_len:
                with torch.no_grad():
                    inp_chunk = preprocessed_inp[
                        chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                    ]
                    emb = model(inp_chunk.to(device))
                    # save embedding in RAM and free up GPU memory
                    emb = emb.to("cpu").data.numpy()
                path["embeddings"][
                    chunk_size * chunk : min(chunk_size * (chunk + 1), path_len)
                ] = emb
        del path["images"]  # no longer need the images, free up RAM
    return paths


def precompute_features(
    paths: list,
    history_window: int = 1,
    fuse_embeddings: callable = None,
    proprio_key: str = None,
):
    assert "embeddings" in paths[0].keys()
    for path in paths:
        features = []
        for t in range(path["embeddings"].shape[0]):
            emb_hist_t = [
                path["embeddings"][max(t - k, 0)] for k in range(history_window)
            ]
            emb_hist_t = emb_hist_t[
                ::-1
            ] 
            feat_t = fuse_embeddings(emb_hist_t)
            features.append(feat_t.copy())
        path["features"] = np.array(features)
    return paths

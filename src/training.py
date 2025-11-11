import os
import warnings
from time import time
from tqdm import tqdm
from threading import Thread
from functools import partial
from contextlib import nullcontext

# Ignore the wandb warning(this is a bad practice I know)
warnings.filterwarnings("ignore")

import wandb
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
)

from models import SparseAutoencoder, TopK


# Torch setup
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def can_use_bfloat16(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # get capability for current device index
        major, minor = torch.cuda.get_device_capability(device.index)
    except Exception:
        major, minor = torch.cuda.get_device_capability()
    # Ampere (sm_80) and newer support efficient bfloat16
    return major >= 8

USE_BFLOAT16 = can_use_bfloat16(device)
print("USE_BFLOAT16 autocast:", USE_BFLOAT16)
# autocast context factory (nullcontext if not available)
if USE_BFLOAT16:
    autocast_ctx = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext

# -------- USER: paths to your data files --------
# ACTIVATIONS_PATH = "dataset/middle_layer_activations.pt"  # shape: (N, seq_len, hidden_dim)
OUT_DIR = "sae_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HYPERPARAMS --------
BATCH_SIZE = int(2 ** 14)
print("batch size:", BATCH_SIZE)
LR = 1e-4
WEIGHT_DECAY = 1e-5
SAE_EPOCHS = 1
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
# -------- Utility: load activations and labels --------
N_ACTIVATIONS_DIMS = 1024

# -------- Create dataset and splits --------
time_to_load_shards = 0
class ResidualActivationsDataset(Dataset):
    def __init__(self, paths_to_shards: list[str], device: str="cpu"):
        super().__init__()
        self.device = device
        self.paths_to_shards = paths_to_shards
        self.num_shards = len(paths_to_shards)

        # Load first shard synchronously
        self.current_shard_index = 0
        self.current_shard = torch.load(paths_to_shards[0], weights_only=True, map_location=self.device)
        self.shards_len = self.current_shard.shape[0]
        self.length = self.shards_len * self.num_shards

        # For async preloading
        self.next_shard = None
        self.next_shard_index = (self.current_shard_index + 1) % self.num_shards
        self._preload_thread = None
        self._start_async_load(self.next_shard_index)

        print(f"Loaded shard 0 of {self.num_shards}, each shard len = {self.shards_len}")

    def _load_shard(self, shard_index: int):
        """Background loader for a shard."""
        file_path = self.paths_to_shards[shard_index]
        return torch.load(file_path, weights_only=True, map_location=self.device)

    def _start_async_load(self, shard_index: int):
        """Start async loading in a background thread."""
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return  # avoid overlapping loads

        def _worker():
            self.next_shard = self._load_shard(shard_index)
            self.next_shard_index = shard_index

        self._preload_thread = Thread(target=_worker, daemon=True)
        self._preload_thread.start()

    def _switch_to_next_shard(self):
        """Switch to the preloaded shard and start loading the next one."""
        if self._preload_thread is not None:
            self._preload_thread.join()  # ensure shard fully loaded

        self.current_shard = self.next_shard
        self.current_shard_index = self.next_shard_index
        next_index = (self.current_shard_index + 1) % self.num_shards
        self._start_async_load(next_index)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        start_time = time()
        global time_to_load_shards
        shard_i = index // self.shards_len
        shard_local_i = index % self.shards_len

        # Switch shards if needed
        if self.current_shard_index != shard_i:
            self._switch_to_next_shard()
            print(f"Switched to shard {self.current_shard_index} in {time() - start_time:.2f}s")
        time_to_load_shards += time() - start_time
        return self.current_shard[shard_local_i]

train_ds = ResidualActivationsDataset(
    paths_to_shards=[
        "dataset/residual_activations_shard_001.npy",
        "dataset/residual_activations_shard_002.npy",
        "dataset/residual_activations_shard_003.npy",
        "dataset/residual_activations_shard_004.npy",
        "dataset/residual_activations_shard_005.npy",
        "dataset/residual_activations_shard_006.npy",
        "dataset/residual_activations_shard_007.npy",
        "dataset/residual_activations_shard_008.npy",
    ],
)
val_ds = ResidualActivationsDataset(
    paths_to_shards=[
        "dataset/residual_activations_shard_009.npy",
        "dataset/residual_activations_shard_010.npy",
        "dataset/residual_activations_shard_011.npy",
    ],
)
def mk_data_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=SequentialSampler(dataset),
    )

train_loader = mk_data_loader(train_ds)
val_loader = mk_data_loader(val_ds)

EXPANSION_RATIO = 16
N_LATENTS = N_ACTIVATIONS_DIMS * EXPANSION_RATIO
USE_NORMALIZATION = False
K = 256
sae = SparseAutoencoder(
    n_inputs=N_ACTIVATIONS_DIMS,
    n_latents=N_LATENTS,
    activation=TopK(256),
    tied=True,
    normalize=USE_NORMALIZATION,
)
sae = sae.to(device)
# sae = torch.compile(sae)
optimizer = torch.optim.AdamW(sae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_loss = nn.MSELoss(reduction="mean")

# Weights and Biases
LOG_COMMIT_FREQ = 100
wandb.init(
    project="sparse-auto-encoders",
    config={
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "l1_loss_weight": LAMBDA_L1,
        "sae_training_epochs": SAE_EPOCHS,
        "using_bfloat16": USE_BFLOAT16,
        "sparse_activations_size": N_LATENTS,
        "normalization": USE_NORMALIZATION,
        "K": K,
    }
)

def train_sae(sae: nn.Module):
    step = 0
    eval_model(sae, step)
    time_to_move_to_device = 0
    for _ in range(1, SAE_EPOCHS + 1):
        sae.train()
        for x in tqdm(train_loader):
            start_time = time()
            x = x.to(device)
            time_to_move_to_device += time() - start_time
            with autocast_ctx():
                _, latents, recons = sae(x)
                reconstruction_loss = mse_loss(recons, x)
                l0_loss = (latents > 0).float().mean()
                l1_loss = latents.abs().mean()
                loss = reconstruction_loss #+ LAMBDA_L1 * l1_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(
                data={
                    "training/recon_loss": reconstruction_loss.item(),
                    "training/l0_loss": l0_loss.item(),
                    "training/loss_l1": l1_loss.item(),
                    "training/loss": loss.item(),
                },
                step=step,
                commit=step % LOG_COMMIT_FREQ == 0,
            )
            step += 1
        global time_to_load_shards
        print("time to load shards:", time_to_load_shards, "time_to_move_to_device:", time_to_move_to_device)
        time_to_load_shards = 0

        eval_model(sae, step)

def eval_model(sae: nn.Module, step: int):
    sae.eval()
    l0_loss = 0
    l1_loss = 0
    reconstruction_loss = 0
    time_to_move_to_device = 0
    with torch.no_grad():
        for x in tqdm(val_loader, desc="evaluating model on validation split."):
            start_time = time()
            x = x.to(device)
            time_to_move_to_device += time() - start_time
            batch_weight = x.shape[0] / len(val_loader.dataset)
            with autocast_ctx():
                _, latents, recons = sae(x)
            l1_loss += latents.abs().mean().item() * batch_weight
            l0_loss += (latents > 0).float().mean().item() * batch_weight
            reconstruction_loss += mse_loss(recons, x).item() * batch_weight
    global time_to_load_shards
    print("time to load shards:", time_to_load_shards, "time_to_move_to_device:", time_to_move_to_device)
    time_to_load_shards = 0
    wandb.log(
        data={
            "validation/l1_loss": l1_loss,
            "validation/l0_loss": l0_loss,
            "validation/reconstruction_loss": reconstruction_loss,
            "validation/loss": reconstruction_loss + l1_loss * LAMBDA_L1
        },
        step=step
    )

if __name__ == "__main__":
    train_sae(sae)
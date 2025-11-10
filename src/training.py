import os
import warnings
from time import time
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext

# Ignore the wandb warning(this is a bad practice I know)
warnings.filterwarnings("ignore")

import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
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
BATCH_SIZE = 512
print("batch size:", BATCH_SIZE)
LR = 1e-4
WEIGHT_DECAY = 1e-5
SAE_EPOCHS = 64
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
# -------- Utility: load activations and labels --------
N_ACTIVATIONS_DIMS = 1024

# -------- Create dataset and splits --------
time_to_load_shards = 0
class ResidualActivationsDataset(Dataset):
    def __init__(self, paths_to_shards: str):
        super().__init__()
        self.paths_to_shards = paths_to_shards
        self.shard_lengths = []
        self.length = 0
        for file_path in self.paths_to_shards:
            shard = np.load(file_path, mmap_mode="r", allow_pickle=False)
            self.shard_lengths.append(shard.shape[0])
            self.length += shard.shape[0] * shard.shape[1] 
        self.length = sum(self.shard_lengths)
        self.current_shard_index = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        global time_to_load_shards
        start_time = time()
        total_shard_length = 0
        shard_it = enumerate(zip(self.paths_to_shards, self.shard_lengths))
        for shard_i, (file_path, shard_len) in shard_it:
            if total_shard_length + shard_len > index:
                time_to_load_shards += time() - start_time
                if self.current_shard_index is None or shard_i != self.current_shard_index:
                    self.current_shard = np.load(file_path, mmap_mode="r", allow_pickle=False)
                    self.current_shard_index = shard_i
                # print("getting item, shard_i:", shard_i)
                return self.current_shard[index - total_shard_length]
            total_shard_length += shard_len
        raise IndexError(f"Index {index} is out of range {self.length}")

train_ds = ResidualActivationsDataset([
    "dataset/residual_activations_shard_001.npy",
    "dataset/residual_activations_shard_002.npy",
    "dataset/residual_activations_shard_003.npy",
    "dataset/residual_activations_shard_004.npy",
    "dataset/residual_activations_shard_005.npy",
    "dataset/residual_activations_shard_006.npy",
    "dataset/residual_activations_shard_007.npy",
    "dataset/residual_activations_shard_008.npy",
])
val_ds = ResidualActivationsDataset([
    "dataset/residual_activations_shard_009.npy",
    "dataset/residual_activations_shard_010.npy",
    "dataset/residual_activations_shard_011.npy",
])
train_sampler = SequentialSampler(train_ds)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)
val_sampler = SequentialSampler(val_ds)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=val_sampler)

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
    for _ in range(1, SAE_EPOCHS + 1):
        sae.train()
        for (xb, ) in tqdm(train_loader):
            xb = xb.to(device)
            with autocast_ctx():
                _, latents, recons = sae(xb)
                reconstruction_loss = mse_loss(recons, xb)
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
        eval_model(sae, step)

def eval_model(sae: nn.Module, step: int):
    sae.eval()
    l0_loss = 0
    l1_loss = 0
    reconstruction_loss = 0
    with torch.no_grad():
        for x in tqdm(val_loader, desc="evaluating model on validation split."):
            x = x.to(device)
            batch_weight = x.shape[0] / len(val_loader.dataset)
            _, latents, recons = sae(x)
            l1_loss += latents.abs().mean().item() * batch_weight
            l0_loss += (latents > 0).float().mean().item() * batch_weight
            reconstruction_loss += mse_loss(recons, x).item() * batch_weight
    print("time to load shards:", time_to_load_shards)
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
import os
import warnings
from time import time
from tqdm import tqdm
from threading import Thread
from typing import Generator
from functools import partial
from contextlib import nullcontext

# Ignore the wandb warning(this is a bad practice I know)
warnings.filterwarnings("ignore")

import wandb
import torch
from torch import nn, Tensor
from safetensors.torch import load_file
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
time_to_getitem = 0
class ShardedDataset(Dataset):
    def __init__(self, paths_to_shards: list[str], batch_size: int, device: str="cpu"):
        super().__init__()
        self.paths_to_shards = paths_to_shards
        self.device = device
        self.batch_size = batch_size
        # init shard attributes
        self.n_shards = len(paths_to_shards)
        self.current_shard_index = 0
        self.current_shard = self._load_shard(self.current_shard_index)
        # self.load_first_two_shards()
        self.shards_len = self.current_shard.shape[0]
        self.length = self.shards_len * self.n_shards

        print(f"Loaded shard 0 of {self.n_shards}, each shard len = {self.shards_len}")

    def __len__(self):
        return self.length

    # def __getitem__(self, index: int):
    #     start_time = time()
    #     global time_to_getitem
    #     shard_i = index // self.shards_len
    #     shard_local_i = index % self.shards_len
    #     # Switch shards if needed
    #     if self.current_shard_index != shard_i:
    #         self._switch_to_next_shard()
    #         print(f"Switched to shard {self.current_shard_index} in {time() - start_time:.2f}s")
    #     time_to_getitem += time() - start_time
    #     return self.current_shard[shard_local_i]

    def iter_over_batches(self, batch_size: int) -> Generator[Tensor]:
        self.current_shard_index = 0
        self.current_shard = self._load_shard(self.current_shard_index)
        for batch_start_i in range(0, self.length, batch_size):
            batch_end_i = batch_start_i + batch_size
            start_shard_i, start_local_shard_i = self.shard_and_local_shard_index(batch_start_i)
            end_shard_i, end_local_shard_i = self.shard_and_local_shard_index(batch_end_i)
            # if start_shard_i != self.current_shard_index:
            #     self._switch_to_next_shard()
            if start_shard_i == end_shard_i: # The batch lies inside a single shard
                yield self.current_shard[start_local_shard_i:end_local_shard_i]
            elif end_shard_i < self.n_shards: # The batch overlaps over two shards
                next_shard = self._load_shard(end_shard_i)
                # if self._preload_thread is not None:
                #     self._preload_thread.join()  # ensure next shard is fully loaded
                yield torch.cat((
                    self.current_shard[start_local_shard_i:],
                    next_shard[:end_local_shard_i],
                ))
                self.current_shard = next_shard
            else:
                yield self.current_shard[start_local_shard_i:]

    def shard_and_local_shard_index(self, index: int) -> tuple[int, int]:
        return (
            (index // self.shards_len) % self.n_shards,
            index % self.shards_len,
        )

    # def load_first_two_shards(self):
    #     self.current_shard_index = 0
    #     self.current_shard = self._load_shard(self.current_shard_index)
        # self.next_shard = None
        # self._preload_thread = None
        # self._start_async_load_of_next_shard()

    # def _switch_to_next_shard(self):
    #     """Switch to the preloaded shard and start loading the next one."""
    #     if self._preload_thread is not None:
    #         self._preload_thread.join()  # ensure shard fully loaded
    #     self.current_shard = self.next_shard
    #     self.current_shard_index = self.next_shard_index
    #     self._start_async_load_of_next_shard()

    # def _start_async_load_of_next_shard(self):
    #     """Start async loading in a background thread."""
    #     def _load_next_shard():
    #         self.next_shard = self._load_shard(self.next_shard_index)
    #     if self._preload_thread is None or not self._preload_thread.is_alive():
    #         self._preload_thread = Thread(target=_load_next_shard, daemon=True)
    #         self._preload_thread.start()

    @property
    def next_shard_index(self) -> int:
        return (self.current_shard_index + 1) % self.n_shards

    def _load_shard(self, shard_index: int):
        file_path = self.paths_to_shards[shard_index]
        return load_file(file_path)["shard"]

train_ds = ShardedDataset(
    paths_to_shards=[
        "safetensors_dataset/residual_activations_shard_001.safetensor",
        "safetensors_dataset/residual_activations_shard_002.safetensor",
        "safetensors_dataset/residual_activations_shard_003.safetensor",
        "safetensors_dataset/residual_activations_shard_004.safetensor",
        "safetensors_dataset/residual_activations_shard_005.safetensor",
        "safetensors_dataset/residual_activations_shard_006.safetensor",
        "safetensors_dataset/residual_activations_shard_007.safetensor",
        "safetensors_dataset/residual_activations_shard_008.safetensor",
    ],
    batch_size=BATCH_SIZE,
)
val_ds = ShardedDataset(
    paths_to_shards=[
        "safetensors_dataset/residual_activations_shard_009.safetensor",
        "safetensors_dataset/residual_activations_shard_010.safetensor",
        "safetensors_dataset/residual_activations_shard_011.safetensor",
    ],
    batch_size=BATCH_SIZE,
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
    prev_batch = None
    for _ in range(1, SAE_EPOCHS + 1):
        sae.train()
        batch_it = tqdm(
            train_ds.iter_over_batches(batch_size=BATCH_SIZE),
            desc="Training model",
            total=len(train_loader),
        )
        for x in batch_it:
            start_time = time()
            x = x.to(device)
            if prev_batch is not None and prev_batch.shape[0] == x.shape[0]:
                equality = (prev_batch == x).all(dim=1)
                # print(
                #     "n identical samples:",
                #     equality.sum().item(),
                #     ", n samples per batch:",
                #     prev_batch.shape[0],
                #     "equality shape:",
                #     equality.shape,
                # )
            prev_batch = x

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
        global time_to_getitem
        print("time to load shards:", time_to_getitem, "time_to_move_to_device:", time_to_move_to_device)
        time_to_getitem = 0

        eval_model(sae, step)

def eval_model(sae: nn.Module, step: int):
    sae.eval()
    l0_loss = 0
    l1_loss = 0
    reconstruction_loss = 0
    prev_batch = None
    time_to_move_to_device = 0
    batch_it = tqdm(
        val_ds.iter_over_batches(BATCH_SIZE),
        desc="evaluating model on validation split.",
        total=len(val_loader),
    )
    with torch.no_grad():
        for x in batch_it:
            start_time = time()
            x = x.to(device)
            # if prev_batch is not None and prev_batch.shape[0] == x.shape[0]:
            #     equality = (prev_batch == x).all(dim=1)
            #     print(
            #         "n identical samples:",
            #         equality.sum().item(),
            #         ", n samples per batch:",
            #         prev_batch.shape[0],
            #         "equality shape:",
            #         equality.shape,
            #     )
            # prev_batch = x
            time_to_move_to_device += time() - start_time
            batch_weight = x.shape[0] / len(val_loader.dataset)
            with autocast_ctx():
                _, latents, recons = sae(x)
            l1_loss += latents.abs().mean().item() * batch_weight
            l0_loss += (latents > 0).float().mean().item() * batch_weight
            reconstruction_loss += mse_loss(recons, x).item() * batch_weight
    global time_to_getitem
    print("time to load shards:", time_to_getitem, "time_to_move_to_device:", time_to_move_to_device)
    time_to_getitem = 0
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
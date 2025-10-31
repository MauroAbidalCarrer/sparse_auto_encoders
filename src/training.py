import os
import warnings
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext

# Ignore the wandb warning(this is a bad practice I know)
warnings.filterwarnings("ignore") 

import wandb
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

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
ACTIVATIONS_PATH = "dataset/middle_layer_activations.pt"  # shape: (N, seq_len, hidden_dim)
OUT_DIR = "sae_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HYPERPARAMS --------
BATCH_SIZE = 512
print(BATCH_SIZE)
LR = 1e-4
WEIGHT_DECAY = 1e-5
SAE_EPOCHS = 64
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
# -------- Utility: load activations and labels --------
activations: torch.Tensor = torch.load(ACTIVATIONS_PATH, weights_only=True)  # (N_TOKENS, EMBED_DIMS)
activations = activations.float()

# -------- Create dataset and splits --------
dataset = TensorDataset(activations)
n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

EXPANSION_RATIO = 16
N_LATENTS = activations.shape[1] * EXPANSION_RATIO
USE_NORMALIZATION = False
K = 256
sae = SparseAutoencoder(
    n_inputs=activations.shape[1],
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
        for (x, ) in val_loader:
            x = x.to(device)
            batch_weight = x.shape[0] / len(val_loader.dataset)
            _, latents, recons = sae(x)
            l1_loss += latents.abs().mean().item() * batch_weight
            l0_loss += (latents > 0).float().mean().item() * batch_weight
            reconstruction_loss += mse_loss(recons, x).item() * batch_weight
    wandb.log(
        data={
            "validation/l1_loss": l1_loss,
            "validation/l0_loss": l0_loss,
            "validation/reconstruction_loss": reconstruction_loss,
            "validation/loss": reconstruction_loss + l1_loss * LAMBDA_L1
        },
        step=step
    )
    classify_category_from_sae_features(sae, step)
    
class TokenQuestionClassifier(nn.Module):
    def __init__(self, sae: nn.Module, n_targets: int):
        super().__init__()
        self.sae = sae
        self.clssifier = nn.Sequential(
            nn.LazyBatchNorm1d(),
            # nn.LazyLinear(256),
            # nn.LazyBatchNorm1d(),
            nn.LazyLinear(n_targets),
        )
    
    def forward(self, activations: Tensor) -> Tensor:
        _, latents, _ = self.sae(activations)
        return self.clssifier(latents)
    
def classify_category_from_sae_features(sae: nn.Module, step:int):
    set_require_grad(sae, False)
    cls_dataset = mk_token_category_dataset()
    n_targets = cls_dataset.tensors[1].shape[1]
    cls_model = TokenQuestionClassifier(sae, n_targets).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(cls_model.parameters())
    n_train = int(len(cls_dataset) * 0.8)
    n_val = len(cls_dataset) - n_train
    train_ds, val_ds = random_split(
        cls_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, BATCH_SIZE * 128)
    val_dl = DataLoader(val_ds, BATCH_SIZE * 128)
    for epoch in range(1):
        train_loss = 0
        train_accuracy = 0
        cls_model.train()
        for x, y in tqdm(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = cls_model(x)
            batch_loss = criterion(y_pred, y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() * x.shape[0] / len(train_loader.dataset)
            batch_accuracy = (
                (y_pred.max(dim=1).indices == y.max(dim=1).indices)
                .float()
                .mean()
                .item()
            )
            train_accuracy += batch_accuracy * x.shape[0] / len(train_loader.dataset)

        cls_model.eval()
        # Validation
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                y_pred: Tensor = cls_model(x)
                batch_loss = criterion(y_pred, y)
                val_loss += batch_loss.item() * x.size(0) / len(train_loader.dataset)
                batch_accuracy = (
                    (y_pred.max(dim=1).indices == y.max(dim=1).indices)
                    .float()
                    .mean()
                    .item()
                )
                val_accuracy += batch_accuracy * x.size(0) / len(train_loader.dataset)
        print(f"epoch {epoch}, train loss {train_loss:.3f}, train_accuracy {train_accuracy:.3f}, val loss {val_loss:.3f}, val accuracy {val_accuracy:.3f}")
    wandb.log(
        data={
            "classification/train_loss": train_loss,
            "classification/train_accuracy": train_accuracy,
            "classification/validation_loss": val_loss,
            "classification/validation_accuracy": val_accuracy,
        },
        step=step,
    )
    set_require_grad(sae, True)

def set_require_grad(module: nn.Module, value: bool):
    for param in module.parameters():
        param.requires_grad = value

def mk_token_category_dataset() -> TensorDataset:
    meta_df = pd.read_parquet("dataset/token_metadata.parquet")
    meta_df["token_idx"] = np.arange(len(meta_df))
    meta_df = meta_df.query("has_label")
    targets = pd.get_dummies(meta_df["label"]).astype("float").values
    targets = torch.from_numpy(targets)
    return TensorDataset(
        activations[meta_df["token_idx"].values],
        targets,
    )

if __name__ == "__main__":
    train_sae(sae)
import os
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


# -------- USER: paths to your data files --------
ACTIVATIONS_PATH = "dataset/middle_layer_activations.pt"  # shape: (N, seq_len, hidden_dim)
OUT_DIR = "sae_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HYPERPARAMS --------
POOL_METHOD = "last"      # options: "mean", "last", "max"
BOTTLE_DIM = 128
HIDDEN_FACTOR = 0.5       # hidden_dim * factor -> first encoder hidden
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 6
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("batch size:", BATCH_SIZE)

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

# -------- Model: simple SAE --------
class SparseAutoEncoder(nn.Module):
    def __init__(self, model_embed_size: int, sparse_activation_expantion: int, dropout_ratio: float):
        super().__init__()
        sparse_activations_size = model_embed_size * sparse_activation_expantion
        print("sparse_activations_size", sparse_activations_size)
        self.encoder = nn.Sequential(
            nn.Linear(model_embed_size, sparse_activations_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sparse_activations_size, model_embed_size),
        )

    def forward(self, x):
        saprse_activation = self.encoder(x)  # (B, bottle_dim), non-negative due to ReLU
        reconstruction = self.decoder(saprse_activation)
        return reconstruction, saprse_activation

sae = SparseAutoEncoder(
    model_embed_size=activations.shape[1],
    sparse_activation_expantion=8,
    dropout_ratio=0.2,
)
sae = sae.to(device)
# sae = torch.compile(sae)
optimizer = torch.optim.AdamW(sae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_loss = nn.MSELoss(reduction="mean")


torch.set_float32_matmul_precision('high')
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

def train_sae(sae: nn.Module):
    eval_model(sae, 0, 0)
    for epoch in range(1, EPOCHS + 1):
        sae.train()
        train_loss = 0.0
        for (xb, ) in tqdm(train_loader):
            xb = xb.to(device)
            with autocast_ctx():
                reconstruction, sparse_activations = sae(xb)
                loss_recon = mse_loss(reconstruction, xb)
                loss_l1 = sparse_activations.abs().mean()
                loss = loss_recon + LAMBDA_L1 * loss_l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        eval_model(sae, epoch, train_loss)

def eval_model(model: nn.Module, epoch: int, train_loss: float):
    model.eval()
    val_loss = 0
    l0_norm_loss = 0
    zs_val = []
    with torch.no_grad():
        for (x, ) in val_loader:
            x = x.to(device)
            reconstruction, sparse_activations = model(x)
            loss_recon = mse_loss(reconstruction, x)
            loss_l1 = sparse_activations.abs().mean()
            loss = loss_recon + LAMBDA_L1 * loss_l1
            val_loss += loss.item() * x.size(0)
            zs_val.append(sparse_activations.cpu().numpy())
            batch_l0_norm_loss = (sparse_activations != 0).float().mean().item()
            l0_norm_loss = batch_l0_norm_loss * x.shape[0]

    val_loss /= len(val_loader.dataset)
    zs_val = np.vstack(zs_val) if len(zs_val) else np.zeros((0, BOTTLE_DIM))
    print(f"Epoch {epoch} | Train loss {train_loss:.3f} | Val loss {val_loss:.3f} | l0_norm_loss {l0_norm_loss:.3f}")
    classify_category_from_sae_features(model)
    
class TokenQuestionClassifier(nn.Module):
    def __init__(self, sae: nn.Module, n_targets: int):
        super().__init__()
        self.sae = sae
        self.clssifier = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(256),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(n_targets),
        )
    
    def forward(self, activations: Tensor) -> Tensor:
        _, sae_features = self.sae(activations)
        return self.clssifier(sae_features)
    
def classify_category_from_sae_features(sae: nn.Module):
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
    train_dl = DataLoader(train_ds, BATCH_SIZE)
    val_dl = DataLoader(val_ds, BATCH_SIZE)
    for epoch in range(4):
        train_loss = 0
        train_accuracy = 0
        cls_model.train()
        for x, y in train_dl:
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
    set_require_grad(sae, True)

def set_require_grad(module: nn.Module, value: bool):
    for param in module.parameters():
        param.requires_grad = value

def mk_token_category_dataset() -> TensorDataset:
    meta_df = pd.read_parquet("dataset/token_metadata.parquet")
    meta_df["token_idx"] = np.arange(len(meta_df))
    meta_df = meta_df.query("subcategory.notna() & token_pos >= 10")
    targets = pd.get_dummies(meta_df["subcategory"]).astype("float").values
    targets = torch.from_numpy(targets)
    return TensorDataset(
        activations[meta_df["token_idx"].values],
        targets,
    )

if __name__ == "__main__":
    train_sae(sae)
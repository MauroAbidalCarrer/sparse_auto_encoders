import os
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext

import torch
import numpy as np
from torch import nn
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
EPOCHS = 3
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
LAMBDA_KL = 1e-2          # coefficient for KL sparsity
RHO = 0.05                # target mean activation for KL sparsity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("batch size:", BATCH_SIZE)

# -------- Utility: load activations and labels --------
activations = torch.load(ACTIVATIONS_PATH, weights_only=True)  # (N_TOKENS, EMBED_DIMS)
# ensure float32
activations = activations.float()

print(f"Loaded activations, shape:", activations.shape)

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
class SAE(nn.Module):
    def __init__(self, model_embed_size: int, sparse_activation_expantion: int, dropout_ratio: float):
        super().__init__()
        sparse_activations_size = model_embed_size * sparse_activation_expantion
        self.encoder = nn.Sequential(
            # nn.LayerNorm(model_embed_size),
            # nn.Dropout(dropout_ratio),
            # nn.Linear(model_embed_size, sparse_activations_size // 2),
            # nn.Dropout(dropout_ratio),
            # nn.ReLU(inplace=True),
            # nn.Linear(sparse_activations_size // 2, sparse_activations_size),
            # nn.ReLU(inplace=True)
            nn.Linear(model_embed_size, sparse_activations_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sparse_activations_size, model_embed_size),
            # nn.Tanh(),
            # nn.ReLU(inplace=True),
            # nn.Linear(sparse_activations_size, model_embed_size)
        )

    def forward(self, x):
        saprse_activation = self.encoder(x)  # (B, bottle_dim), non-negative due to ReLU
        reconstruction = self.decoder(saprse_activation)
        return reconstruction, saprse_activation

model = torch.compile(
    SAE(
        model_embed_size=activations.shape[1],
        sparse_activation_expantion=8,
        dropout_ratio=0.2,
    )
    .to(device)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_loss = nn.MSELoss(reduction="mean")

# -------- Sparsity helper: KL divergence for Bernoulli-like activations --------
# We treat activations as >=0 (ReLU). We compute mean activation per hidden unit across batch,
# normalize by (max activation) to get pseudo-probabilities. Simpler approach: use mean(z) directly.
def kl_sparsity(z, rho, eps=1e-8):
    # z: (B, D) non-negative; convert to mean activation per-unit (averaged across batch)
    # Scale to [0,1] by dividing by (mean + small constant) OR use sigmoid. Here we use sigmoid to map to (0,1).
    p_hat = torch.clamp(z.mean(dim=0), eps, 1.0 - eps)  # mean activation per unit
    # Map p_hat to (0,1) range by using a running scaling factor. If p_hat >1, scale with sigmoid:
    p_hat = torch.sigmoid(p_hat)  
    rho_t = torch.tensor(rho, device=z.device)
    # KL divergence between Bernoulli(rho) and Bernoulli(p_hat)
    kl = rho_t * torch.log((rho_t + eps) / (p_hat + eps)) + (1 - rho_t) * torch.log(((1 - rho_t) + eps) / ((1 - p_hat) + eps))
    return kl.sum()

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

# -------- Training loop --------
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for (xb, ) in tqdm(train_loader):
        xb = xb.to(device)
        with autocast_ctx():
            reconstruction, sparse_activations = model(xb)
            loss_recon = mse_loss(reconstruction, xb)

            loss_l1 = sparse_activations.abs().mean()  # alternative: z.abs().mean() (L1 on activations)
            loss_kl = kl_sparsity(sparse_activations, RHO)

            loss = loss_recon + LAMBDA_L1 * loss_l1 #+ LAMBDA_KL * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    zs_val = []
    with torch.no_grad():
        for (xb, ) in val_loader:
            xb = xb.to(device)
            reconstruction, sparse_activations = model(xb)
            loss_recon = mse_loss(reconstruction, xb)
            loss_l1 = sparse_activations.abs().mean()
            loss_kl = kl_sparsity(sparse_activations, RHO)
            loss = loss_recon + LAMBDA_L1 * loss_l1 #+ LAMBDA_KL * loss_kl
            val_loss += loss.item() * xb.size(0)
            zs_val.append(sparse_activations.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    zs_val = np.vstack(zs_val) if len(zs_val) else np.zeros((0, BOTTLE_DIM))

    print(f"Epoch {epoch:03d} | Train loss {train_loss:.6f} | Val loss {val_loss:.6f}")

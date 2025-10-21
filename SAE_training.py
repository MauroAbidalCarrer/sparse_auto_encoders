import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
# from sklearn.metrics import roc_auc_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# -------- USER: paths to your data files --------
ACTIVATIONS_PATH = "middle_layer_activations.pt"  # shape: (N, seq_len, hidden_dim)
LABELS_PATH = "labels.npy"                        # shape: (N,) with 0/1 labels (optional)
OUT_DIR = "sae_output"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- HYPERPARAMS --------
POOL_METHOD = "last"      # options: "mean", "last", "max"
BOTTLE_DIM = 128
HIDDEN_FACTOR = 0.5       # hidden_dim * factor -> first encoder hidden
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 50
LAMBDA_L1 = 1e-4          # coefficient for L1 on latent
LAMBDA_KL = 1e-2          # coefficient for KL sparsity
RHO = 0.05                # target mean activation for KL sparsity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Utility: load activations and labels --------
if not os.path.exists(ACTIVATIONS_PATH):
    raise FileNotFoundError(f"Activation tensor not found at {ACTIVATIONS_PATH}")

activations = torch.load(ACTIVATIONS_PATH, weights_only=True)  # expected torch.Tensor
# ensure float32
activations = activations.float()

if activations.dim() != 3:
    raise ValueError("Expected activations shape (N, seq_len, hidden_dim)")

N, seq_len, hidden_dim = activations.shape
print(f"Loaded activations: N={N}, seq_len={seq_len}, hidden_dim={hidden_dim}")

# Load labels if provided, otherwise create dummy labels for examples
if os.path.exists(LABELS_PATH):
    labels = np.load(LABELS_PATH)
    assert labels.shape[0] == N
else:
    # fallback: if N==3 use small manual labels for your example prompts
    if N == 3:
        labels = np.array([0, 1, 0])  # mark second prompt (bomb) as unsafe
        print("No labels file found; using fallback labels:", labels.tolist())
    else:
        raise FileNotFoundError("No labels file found. Please provide labels.npy matching activations.")

# -------- Pool sequence tokens -> per-example vector --------
if POOL_METHOD == "mean":
    X = activations.mean(dim=1)   # (N, hidden_dim)
elif POOL_METHOD == "max":
    X, _ = activations.max(dim=1)
elif POOL_METHOD == "last":
    X = activations[:, -1, :]
else:
    raise ValueError("Unknown POOL_METHOD")

X = X.numpy()  # convert to numpy for scaler, then back to tensor
# scaler = StandardScaler()
# X = scaler.fit_transform(X)     # zero-mean, unit-variance
X = torch.from_numpy(X).float()
y = torch.from_numpy(labels).long()

# -------- Create dataset and splits --------
dataset = TensorDataset(X, y)
n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------- Model: simple SAE --------
class SAE(nn.Module):
    def __init__(self, input_dim, bottle_dim, hidden_factor=0.5):
        super().__init__()
        hid = max(4, int(input_dim * hidden_factor))
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, bottle_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottle_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)  # (B, bottle_dim), non-negative due to ReLU
        recon = self.decoder(z)
        return recon, z

model = SAE(input_dim=hidden_dim, bottle_dim=BOTTLE_DIM, hidden_factor=HIDDEN_FACTOR).to(DEVICE)
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

# -------- Training loop --------
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        recon, z = model(xb)
        loss_recon = mse_loss(recon, xb)

        loss_l1 = z.abs().mean()  # alternative: z.abs().mean() (L1 on activations)
        loss_kl = kl_sparsity(z, RHO)

        loss = loss_recon + LAMBDA_L1 * loss_l1 + LAMBDA_KL * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    zs_val = []
    ys_val = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            recon, z = model(xb)
            loss_recon = mse_loss(recon, xb)
            loss_l1 = z.abs().mean()
            loss_kl = kl_sparsity(z, RHO)
            loss = loss_recon + LAMBDA_L1 * loss_l1 + LAMBDA_KL * loss_kl
            val_loss += loss.item() * xb.size(0)
            zs_val.append(z.cpu().numpy())
            ys_val.append(yb.numpy())

    val_loss /= len(val_loader.dataset)
    zs_val = np.vstack(zs_val) if len(zs_val) else np.zeros((0, BOTTLE_DIM))
    ys_val = np.concatenate(ys_val) if len(ys_val) else np.zeros((0,))

    print(f"Epoch {epoch:03d} | Train loss {train_loss:.6f} | Val loss {val_loss:.6f}")

import os

import torch

from models import SparseAutoencoder, TopK
from SAEBench.sae_bench.evals.


N_ACTIVATIONS_DIMS = 1024
EXPANSION_RATIO = 16
N_LATENTS = N_ACTIVATIONS_DIMS * EXPANSION_RATIO
USE_NORMALIZATION = False
K = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    if not os.path.exists("models/sae.pt"):
        print("Error: no sae.pt in models.")
        exit(1)
    sae_state_dict = torch.load("models/sae.pt", weights_only=True)
    sae = (
        SparseAutoencoder(
            n_inputs=N_ACTIVATIONS_DIMS,
            n_latents=N_LATENTS,
            activation=TopK(K),
            tied=True,
            normalize=USE_NORMALIZATION,
        )
        .load_state_dict(sae_state_dict)
        .to(device)
    )
    

if __name__ == "__main__":
    main()
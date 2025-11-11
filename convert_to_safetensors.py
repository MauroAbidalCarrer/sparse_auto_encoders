import os
from tqdm import tqdm

import torch
from safetensors.torch import save_file
# import safetensors

os.makedirs("safetensors_dataset", exist_ok=True)
for pt_shard_filename in tqdm(os.listdir("dataset")):
    pt_shard_path = os.path.join("dataset", pt_shard_filename)
    shard = torch.load(pt_shard_path, weights_only=True)
    safe_tensor_filename = pt_shard_filename.split(".")[0] + ".safetensor"
    safe_tensor_path = os.path.join("safetensors_dataset", safe_tensor_filename)
    save_file({"shard": shard}, safe_tensor_path)
import os
import gc
from tqdm import tqdm
from typing import Any
from pathlib import Path
from code import interact

import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


BATCH_SIZE = 256
MODEL_ID = "roneneldan/TinyStories-1Layer-21M"
INPUT_DATASETS_CFGS = [
    {
        "id": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "input_col": "input_ids",
        "split": "train",
    },
]
CONTEXT_WINDOW = 512
LAYER_IDX_FRACTION = 3 / 4
RESID_ACT_DATASET_PATH = "dataset"
DATASET_SHARD_CACHING_FREQ = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Made specifically for Tiny stories for now, will most likely require modifications to work on most HF datasets.
def mk_dataset(
        model_id: str,
        datasets_cfgs: list[dict[str, Any]],
        context_window: int,
        batch_size: int,
        output_dir_path: str,
    ):
    """Creates a tensor of the residual activations of the input datasets and savec it in out_dir.

    Args:
        model_id (str): hugging face model id
        datasets_attrs (list[dict[str, Any]]): list of the datasets configs
        context_window (int): Context window size (number of tokens passed to the model) during the recording of the activatinos.
        batch_size (int): Pretty self explanatory.
    """
    os.makedirs(output_dir_path, exist_ok=True)
    # Load model, config and tokenizer
    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .eval()
    )
    model_config = AutoConfig.from_pretrained(model_id)
    # record residual activations
    input_dataset = load_datasets_as_df(datasets_cfgs)
    recorder = ResidualStreamRecorder(
        model,
        model_config,
        RESID_ACT_DATASET_PATH,
        DATASET_SHARD_CACHING_FREQ,
    )
    recorder.record_residual_activations(
        input_dataset,
        context_window,
        batch_size,
    )

def get_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_datasets_as_df(datasets_cfgs: list[dict[str, Any]]) -> pd.DataFrame:
    input_datasets = list(map(load_dataset_as_df, datasets_cfgs))
    input_datasets = (
        pd.concat(input_datasets)
        .reset_index(drop=True)
        .astype({
            "dataset_id": "category",
        })
        .loc[:, [
            "input_ids",
            "dataset_id",
        ]]
    )
    input_datasets["input_idx"] = np.arange(len(input_datasets))
    return input_datasets

def load_dataset_as_df(datasets_cfgs: dict[str, str]) -> pd.DataFrame:
    return (
        load_dataset(datasets_cfgs["id"], split=datasets_cfgs["split"])
        .to_pandas()
        .assign(dataset_id=datasets_cfgs["id"])
        .rename(columns={
            datasets_cfgs["input_col"]: "input_ids",
        })
        .astype({
            "dataset_id": "category",
        })
    )

class ResidualStreamRecorder:
    def __init__(self, model: AutoModelForCausalLM, model_config, output_dir: Path, dataset_shard_recording_freq: int):
        self.model = model
        self.output_dir = output_dir
        self.dataset_shard_recording_freq = dataset_shard_recording_freq
        self.batch_input_ids = None          # torch.Tensor on CPU
        self.collected_activations = []      # list of tensors [num_nonpad_tokens, hidden_dim]

        num_layers = model_config.num_hidden_layers
        recording_layer = int(num_layers * LAYER_IDX_FRACTION)
        print(f"Capturing residual-stream-before-layer (input) at layer {recording_layer} (of {num_layers}).")

        # register hook on the middle layer; we will read inputs[0] to get the residual stream
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            print("using model.layers as layer module")
            layer_modules = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            print("using transformer.h as layer module")
            layer_modules = model.transformer.h
        else:
            raise ValueError("Unknown model architecture — cannot locate transformer blocks")
        self.handle = layer_modules[recording_layer].register_forward_hook(self._residual_activations_recording_hook)

    def _residual_activations_recording_hook(self, _: nn.Module, inp: torch.Tensor, outp: torch.Tensor):
        """
        inp is a tuple of inputs to the layer; inp[0] is the residual stream entering the layer:
        hidden_states shape = (batch, seq_len, hidden_dim).
        """
        hidden_states = inp[0] if isinstance(inp, (tuple, list)) else inp
        # move to cpu and detach
        hidden_states = hidden_states.detach().cpu()  # (B, S, H)
        # B, S, H = hidden_states.shape
        # flatten batch and sequence dims to index by mask
        hidden_flat = hidden_states #.reshape(B * S, H)                # (B*S, H)
        self.collected_activations.append(hidden_flat)

    @torch.no_grad
    def record_residual_activations(
            self,
            input_dataset_df: pd.DataFrame,
            seq_len: int,
            batch_size: int,
        ):
        """
        Records the residual activations and saves them into self.collected_activations.
        Assumes the dataset to be tokenized, and that each row is a sequence of same length without any padding tokens.
        That's a lot of assumptions I know.
        """
        input_ids = np.concat(input_dataset_df["input_ids"])
        input_ids = torch.from_numpy(input_ids)
        input_ids_ds = TensorDataset(input_ids)
        input_ids_dl = DataLoader(input_ids_ds, batch_size * seq_len, drop_last=True)
        device_type = torch.device(device).type
        with torch.amp.autocast_mode.autocast(device_type, torch.bfloat16):
            batch_it = tqdm(input_ids_dl, desc="recording residual activations")
            batch_it = enumerate(batch_it)
            for batch_i, (batch_input_ids, ) in batch_it:
                batch_input_ids = (
                    batch_input_ids
                    .reshape(batch_size, seq_len)
                    .to(device)
                )
                _ = self.model(
                    input_ids=batch_input_ids,
                )
                if batch_i % self.dataset_shard_recording_freq == 0 and batch_i != 0:
                    self.save_results(batch_i // len(input_ids_dl), )

    def save_results(self, shard_index: int):
        # concatenate
        self.collected_activations = torch.cat(self.collected_activations, dim=0)
        activations_path = os.path.join(self.output_dir, f"residual_activations_shard_{shard_index}.pt")
        torch.save(self.collected_activations, activations_path)
        print(
            "activations_tensor.shape:",
            self.collected_activations.shape,
            "saved activations tensor to:",
            activations_path,
        )
        # Free up memory
        del self.collected_activations
        torch.cuda.empty_cache()
        gc.collect()
        self.collected_activations = []



if __name__ == "__main__":
    mk_dataset(
        MODEL_ID,
        INPUT_DATASETS_CFGS,
        CONTEXT_WINDOW,
        BATCH_SIZE,
        RESID_ACT_DATASET_PATH,
    )

import os
import threading
from time import time
from tqdm import tqdm
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM


BATCH_SIZE = 100
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
N_ACTIVATIONS_TO_RECORD = 4096 * 30_000
DATASET_SHARD_CACHING_FREQ = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

# Made specifically for Tiny stories for now, will most likely require modifications to work on most HF datasets.
def mk_dataset(
        model_id: str,
        datasets_cfgs: list[dict[str, Any]],
        batch_size: int,
        seq_len : int,
        dataset_shard_recording_freq: int,
        output_dir_path: str,
        n_activations_to_record: Optional[int] = None,
    ):
    """Creates a tensor of the residual activations of the input datasets and savec it in out_dir.

    Args:
        model_id (str): hugging face model id
        datasets_attrs (list[dict[str, Any]]): list of the datasets configs
        context_window (int): Context window size (number of tokens passed to the model) during the recording of the activatinos.
        batch_size (int): Pretty self explanatory.
    """
    torch.set_float32_matmul_precision('high')
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
        output_dir_path,
        dataset_shard_recording_freq,
        batch_size,
        seq_len,
    )
    recorder.record_residual_activations(input_dataset, n_activations_to_record)

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
    def __init__(self,
            model: AutoModelForCausalLM,
            model_config,
            output_dir: Path,
            dataset_shard_recording_freq: int,
            batch_size: int,
            seq_len: int,
        ):
        self.model = model
        self.output_dir = output_dir
        self.dataset_shard_recording_freq = dataset_shard_recording_freq
        self.batch_input_ids = None          # torch.Tensor on CPU
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shard_buffer = torch.empty(
            (dataset_shard_recording_freq + 1) * batch_size,
            seq_len,
            model_config.hidden_size,
            dtype=torch.float32,
        )
        self.next_index_to_store_at = 0

        num_layers = model_config.num_hidden_layers
        recording_layer = int(num_layers * LAYER_IDX_FRACTION)
        print(f"Capturing residual-stream-before-layer (input) at layer {recording_layer} (of {num_layers}).")

        # register hook on the middle layer; we will read inputs[0] to get the residual stream
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_modules = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
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
        B, S, H = hidden_states.shape
        collected_slice = slice(
            self.next_index_to_store_at,
            self.next_index_to_store_at + B,
        )
        self.shard_buffer[collected_slice] = hidden_states
        self.next_index_to_store_at += B

    @torch.no_grad
    def record_residual_activations(
        self,
        input_dataset_df: pd.DataFrame,
        n_activations_to_record: Optional[int] = None,
    ):
        """
        Records the residual activations and saves them into self.collected_activations.
        Assumes the dataset to be tokenized, and that each row is a sequence of same length without any padding tokens.
        That's a lot of assumptions I know.
        """
        input_ids = np.concat(input_dataset_df["input_ids"])
        n_activations_to_record = n_activations_to_record or len(input_ids)
        input_ids = input_ids[:n_activations_to_record]
        input_ids = torch.from_numpy(input_ids)
        input_ids_ds = TensorDataset(input_ids)
        input_ids_dl = DataLoader(
            input_ids_ds,
            self.batch_size * self.seq_len,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
        )
        device_type = torch.device(device).type
        with torch.amp.autocast_mode.autocast(device_type, torch.bfloat16):
            batch_it = tqdm(input_ids_dl, desc="recording residual activations")
            batch_it = enumerate(batch_it)
            for batch_i, (batch_input_ids, ) in batch_it:
                batch_input_ids = (
                    batch_input_ids
                    .reshape(self.batch_size, self.seq_len)
                    .to(device)
                )
                _ = self.model(
                    input_ids=batch_input_ids,
                )

                if batch_i % self.dataset_shard_recording_freq == 0 and batch_i != 0:
                    self.save_shard(batch_i // self.dataset_shard_recording_freq)

    def save_shard(self, shard_index: int):
        def _save_worker(shard: Tensor, index: int):
            print("Starting to save npy shard")
            start_time = time()
            activations_path = os.path.join(self.output_dir, f"residual_activations_shard_{index:03d}.pt")
            torch.save(shard, activations_path)
            print(
                f"[Async save] shard {index} | shape: {shard.shape} | "
                f"saved to: {activations_path} | time: {time() - start_time:.2f}s"
            )
            torch.cuda.empty_cache()
        shard_buff = (
            self.shard_buffer
            .reshape(-1, self.shard_buffer.shape[-1])
        )
        thread = threading.Thread(target=_save_worker, args=(shard_buff, shard_index), daemon=True)
        thread.start()
        self.next_index_to_store_at = 0

if __name__ == "__main__":
    mk_dataset(
        MODEL_ID,
        INPUT_DATASETS_CFGS,
        BATCH_SIZE,
        CONTEXT_WINDOW,
        DATASET_SHARD_CACHING_FREQ,
        RESID_ACT_DATASET_PATH,
        N_ACTIVATIONS_TO_RECORD
    )

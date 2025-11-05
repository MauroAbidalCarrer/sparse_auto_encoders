import os
from tqdm import tqdm
from typing import Any
from code import interact

import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


BATCH_SIZE = 64
MODEL_ID = "training_a_sparse_autoencoder.ipynb"
INPUT_DATASETS_ATTRS = [
    {
        "id": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "input_col": "input_ids",
        "split": "train",
    },
]
LAYER_IDX_FRACTION = 3 / 4
OUT_DIR = "dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"

def mk_dataset(model_id: str, datasets_attrs: list[dict[str, Any]]):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load model, config and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = get_tokenizer(model_id)
    # record residual activations
    recorder = ResidualStreamRecorder(model, config)
    input_dataset = load_datasets_as_df(datasets_attrs)
    recorder.record_residual_activations(
        recorder,
        input_dataset,
        model,
        tokenizer,
    )
    # Save results: activations + parquet metadata
    recorder.save_results(tokenizer, input_dataset)

def get_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_datasets_as_df(datasets_attrs: list[dict[str, Any]]) -> pd.DataFrame:
    input_datasets = list(map(load_dataset_as_df, datasets_attrs))
    input_datasets = (
        pd.concat(input_datasets)
        .reset_index(drop=True)
        .astype({
            "input": "string",
            "dataset_id": "category",
        })
        .loc[:, [
            "input",
            "dataset_id",
        ]]
    )
    input_datasets["input_idx"] = np.arange(len(input_datasets))
    return input_datasets

def load_dataset_as_df(dataset_attrs: dict[str, str]) -> pd.DataFrame:
    return (
        load_dataset(dataset_attrs["id"], split=dataset_attrs["split"])
        .to_pandas()
        .assign(dataset_id=dataset_attrs["id"])
        .rename(columns={
            dataset_attrs["input_col"]: "input",
        })
        .astype({
            "input": "string",
            "dataset_id": "category",
        })
    )

class ResidualStreamRecorder:
    def __init__(self, model: AutoModelForCausalLM, model_config):
        self.batch_attention_mask = None   # torch.Tensor on CPU
        self.batch_input_ids = None        # torch.Tensor on CPU
        self.input_indices = None # torch.Tensor on CPU (question idx for each token)
        self.current_token_positions = None  # torch.Tensor on CPU (pos in sequence for each token)

        # collected lists (per-batch fragments)
        self.collected_activations = []      # list of tensors [num_nonpad_tokens, hidden_dim]
        self.collected_token_ids = []        # list of 1D int tensors [num_nonpad_tokens]
        self.collected_input_idx = []        # list of 1D int tensors [num_nonpad_tokens]
        self.collected_token_pos = []        # list of 1D int tensors [num_nonpad_tokens]

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
        self.handle = layer_modules[recording_layer].register_forward_hook(self._record_residual_activations_hook)

    def _record_residual_activations_hook(self, module: nn.Module, inp: torch.Tensor, outp):
        """
        inp is a tuple of inputs to the layer; inp[0] is the residual stream entering the layer:
        hidden_states shape = (batch, seq_len, hidden_dim).
        """
        hidden_states = inp[0] if isinstance(inp, (tuple, list)) else inp
        # move to cpu and detach
        hidden_states = hidden_states.detach().cpu()  # (B, S, H)
        B, S, H = hidden_states.shape
        # flatten batch and sequence dims to index by mask
        hidden_flat = hidden_states.reshape(B * S, H)                # (B*S, H)
        tokens_flat = self.batch_input_ids.reshape(B * S)          # (B*S,)
        # import code; code.interact(local=locals())
        mask_flat = self.batch_attention_mask.reshape(B * S).bool()# (B*S,)
        input_idx_flat = self.input_indices.repeat(S).reshape(B * S)           # (B*S,)
        pos_flat = self.current_token_positions.reshape(B * S)       # (B*S,)
        # select non-padding positions (keeps order)
        nonpad_hidden = hidden_flat[mask_flat]   # (num_nonpad_tokens, H)
        nonpad_tokens = tokens_flat[mask_flat]   # (num_nonpad_tokens,)
        nonpad_input_idx = input_idx_flat[mask_flat]  # (num_nonpad_tokens,)
        nonpad_pos = pos_flat[mask_flat]         # (num_nonpad_tokens,)
        # append
        self.collected_activations.append(nonpad_hidden)
        self.collected_token_ids.append(nonpad_tokens)
        self.collected_input_idx.append(nonpad_input_idx)
        self.collected_token_pos.append(nonpad_pos)

    def save_results(self, tokenizer: AutoTokenizer, intput_dataset_df: pd.DataFrame):
        # concatenate
        self.collected_token_ids   = torch.cat(self.collected_token_ids, dim=0)    # (N_tokens,)
        self.collected_activations = torch.cat(self.collected_activations, dim=0)  # (N_tokens, H)
        self.collected_input_idx = np.concatenate(self.collected_input_idx)
        self.collected_input_idx   = torch.from_numpy(self.collected_input_idx)    # (N_tokens,)
        self.collected_token_pos   = torch.cat(self.collected_token_pos, dim=0)    # (N_tokens,)
        print("self.collected_token_ids.shape:", self.collected_token_ids.shape)
        print("activations_tensor.shape:", self.collected_activations.shape)
        # Sanity check
        # assert tokens_tensor.shape[0] == activations_tensor.shape[0] == input_idx_tensor.shape[0] == pos_tensor.shape[0]
        # Save activations tensor (efficient contiguous storage)
        activations_path = os.path.join(OUT_DIR, "middle_layer_activations.pt")
        torch.save(self.collected_activations, activations_path)
        print("Saved activations tensor to:", activations_path)
        # convert token ids -> token strings (batch conversion)
        df_meta = self.mk_token_meta_df(tokenizer, intput_dataset_df)
        # Save metadata as parquet (one row per token)
        meta_path = os.path.join(OUT_DIR, "token_metadata.parquet")
        # import code; code.interact(local=locals())
        df_meta.to_parquet(meta_path)
        print("Saved token metadata parquet to:", meta_path)
        print(df_meta)
    
    def mk_token_meta_df(self, tokenizer, dataset_dfs: pd.DataFrame) -> pd.DataFrame:
        tokens_ids_lst = self.collected_token_ids.tolist()
        tokens_str = tokenizer.convert_ids_to_tokens(tokens_ids_lst, skip_special_tokens=False)
        token_meta_df = pd.DataFrame({
            "token_id": self.collected_token_ids.numpy(),
            "token_str": tokens_str,
            "input_idx": self.collected_input_idx.numpy(),
            "token_pos": self.collected_token_pos.numpy()
        })
        # Join question-level metadata from original dataframe (by question_idx)
        # Ensure questions_df has an index that corresponds to the original position
        # If your questions_df has a default RangeIndex aligned with original dataset, join will work.
        # We'll reset to ensure index is integer position:
        token_meta_df = token_meta_df.merge(
            dataset_dfs[["input_idx", "dataset_id"]],
            on="input_idx",
            how="left"
        )
        print("token meta data frame")
        print(token_meta_df)

        return token_meta_df

    def record_residual_activations(
            self,
            input_dataset_df: pd.DataFrame,
            model: nn.Module,
            tokenizer: AutoTokenizer,
        ):
        # Load questions (including a 'subcategory' column if available)
        inputs = tokenizer(input_dataset_df["input"].tolist(), return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_len = input_ids.size(1)
        with torch.no_grad():
            for i in tqdm(range(0, len(input_dataset_df), BATCH_SIZE), desc="recording residual activations"):
                batch_slice = slice(i, min(i + BATCH_SIZE, len(input_dataset_df)))
                self.batch_input_ids = input_ids[batch_slice].to(device)
                self.batch_attention_mask = attention_mask[batch_slice].to(device)
                # prepare per-batch auxiliary tensors (on CPU) for recorder
                # question indices in the original DF
                start_idx = batch_slice.start
                stop_idx = batch_slice.stop
                B = stop_idx - start_idx
                # create question index matrix shape (B, seq_len) with stop_idx because loc is not pythonic idk why...
                self.input_indices = input_dataset_df.loc[start_idx:stop_idx - 1, "input_idx"].values
                # token positions per sequence [0..seq_len-1] shape (B, seq_len)
                self.current_token_positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(B, 1).cpu()
                # forward (hook will fire and capture the input residual stream)
                _ = model(
                    input_ids=self.batch_input_ids,
                    attention_mask=self.batch_attention_mask,
                )


if __name__ == "__main__":
    mk_dataset()

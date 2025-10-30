import os
from tqdm import tqdm

import torch
import pandas as pd
from torch import nn
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


BATCH_SIZE = 64
MODEL_ID = "openai-community/gpt2"
INPUT_DATASETS_ATTRS = [
    {
        "id": "fedric95/T2TSyntheticSafetyBench",
        "input_col": "question",
        "split": "train",
        "label_col": "subcategory",
        "input_has_label_eval_str": "input_has_label = label.notna()",
    },
    {
        "id": "szhuggingface/ag_news",
        "input_col": "text",
        "split": "train_1_48k",
        "label_col": "label",
        "input_has_label_eval_str": "input_has_label = True",
    },
]
LAYER_IDX_FRACTION = 3 / 4
OUT_DIR = "dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualStreamRecorder:
    def __init__(self, model: AutoModelForCausalLM, model_config):
        self.current_attention_mask = None   # torch.Tensor on CPU
        self.current_input_ids = None        # torch.Tensor on CPU
        self.current_question_indices = None # torch.Tensor on CPU (question idx for each token)
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
        self.handle = layer_modules[recording_layer].register_forward_hook(self.save_activation_hook)

    def save_activation_hook(self, module: nn.Module, inp, outp):
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
        tokens_flat = self.current_input_ids.reshape(B * S)          # (B*S,)
        mask_flat = self.current_attention_mask.reshape(B * S).bool()# (B*S,)
        qidx_flat = self.current_question_indices.reshape(B * S)     # (B*S,)
        pos_flat = self.current_token_positions.reshape(B * S)       # (B*S,)

        # select non-padding positions (keeps order)
        nonpad_hidden = hidden_flat[mask_flat]   # (num_nonpad_tokens, H)
        nonpad_tokens = tokens_flat[mask_flat]   # (num_nonpad_tokens,)
        nonpad_qidx = qidx_flat[mask_flat]       # (num_nonpad_tokens,)
        nonpad_pos = pos_flat[mask_flat]         # (num_nonpad_tokens,)
        
        # append
        self.collected_activations.append(nonpad_hidden)
        self.collected_token_ids.append(nonpad_tokens)
        self.collected_input_idx.append(nonpad_qidx)
        self.collected_token_pos.append(nonpad_pos)

    def save_results(self, tokenizer: AutoTokenizer, datasets_df: pd.DataFrame):
        # concatenate
        tokens_tensor = torch.cat(self.collected_token_ids, dim=0)           # (N_tokens,)
        activations_tensor = torch.cat(self.collected_activations, dim=0)    # (N_tokens, H)
        input_idx_tensor = torch.cat(self.collected_input_idx, dim=0)        # (N_tokens,)
        pos_tensor = torch.cat(self.collected_token_pos, dim=0)              # (N_tokens,)
        print("tokens_tensor.shape:", tokens_tensor.shape)
        print("activations_tensor.shape:", activations_tensor.shape)
        # Sanity check
        assert tokens_tensor.shape[0] == activations_tensor.shape[0] == input_idx_tensor.shape[0] == pos_tensor.shape[0]
        # Save activations tensor (efficient contiguous storage)
        activations_path = os.path.join(OUT_DIR, "middle_layer_activations.pt")
        torch.save(activations_tensor, activations_path)
        print("Saved activations tensor to:", activations_path)
        # convert token ids -> token strings (batch conversion)
        token_id_list = tokens_tensor.tolist()
        token_strs = tokenizer.convert_ids_to_tokens(token_id_list, skip_special_tokens=False)
        df_meta = self.mk_token_meta_df(datasets_df)
        # Save metadata as parquet (one row per token)
        meta_path = os.path.join(OUT_DIR, "token_metadata.parquet")
        # import code; code.interact(local=locals())
        df_meta.to_parquet(meta_path)
        print("Saved token metadata parquet to:", meta_path)
        print(df_meta)
    
    def mk_token_meta_df(self, dataset_dfs: pd.DataFrame) -> pd.DataFrame:
        # Build pandas DataFrame: one row per token
        df_meta = pd.DataFrame({
            "activation_idx": range(len(token_id_list)),  # index into activations_tensor
            "token_id": token_id_list,
            "token_str": token_strs,
            "input_idx": input_idx_tensor.tolist(),
            "token_pos": pos_tensor.tolist()
        })

        # Join question-level metadata from original dataframe (by question_idx)
        # Ensure questions_df has an index that corresponds to the original position
        # If your questions_df has a default RangeIndex aligned with original dataset, join will work.
        # We'll reset to ensure index is integer position:
        df_meta = df_meta.merge(
            dataset_dfs[["input_idx", "label", "input"]],
            on="input_idx",
            how="left"
        ).astype({"label": "category"})
        


def record_residual_activations(
        recorder: ResidualStreamRecorder,
        dataset_df: pd.DataFrame,
        dataset_id: str,
        model: nn.Module,
        tokenizer: AutoTokenizer,
    ):
    # Load questions (including a 'subcategory' column if available)
    inputs = tokenizer(dataset_df["input"].tolist(), return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = input_ids.size(1)
    with torch.no_grad():
        tqdm_desc = "recording residual activations of " + dataset_id.split("/")[-1]
        for i in tqdm(range(0, len(dataset_df), BATCH_SIZE), desc=tqdm_desc):
            batch_slice = slice(i, min(i + BATCH_SIZE, len(dataset_df)))
            batch_input_ids = input_ids[batch_slice].to(device)
            batch_attn = attention_mask[batch_slice].to(device)
            batch = {"input_ids": batch_input_ids, "attention_mask": batch_attn}
            print(batch_input_ids.shape)
            # prepare per-batch auxiliary tensors (on CPU) for recorder
            # question indices in the original DF
            start_idx = batch_slice.start
            stop_idx = batch_slice.stop
            B = stop_idx - start_idx
            # create question index matrix shape (B, seq_len)
            q_indices = torch.arange(start_idx, stop_idx, dtype=torch.long).unsqueeze(1).repeat(1, seq_len).cpu()
            # token positions per sequence [0..seq_len-1] shape (B, seq_len)
            pos_mat = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(B, 1).cpu()
            # Set recorder internals
            recorder.current_input_ids = batch_input_ids.detach().cpu()
            recorder.current_attention_mask = batch_attn.detach().cpu()
            recorder.current_question_indices = q_indices
            recorder.current_token_positions = pos_mat
            # forward (hook will fire and capture the input residual stream)
            _ = model(**batch)

def get_set_up_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_dataset_as_df(dataset_attrs: dict[str, str]) -> pd.DataFrame:
    return (
        load_dataset(dataset_attrs["id"], split=dataset_attrs["split"])
        .to_pandas()
        .assign(dataset_id=dataset_attrs["id"])
        .rename(columns={
            dataset_attrs["label_col"]: "label",
            dataset_attrs["input_col"]: "input",
        })
        .astype({
            "input": "string",
            "label": "category"
        })
        .eval(dataset_attrs["input_has_label_eval_str"])
        .reset_index(names="input_idx")
    )

def mk_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load model, config and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.eval()
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = get_set_up_tokenizer(MODEL_ID)
    # record residual activations
    recorder = ResidualStreamRecorder(model, config)
    dataset_dfs = []
    for dataset_attrs in INPUT_DATASETS_ATTRS:
        dataset_df = load_dataset_as_df(dataset_attrs)
        record_residual_activations(
            recorder,
            dataset_df,
            dataset_attrs["id"],
            model,
            tokenizer,
        )
        dataset_dfs.append(dataset_df)
    # Concat and reset the label as category after 
    dataset_dfs = pd.concat(dataset_dfs).astype({"label": "category"})
    # Save results: activations + parquet metadata
    recorder.save_results(tokenizer, dataset_dfs)


if __name__ == "__main__":
    mk_dataset()

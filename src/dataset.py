import os
from tqdm import tqdm

import torch
from torch import nn, Tensor
import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


BATCH_SIZE = 64
MODEL_ID = "openai-community/gpt2"
HARMFUL_QUESTIONS_DATASET = "fedric95/T2TSyntheticSafetyBench"
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

class ActivationsRecorder:
    def __init__(self, model: AutoModelForCausalLM, model_config):
        self.current_attention_mask = None   # torch.Tensor on CPU
        self.current_input_ids = None        # torch.Tensor on CPU
        self.current_question_indices = None # torch.Tensor on CPU (question idx for each token)
        self.current_token_positions = None  # torch.Tensor on CPU (pos in sequence for each token)

        # collected lists (per-batch fragments)
        self.collected_activations = []      # list of tensors [num_nonpad_tokens, hidden_dim]
        self.collected_token_ids = []        # list of 1D int tensors [num_nonpad_tokens]
        self.collected_question_idx = []     # list of 1D int tensors [num_nonpad_tokens]
        self.collected_token_pos = []        # list of 1D int tensors [num_nonpad_tokens]

        num_layers = model_config.num_hidden_layers
        middle_layer = num_layers // 2
        print(f"Capturing residual-stream-before-layer (input) at layer {middle_layer} (of {num_layers}).")

        # register hook on the middle layer; we will read inputs[0] to get the residual stream
        self.handle = (
            model
            .model
            .layers[middle_layer]
            .register_forward_hook(self.save_activation_hook)
        )

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
        self.collected_question_idx.append(nonpad_qidx)
        self.collected_token_pos.append(nonpad_pos)

    def save_results(self, tokenizer, questions_df):
        # concatenate
        tokens_tensor = torch.cat(self.collected_token_ids, dim=0)           # (N_tokens,)
        activations_tensor = torch.cat(self.collected_activations, dim=0)    # (N_tokens, H)
        qidx_tensor = torch.cat(self.collected_question_idx, dim=0)          # (N_tokens,)
        pos_tensor = torch.cat(self.collected_token_pos, dim=0)              # (N_tokens,)

        print("tokens_tensor.shape:", tokens_tensor.shape)
        print("activations_tensor.shape:", activations_tensor.shape)

        # Sanity check
        assert tokens_tensor.shape[0] == activations_tensor.shape[0] == qidx_tensor.shape[0] == pos_tensor.shape[0]

        # Save activations tensor (efficient contiguous storage)
        activations_path = os.path.join(OUT_DIR, "middle_layer_activations.pt")
        torch.save(activations_tensor, activations_path)
        print("Saved activations tensor to:", activations_path)

        # convert token ids -> token strings (batch conversion)
        token_id_list = tokens_tensor.tolist()
        token_strs = tokenizer.convert_ids_to_tokens(token_id_list, skip_special_tokens=False)

        # Build pandas DataFrame: one row per token
        df_meta = pd.DataFrame({
            "activation_idx": range(len(token_id_list)),  # index into activations_tensor
            "token_id": token_id_list,
            "token_str": token_strs,
            "question_idx": qidx_tensor.tolist(),
            "token_pos": pos_tensor.tolist()
        })

        # Join question-level metadata from original dataframe (by question_idx)
        # Ensure questions_df has an index that corresponds to the original position
        # If your questions_df has a default RangeIndex aligned with original dataset, join will work.
        # We'll reset to ensure index is integer position:
        questions_df_reset = questions_df.reset_index(drop=False).rename(columns={"index": "question_idx"})
        # map question-level columns you want (example: 'subcategory' and 'question' text)
        if "subcategory" in questions_df_reset.columns:
            df_meta = df_meta.merge(
                questions_df_reset[["question_idx", "subcategory", "question"]],
                on="question_idx",
                how="left"
            )
        else:
            # still add the question text for convenience
            df_meta = df_meta.merge(
                questions_df_reset[["question_idx", "question"]],
                on="question_idx",
                how="left"
            )

        # Save metadata as parquet (one row per token)
        meta_path = os.path.join(OUT_DIR, "token_metadata.parquet")
        df_meta.to_parquet(meta_path)
        print("Saved token metadata parquet to:", meta_path)
        print(df_meta)

        # remove hook
        self.handle.remove()


def main():
    # Load questions (including a 'subcategory' column if available)
    ds = load_dataset(HARMFUL_QUESTIONS_DATASET, split="train")
    df = ds.to_pandas().astype("string")

    # Ensure we have a 'question' column
    if "question" not in df.columns:
        raise RuntimeError("Dataset has no 'question' column")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, legacy=False)

    # Load model (auto device mapping)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    config = AutoConfig.from_pretrained(MODEL_ID)

    recorder = ActivationsRecorder(model, config)

    # Tokenize all questions once (CPU tensors)
    inputs = tokenizer(df["question"].tolist(), return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = input_ids.size(1)

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Forwarding batches"):
            batch_slice = slice(i, min(i + BATCH_SIZE, len(df)))
            batch_input_ids = input_ids[batch_slice].to(device)
            batch_attn = attention_mask[batch_slice].to(device)
            batch = {"input_ids": batch_input_ids, "attention_mask": batch_attn}

            # prepare per-batch auxiliary tensors (on CPU) for recorder
            # question indices in the original DF
            start_idx = batch_slice.start
            stop_idx = batch_slice.stop
            B = stop_idx - start_idx
            # create question index matrix shape (B, seq_len)
            q_indices = torch.arange(start_idx, stop_idx, dtype=torch.long).unsqueeze(1).repeat(1, seq_len).cpu()
            # token positions per sequence [0..seq_len-1] shape (B, seq_len)
            pos_mat = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(B, 1).cpu()

            recorder.current_input_ids = batch_input_ids.detach().cpu()
            recorder.current_attention_mask = batch_attn.detach().cpu()
            recorder.current_question_indices = q_indices
            recorder.current_token_positions = pos_mat

            # forward (hook will fire and capture the input residual stream)
            _ = model(**batch)

    # Save results: activations + parquet metadata
    recorder.save_results(tokenizer, df)


if __name__ == "__main__":
    main()

import os
from tqdm import tqdm

import torch
from torch import nn, Tensor
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel


BATCH_SIZE = 64
MODEL_ID = "QuixiAI/WizardLM-13B-Uncensored"
HARMFULL_QUESTIONS_DATASET = "fedric95/T2TSyntheticSafetyBench"

class ActivationsRecorder:
    def __init__(self, model: AutoModel, model_config):
        self.current_attention_mask = None   # torch.Tensor on CPU
        self.current_input_ids = None        # torch.Tensor on CPU
        self.collected_activations = []      # list of tensors [num_nonpad_tokens, hidden_dim]
        self.collected_tokens = []           # list of 1D token tensors [num_nonpad_tokens]
        num_layers = model_config.num_hidden_layers
        middle_layer = num_layers // 2
        print(f"Capturing activations after layer {middle_layer} (of {num_layers}).")
        self.handle = (
            model
            .model
            .layers[middle_layer]
            .register_forward_hook(self.save_activation_hook)
        )

    def save_activation_hook(self, module: nn.Module, inp, outp):
        """
        outp can be a single tensor or tuple; the hidden states are typically outp[0].
        We assume hidden_states shape = (batch, seq_len, hidden_dim).
        """
        # Get hidden states tensor robustly:
        hidden_states = outp[0] if isinstance(outp, (tuple, list)) else outp
        # Move to CPU and detach
        hidden_states = hidden_states.detach().cpu()  # (B, S, H)

        B, S, H = hidden_states.shape
        # flatten batch and sequence dims to index by mask
        hidden_flat = hidden_states.reshape(B * S, H)        # (B*S, H)
        tokens_flat = self.current_input_ids.reshape(B * S)           # (B*S,)
        mask_flat = self.current_attention_mask.reshape(B * S).bool()      # (B*S,)

        # Select non-padding positions in the same order as flattening
        nonpad_hidden = hidden_flat[mask_flat]               # (num_nonpad_tokens, H)
        nonpad_tokens = tokens_flat[mask_flat]               # (num_nonpad_tokens,)
        self.collected_activations.append(nonpad_hidden)
        self.collected_tokens.append(nonpad_tokens)

    def save_results(self):
        tokens_tensor = torch.cat(self.collected_tokens, dim=0)            # shape: (total_nonpad_tokens,)
        activations_tensor = torch.cat(self.collected_activations, dim=0)  # shape: (total_nonpad_tokens, hidden_dim)
        print("tokens_tensor.shape:", tokens_tensor.shape)
        print("activations_tensor.shape:", activations_tensor.shape)
        # Sanity check: same first N tokens <-> activations
        assert tokens_tensor.shape[0] == activations_tensor.shape[0], "Number of tokens must equal number of activations"
        # Save
        os.makedirs("dataset", exist_ok=True)
        torch.save(activations_tensor, "dataset/middle_layer_activations_no_pad.pt")
        torch.save(tokens_tensor, "dataset/middle_layer_tokens_no_pad.pt")
        print("Saved activations and tokens (padding removed).")

def main():
    # Load questions
    df = (
        load_dataset(HARMFULL_QUESTIONS_DATASET, split="train")
        .to_pandas()
        .loc[:, ["question"]]
        .astype("string")
    )
    questions = df["question"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, legacy=False)
    # Load model (already on GPU if possible)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Locate middle layer index
    config = AutoConfig.from_pretrained(MODEL_ID)

    # Register hook on the middle layer (adjust path if model submodule differs)
    recorder = ActivationsRecorder(model, config)

    # Tokenize everything once (CPU tensors)
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Forwarding batches"):
            batch_slice = slice(i, i + BATCH_SIZE)
            # Prepare batch on device
            batch_input_ids = input_ids[batch_slice].to(device)
            batch_attn = attention_mask[batch_slice].to(device)
            batch = {"input_ids": batch_input_ids, "attention_mask": batch_attn}
            # Set globals for hook to read (BUT move copies to CPU to avoid device sync issues)
            recorder.current_input_ids = batch_input_ids.detach().cpu()
            recorder.current_attention_mask = batch_attn.detach().cpu()
            _ = model(**batch)

    recorder.save_results()
    
if __name__ == "__main__":
    main()
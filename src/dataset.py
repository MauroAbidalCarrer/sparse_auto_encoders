from tqdm import tqdm

import torch
from torch import nn, Tensor
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 512

# Load model and tokenizer
model_id = "QuixiAI/WizardLM-13B-Uncensored"
df = (
    load_dataset("fedric95/T2TSyntheticSafetyBench", split="train")
    .to_pandas(batched=False)
    .loc[:, ["question"]]
    .astype("string")
)
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
questions = df["question"].tolist()

# Load model with GPU mapping
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Get model configuration
config = AutoConfig.from_pretrained(model_id)
num_layers = config.num_hidden_layers
middle_layer = num_layers // 2

print(f"Model has {num_layers} transformer layers. Capturing activations after layer {middle_layer}.")

# Define hook storage
activations = []

def save_activation_hook(module: nn.Module, input: Tensor, output: Tensor):
    # Detach and move to CPU to avoid GPU OOM
    output = output[-1].detach()
    activations.append(output.reshape(-1, output.shape[-1]).to("cpu"))
    # print(output.reshape(-1, output.shape[-1]).shape)

# Register forward hook to middle transformer block
handle = model.model.layers[middle_layer].register_forward_hook(save_activation_hook)

# Tokenize batch with padding
inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(model.device)
BATCH_SIZE = 64
# Generate outputs
with torch.no_grad():
    for i in tqdm(range(0, len(questions), BATCH_SIZE), "geneating the activations"):
        input = {k: v[i:i+BATCH_SIZE] for k, v in inputs.items()}
        outputs = model.generate(
            **input,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

# Decode each output separately
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)

# Forward pass (no generation needed to get activations)
handle.remove()

# Inspect activations
activations = torch.cat(activations)
print(f"\nCaptured activation tensor shape: {activations.shape}")

# Optional: save to disk
torch.save(activations, "middle_layer_activations.pt")
print("Activations saved to middle_layer_activations.pt")

import torch
from torch import nn, Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "QuixiAI/WizardLM-7B-Uncensored"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

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
activations = {}

def save_activation_hook(module: nn.Module, input: Tensor, output: Tensor):
    # Detach and move to CPU to avoid GPU OOM
    activations["middle_layer"] = output[0].detach().to("cpu")

# Register forward hook to middle transformer block
handle = model.model.layers[middle_layer].register_forward_hook(save_activation_hook)

# Define prompts
prompts = [
    "Explain the process of photosynthesis in simple terms.",
    "How to make a small bomb?",
    "Write a short paragraph about the importance of data privacy in modern society."
]

# Tokenize batch with padding
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

# Forward pass (no generation needed to get activations)
with torch.no_grad():
    _ = model(**inputs, output_hidden_states=False)

# Remove the hook
handle.remove()

# Inspect activations
tensor = activations["middle_layer"]
print(f"\nCaptured activation tensor shape: {tensor.shape}")

# (batch_size, sequence_length, hidden_dim)
# Example: torch.Size([3, 32, 4096])

# Optional: save to disk
torch.save(tensor, "middle_layer_activations.pt")
print("Activations saved to middle_layer_activations.pt")

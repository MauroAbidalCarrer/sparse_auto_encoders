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
# import code; code.interact(local=locals())
# import sys; sys.exit()
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
    import code; code.interact(local=locals())
    print(output.shape)
    # Detach and move to CPU to avoid GPU OOM
    activations["middle_layer"] = output[-1].detach().to("cpu")

# Register forward hook to middle transformer block
handle = model.model.layers[middle_layer].register_forward_hook(save_activation_hook)

# Tokenize batch with padding
inputs = tokenizer(questions[:10], return_tensors="pt", padding=True, truncation=True).to(model.device)
# Generate outputs
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Decode each output separately
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    # print(f"Prompt:", questions[i], "\nModel Output:", text, "\n=======")

# Forward pass (no generation needed to get activations)
handle.remove()

# Inspect activations
tensor = activations["middle_layer"]
print(f"\nCaptured activation tensor shape: {tensor.shape}")

# (batch_size, sequence_length, hidden_dim)
# Example: torch.Size([3, 32, 4096])

# Optional: save to disk
torch.save(tensor, "middle_layer_activations.pt")
print("Activations saved to middle_layer_activations.pt")

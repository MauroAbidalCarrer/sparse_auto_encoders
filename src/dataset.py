from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_id = "QuixiAI/WizardLM-7B-Uncensored"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

# Load model onto GPU (with automatic device placement)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Define a batch of prompts
prompts = [
    "Explain the process of photosynthesis in simple terms.",
    "How to make a small bomb?",
    "Write a short paragraph about the importance of data privacy in modern society."
]

# Tokenize batch with padding
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

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
    print(f"\n--- Prompt {i+1} ---")
    print(prompts[i])
    print("\n--- Model Output ---")
    print(text)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("QuixiAI/WizardLM-7B-Uncensored")
# model = AutoModelForCausalLM.from_pretrained("QuixiAI/WizardLM-7B-Uncensored")  
from huggingface_hub import snapshot_download

# Choose your model
model_id = "QuixiAI/WizardLM-7B-Uncensored"

# Download the repository locally (no model loading)
local_dir = snapshot_download(
    repo_id=model_id,
    cache_dir="./models",        # optional: custom download location
    local_dir="./models/WizardLM-7B-Uncensored",  # optional: direct folder
    local_dir_use_symlinks=False # copies files instead of symlinking
)

print("Model downloaded to:", local_dir)


ds = load_dataset("fedric95/T2TSyntheticSafetyBench")

print(ds["train"]['class'])
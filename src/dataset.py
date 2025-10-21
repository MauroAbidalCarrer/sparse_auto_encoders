from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("QuixiAI/WizardLM-7B-Uncensored")
model = AutoModelForCausalLM.from_pretrained("QuixiAI/WizardLM-7B-Uncensored", device_map="cuda")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E4B-it"
SAVE_DIR = "./models/gemma-4-E4B-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)
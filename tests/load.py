from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAVE_DIR = "./models/Qwentts"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

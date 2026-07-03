from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/gemma-4-E4B-it")
model = AutoModelForCausalLM.from_pretrained("./models/gemma-4-E4B-it")

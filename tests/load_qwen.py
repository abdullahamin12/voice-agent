from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAVE_DIR = "./models/Qwentts"

print(f"Downloading {MODEL_ID} directly to {SAVE_DIR}...")

# This downloads the raw weights directly to your folder
snapshot_download(repo_id=MODEL_ID, local_dir=SAVE_DIR)

print(f"Model successfully saved to {SAVE_DIR}!")

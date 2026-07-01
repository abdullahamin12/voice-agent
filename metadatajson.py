import json
from pathlib import Path

MODEL_DIR = Path("./models/gemma-4-E4B-it")

metadata = {
    "model_id": "google/gemma-4-E4B-it",
    "local_path": str(MODEL_DIR),
    "precision": "default",
    "purpose": "voice agent brain",
    "notes": "Saved locally for offline loading",
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

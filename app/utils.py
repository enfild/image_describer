import json
from datetime import datetime, timezone
from pathlib import Path
from config import Config

def save_upload(file_bytes: bytes, suffix: str = ".jpg") -> str:
    uploads_dir = Config.RESULTS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")
    filename = f"{ts}{suffix}"
    path = uploads_dir / filename
    with open(path, "wb") as f:
        f.write(file_bytes)
    return str(Path("uploads") / filename)

def save_result(image_path, model, captions):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "model": model,
        "captions": captions,
    }
    file_path = Config.RESULTS_DIR / f"{datetime.now(timezone.utc).date()}.json"
    data = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(entry)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return entry

def load_history():
    all_entries = []
    for file in Config.RESULTS_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            all_entries.extend(json.load(f))
    return all_entries

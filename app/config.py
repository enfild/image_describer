import os
from pathlib import Path

class Config:
    LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    BASE_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    NUM_CAPTIONS = 3
    MAX_TOKENS = 30
    TEMPERATURE = 0.9

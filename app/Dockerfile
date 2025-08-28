FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    curl \
    libgl1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip==24.2

RUN pip3 install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# HuggingFace + utils
RUN pip3 install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.0 \
    git+https://github.com/huggingface/transformers.git \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.5" \
    "tokenizers>=0.20.0" \
    "pillow>=10.3.0" \
    "requests>=2.32.3" \
    "numpy>=1.26.4" \
    "sentencepiece>=0.1.99" \
    "timm>=0.9.16" \
    "einops>=0.7.0" \
    "protobuf>=4.25.3" \
    "huggingface_hub>=0.23.4" \
    hf_transfer

# --- Models downloading ---
RUN python3 - <<EOF
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoProcessor, AutoModel

# BLIP
BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# ViT-GPT2
VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# LLaVA (must have - trust_remote_code=True)
# AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", trust_remote_code=True, use_fast=False)
# AutoModel.from_pretrained("llava-hf/llava-1.5-7b-hf", trust_remote_code=True)
EOF
# ---------------------------------------------------

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

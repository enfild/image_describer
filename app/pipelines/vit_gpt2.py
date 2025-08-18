from typing import List
from pathlib import Path
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class ViTGPT2Pipeline:
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        num_return_sequences: int = 3,
        max_new_tokens: int = 40,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

        outputs = self.model.generate(pixel_values, **gen_kwargs)
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        cleaned, seen = [], set()
        for c in captions:
            c = c.strip()
            if c and not c.endswith(('.', '!', '?')):
                c += '.'
            if c not in seen:
                seen.add(c)
                cleaned.append(c)
        return cleaned[:num_return_sequences] or ["A photo."]

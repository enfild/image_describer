from typing import List
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPPipeline:

    # blip-image-captioning-large
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)
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
        image = Image.open(image_path).convert("RGB")
        # text = "a photography of"
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
    
        if num_return_sequences > 1:
            pixel_values = pixel_values.repeat(num_return_sequences, 1, 1, 1).clone()
    
        outputs = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )
    
        texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        cleaned, seen = [], set()
        for t in texts:
            t = t.strip().strip(". ").capitalize()
            if t and not t.endswith(('.', '!', '?')):
                t += '.'
            if t and t not in seen:
                seen.add(t)
                cleaned.append(t)
        if not cleaned:
            cleaned = ["A photo."]
        return cleaned[:num_return_sequences]
    
    
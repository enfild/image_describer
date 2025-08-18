from typing import List
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel


class LLavaPipeline:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        device_map: str | None = "auto",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        if device_map is None:
            self.model.to(self.device)
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
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")

        if hasattr(self.processor, "apply_chat_template"):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image in one short sentence."},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = "USER: <image>\nDescribe the image in one short sentence.\nASSISTANT:"

        texts = [prompt] * num_return_sequences
        images = [image] * num_return_sequences

        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id is not None
                else self.processor.tokenizer.eos_token_id,
        )
        if "image_sizes" in inputs:
            gen_kwargs["image_sizes"] = inputs["image_sizes"]

        outputs = self.model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            **gen_kwargs,
        )

        input_len = inputs["input_ids"].shape[1]
        decoded = []
        for i in range(outputs.size(0)):
            seq = outputs[i][input_len:]
            text = self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            decoded.append(text)

        seen, cleaned = set(), []
        for t in decoded:
            if t and t not in seen:
                seen.add(t)
                cleaned.append(t)
        return cleaned[:num_return_sequences] or ["A photo."]

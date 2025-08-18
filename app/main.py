# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from models import CaptionResponse, HistoryResponse
from utils import save_result, load_history, save_upload
from config import Config
from typing import Optional

import torch

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---- Lazy singletons ----
_blip_pipeline = None
_vitgpt2_pipeline = None
_llava_pipeline = None


def _kill_others(keep: str):
    global _blip_pipeline, _vitgpt2_pipeline, _llava_pipeline

    if keep != "blip" and _blip_pipeline is not None:
        try:
            del _blip_pipeline
        except Exception:
            pass
        _blip_pipeline = None

    if keep != "vitgpt2" and _vitgpt2_pipeline is not None:
        try:
            del _vitgpt2_pipeline
        except Exception:
            pass
        _vitgpt2_pipeline = None

    if keep != "llava" and _llava_pipeline is not None:
        try:
            del _llava_pipeline
        except Exception:
            pass
        _llava_pipeline = None

    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_blip_pipeline():
    global _blip_pipeline
    if _blip_pipeline is None:
        from pipelines.blip import BLIPPipeline
        _blip_pipeline = BLIPPipeline()
        _kill_others("blip")
    return _blip_pipeline


def get_vitgpt2_pipeline():
    global _vitgpt2_pipeline
    if _vitgpt2_pipeline is None:
        from pipelines.vit_gpt2 import ViTGPT2Pipeline
        model_name = getattr(Config, "VITGPT2_MODEL_NAME", "nlpconnect/vit-gpt2-image-captioning")
        _vitgpt2_pipeline = ViTGPT2Pipeline(model_name=model_name)
        _kill_others("vitgpt2")
    return _vitgpt2_pipeline


def get_llava_pipeline():
    global _llava_pipeline
    if _llava_pipeline is None:
        from pipelines.llava import LLavaPipeline
        model_name = getattr(Config, "LLAVA_MODEL_NAME", "llava-hf/llava-1.5-7b-hf")
        _llava_pipeline = LLavaPipeline(model_name=model_name)
        _kill_others("llava")
    return _llava_pipeline


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(...),
    model: str = Form(...),
    num_captions: Optional[int] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None)):
    try:
        file_bytes = await file.read()
        suffix = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ".jpg"
        image_rel_path = save_upload(file_bytes, suffix=suffix)
        abs_path = str((Config.RESULTS_DIR / image_rel_path).resolve())

        m = model.lower().strip()

        if m == "blip":
            captions = get_blip_pipeline().generate(
                abs_path,
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        elif m == "vitgpt2":
            captions = get_vitgpt2_pipeline().generate(
                abs_path,
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        elif m == "llava":
            captions = get_llava_pipeline().generate(
                abs_path,
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        else:
            raise HTTPException(status_code=400, detail="Unknown model (use: blip | vitgpt2 | llava)")

        result = save_result(image_rel_path, m, captions)
        return CaptionResponse(**result)

    except Exception as e:
        print(f"[caption] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse)
def history():
    return HistoryResponse(entries=load_history())


@app.get("/image/{rel_path:path}")
def get_image(rel_path: str):
    base = Config.RESULTS_DIR.resolve()
    target = (base / rel_path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(target)

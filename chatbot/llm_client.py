from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from config import FLAN_MODEL

_tokenizer = None
_model = None
_pipeline = None

def _ensure_model():
    global _tokenizer, _model, _pipeline
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL)
        # device=-1 => CPU. If ai GPU, schimba device=0
        _pipeline = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=-1)
    return _pipeline

def generate_answer(prompt: str, max_length=256):
    pipe = _ensure_model()
    outputs = pipe(prompt, max_length=max_length, do_sample=False)
    if isinstance(outputs, list) and outputs:
        return outputs[0].get('generated_text', '')
    return ""

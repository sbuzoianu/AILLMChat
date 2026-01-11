from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from config import FLAN_MODEL   # flan-t5-small folosit pentru a genera răspunsuri la diverse întrebări sau comenzi (prompt-uri).

_tokenizer = None # "Lazy Loading" - nu încarci în memorie de fiecare dată când rulezi o funcție, ci le încarci o singură dată și să le refolosești.
_model = None
_pipeline = None

# Verifică dacă modelul este deja încărcat. Dacă nu, îl descarcă și îl configurează; dacă da, îl returnează direct pe cel existent.
def _ensure_model(): 
    global _tokenizer, _model, _pipeline
    if _pipeline is None:
        _tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL)
        # device=-1 => CPU. Daca ai GPU ->  schimba device=0
        # pipeline: Este o abstractizare care simplifică procesul de procesare a textului. Acesta face automat trei pași:
        # Tokenizare: Transformă textul uman în numere pe care modelul le înțelege.
        # Inferență: Trece numerele prin modelul matematic pentru a genera un răspuns.
        # Post-procesare: Transformă numerele generate înapoi în text lizibil.
        _pipeline = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=-1)
    return _pipeline

def generate_answer(prompt: str, max_length=256):
    pipe = _ensure_model()
    outputs = pipe(prompt, max_length=max_length, do_sample=False)
    if isinstance(outputs, list) and outputs:
        return outputs[0].get('generated_text', '')
    return ""

# max_length=256: Limitează cât de lung poate fi răspunsul generat, pentru a preveni consumul excesiv de resurse sau răspunsurile care se repetă la infinit.
# do_sample=False: Aceasta activează "Greedy Search". Înseamnă că modelul va alege mereu cel mai probabil cuvânt următor. Rezultatul este determinist (vei primi același răspuns la același prompt), fiind ideal pentru sarcini de logică sau extragere de date.
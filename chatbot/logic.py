from knowledge.service import get_knowledge
from knowledge.embeddings import ensure_index, search_semantic
from .llm_client import generate_answer

# Asigurăm indexul la pornire
ensure_index()


def normalize_ro_diacritics(text: str) -> str:
    if not text:
        return text
    return (text
            .replace("ş", "ș").replace("Ş", "Ș")
            .replace("ţ", "ț").replace("Ţ", "Ț"))


def reply_to_user(question, grade, subject):
    kb_exact = None
    if subject and grade:
        kb_exact = get_knowledge(subject, grade)

    context_parts = []
    if kb_exact:
        context_parts.append(kb_exact)


    sem_results = search_semantic(question, 10, subject, grade)
    print("KB exact:", kb_exact[:200] if kb_exact else None)
    print("SEM sample:", sem_results[0]['content'][:200] if sem_results else None)

    for r in sem_results:
        # evităm duplicate identice
        if r['content'] not in context_parts:
            context_parts.append(r['content'])

    if not context_parts:
        return "Nu pot genera un răspuns."

    context = "\n".join(context_parts)
    prompt = f"""Ești un asistent educațional.
    Răspunde în limba română cu diacritice corecte (ă â î ș ț).
    REGULI:1) Folosește DOAR informațiile din CONTEXT.
    2) Dacă găsești răspunsul în context, COPIAZĂ propozițiile relevante cât mai fidel (fără reformulări).
    3) Dacă nu există informația, răspunde exact: Nu pot genera un răspuns.
    CONTEXT:{context}
    ÎNTREBARE:{question}
    RĂSPUNS:"""

    ans = generate_answer(prompt).strip()
    ans = normalize_ro_diacritics(ans)
    ans = ans.replace(". ", "\n")
    print("=== ANSWER SENT TO UI ===")
    print(ans)
    print("=========================")

    return ans

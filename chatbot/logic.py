from knowledge.service import get_knowledge
from knowledge.embeddings import ensure_index, search_semantic
from .llm_client import generate_answer

# Asigurăm indexul la pornire
ensure_index()

def extract_subject_and_grade(question):
    subjects = ["fizica", "matematica", "chimie", "biologie"]
    grades = ["V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

    q = question.lower()
    subject = next((s for s in subjects if s in q), None)
    grade = next((g for g in grades if g.lower() in q or g in question), None)
    return subject, grade

def reply_to_user(question):
    subject, grade = extract_subject_and_grade(question)

#Căutare Exactă (kb_exact): Dacă a găsit materia și clasa, merge direct la fișierul/tabelul corespunzător (ex: tot ce ai salvat despre Fizică, clasa a IX-a).
    kb_exact = None
    if subject and grade:
        kb_exact = get_knowledge(subject, grade)

    context_parts = []
    if kb_exact:
        context_parts.append(kb_exact)

# Căutare Semantică (search_semantic): Aceasta este partea „inteligentă”. 
# Chiar dacă nu există o potrivire exactă de cuvinte, sistemul caută fragmente de text care au un înțeles similar cu întrebarea.
# Exemplu: Dacă întrebi de "viteză", căutarea semantică ar putea găsi fragmente despre "deplasare" sau "cinematică".
    sem_results = search_semantic(question, top_k=3)
    for r in sem_results:
        # evităm duplicate identice
        if r['content'] not in context_parts:
            context_parts.append(r['content'])


# Construirea Promptului (Instrucțiunile pentru AI)
# Dacă sistemul găsește informații relevante (context), el construiește un mesaj special pentru modelul AI:
# "Ești un profesor... Folosește EXCLUSIV informațiile de mai jos... 
        # Context: {aici se pune textul găsit}... 
        # Întrebare: {întrebarea elevului}"
# Această tehnică se numește Grounding. Ea previne „halucinațiile” (momentele când AI-ul inventează lucruri), forțându-l să rămână fidel materialelor tale didactice.
    if context_parts:
        context = "\n---\n".join(context_parts)
        prompt = (
            "Ești un profesor care explică pe înțelesul elevilor. Folosește EXCLUSIV informațiile de mai jos "
            "pentru a răspunde la întrebare.\n\n"
            f"Context:\n{context}\n\n"
            f"Întrebare: {question}\n\n"
            "Răspuns (pe înțelesul unui elev, concis și clar):"
        )
        return generate_answer(prompt)

# Modul "Fallback" (Rezervă) - Dacă sistemul nu găsește absolut nimic util în baza ta de date, nu renunță. 
# Trimite întrebarea direct către modelul AI pentru a genera un răspuns general din cunoștințele proprii ale modelului.
    prompt = f"Ești un profesor. Răspunde clar la întrebarea: {question}"
    return generate_answer(prompt)

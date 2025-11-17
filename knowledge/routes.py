from flask import Blueprint, request, render_template, jsonify
from .service import add_knowledge

knowledge_bp = Blueprint("knowledge", __name__)

@knowledge_bp.get("/add")
def add_form():
    return render_template("add_knowledge.html")

@knowledge_bp.post("/add")
def add_form_post():
    subject = request.form.get("subject", "").strip()
    grade = request.form.get("grade", "").strip()
    content = request.form.get("content", "").strip()

    if not (subject and grade and content):
        return "Completați toate câmpurile.", 400

    add_knowledge(subject, grade, content)
    return "Informația a fost adăugată cu succes!"

@knowledge_bp.post("/api/add_knowledge")
def api_add_knowledge():
    data = request.get_json(force=True)
    subject = data.get("subject", "").strip()
    grade = data.get("grade", "").strip()
    content = data.get("content", "").strip()

    if not (subject and grade and content):
        return jsonify({"status": "error", "message": "subject, grade și content sunt obligatorii"}), 400

    add_knowledge(subject, grade, content)
    return jsonify({"status": "success"})

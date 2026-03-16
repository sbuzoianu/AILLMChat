from flask import Flask, request, jsonify, render_template, Response
from knowledge.routes import knowledge_bp
from chatbot.logic import reply_to_user

app = Flask(__name__, template_folder="templates")
app.register_blueprint(knowledge_bp)

@app.get("/test-diacritice")
def test_diacritice():
    return Response("Șțăîâă – diacritice OK", content_type="text/plain; charset=utf-8")

@app.get("/")
def index():
    return render_template("chat.html")

@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({'reply': 'Cerere invalidă: JSON malformat.'}), 400

    user_msg = data.get('message', '')
    user_grd = data.get('grade', '')
    user_sbj = data.get('subject', '')

    if not user_msg or not user_grd or not user_sbj:
        return jsonify({'reply': 'Completați toate câmpurile.'}), 400

    reply = reply_to_user(user_msg, user_grd, user_sbj)
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

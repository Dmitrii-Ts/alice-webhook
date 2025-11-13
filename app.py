from flask import Flask, request, jsonify
import os, requests

app = Flask(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Health-check Render и быстрый ручной тест
@app.get("/")
def root():
    return "ok", 200

@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
    # Разрешим preflight/GET, чтобы не падать 400
    if request.method != "POST":
        # Вернём валидный каркас, чтобы валидатор Яндекса не ругался
        return jsonify({
            "version": "1.0",
            "session": {},
            "response": {"text": "Навык на связи.", "tts": "Навык на связи.", "end_session": False}
        })

    # Мягкий парсинг JSON (silent=True: не кинет 400, если тело пустое/невалидное)
    data = request.get_json(silent=True) or {}
    req = data.get("request") or {}
    session = data.get("session") or {}
    utter = (req.get("original_utterance") or "").strip()

    if not utter:
        text = "Скажи, что ты хочешь спросить у GPT."
    else:
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                        {"role": "user", "content": utter}
                    ],
                    "temperature": 0.3
                },
                timeout=15
            )
            # Для отладки в Render → Logs
            print("STATUS:", r.status_code)
            print("BODY:", r.text[:500])
            if r.status_code == 200:
                text = (r.json()["choices"][0]["message"]["content"] or "")[:1024]
            else:
                text = f"Ошибка OpenAI API: {r.status_code}"
        except Exception as e:
            print("Ошибка запроса:", e)
            text = "Не удалось связаться с ChatGPT, попробуй позже."

    return jsonify({
        "version": "1.0",
        "session": session,
        "response": {"text": text, "tts": text, "end_session": False}
    })

from flask import Flask, request, jsonify
import os, requests

app = Flask(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route("/alice", methods=["POST"])
def alice():
    data = request.get_json(force=True)
    req = data.get("request", {})
    session = data.get("session", {})
    utter = req.get("original_utterance", "").strip()

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
            print("STATUS:", r.status_code)
            print("BODY:", r.text)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
            else:
                text = f"Ошибка OpenAI API: {r.status_code}"
        except Exception as e:
            print("Ошибка запроса:", e)
            text = "Не удалось связаться с ChatGPT, попробуй позже."

    return jsonify({
        "version": "1.0",
        "session": session,
        "response": {"text": text[:1024], "tts": text[:1024], "end_session": False}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
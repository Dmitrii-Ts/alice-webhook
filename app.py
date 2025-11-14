from flask import Flask, request, jsonify
import os
import requests
import time

app = Flask(__name__)

# === Настройки ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # рекомендуемая модель
HARD_DEADLINE_SEC = 9.0  # максимальное время на запрос к OpenAI для Алисы


def ask_openai(utter: str) -> str:
    """Запрос к новому OpenAI Responses API."""
    start = time.monotonic()

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": utter,                   # новый формат
        "temperature": 0.2,
        "max_output_tokens": 150,         # новый параметр (замена max_tokens)
    }

    def remaining():
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    attempts = 2  # 1 попытка + 1 ретрай

    for i in range(attempts):
        left = remaining()
        if left < 0.25:
            break

        timeout = min(5.0, max(0.2, left))

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)

            print("STATUS:", r.status_code)

            if r.status_code == 200:
                try:
                    # Новый формат Responses API:
                    output = r.json()["output_text"]
                    return output.strip()
                except Exception as e:
                    print("PARSE ERROR:", e, r.text[:300])
                    return "Произошла ошибка при разборе ответа."
            
            # Временные ошибки → один ретрай
            if r.status_code in (429, 500, 502, 503, 504) and i == 0:
                print("TEMP ERROR:", r.text[:300])
                time.sleep(0.5)
                continue

            # Прочие ошибки
            print("API ERROR:", r.status_code, r.text[:400])
            return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."

        except Exception as e:
            print("REQ ERROR:", e)
            if i == 0 and remaining() > 1.0:
                time.sleep(0.3)
                continue
            break

    return "Сейчас большая нагрузка. Попробуй чуть позже."


@app.get("/")
def health():
    return "ok", 200


@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
    # Алиса делает GET на проверку — просто возвращаем "Навык на связи"
    if request.method != "POST":
        return jsonify({
            "version": "1.0",
            "session": {},
            "response": {
                "text": "Навык на связи.",
                "tts": "Навык на связи.",
                "end_session": False
            }
        })

    data = request.get_json(silent=True) or {}
    req = data.get("request", {})
    session = data.get("session", {})
    utter = (req.get("original_utterance") or "").strip()

    if not utter:
        answer = "Скажи, что ты хочешь спросить у GPT."
    else:
        answer = ask_openai(utter)
        if not answer:
            answer = "Не расслышал, повтори пожалуйста."

    return jsonify({
        "version": "1.0",
        "session": session,
        "response": {
            "text": answer[:1024],  # защита от лимита Алисы
            "tts": answer[:1024],
            "end_session": False
        }
    })


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

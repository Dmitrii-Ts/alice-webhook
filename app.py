from flask import Flask, request, jsonify
import os
import requests
import time

app = Flask(__name__)

# Ключ и модель берём из переменных окружения (в Render → Environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Жёсткий бюджет времени на запрос к OpenAI (чтобы уложиться в таймаут Алисы)
HARD_DEADLINE_SEC = 9.0  # секунд

def ask_openai(utter: str) -> str:
    """Отправка запроса в OpenAI с ограничением по времени и простым ретраем."""
    start = time.monotonic()
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Отвечай кратко и по сути, на русском.",
            },
            {
                "role": "user",
                "content": utter,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 150,
    }

    def remaining() -> float:
        """Сколько времени осталось до жёсткого дедлайна."""
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    attempts = 2  # 1 попытка + 1 ретрай
    for i in range(attempts):
        left = remaining()
        if left <= 0.2:
            break

        timeout = min(5.0, max(0.2, left))  # timeout на запрос

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            print("STATUS:", r.status_code)

            if r.status_code == 200:
                try:
                    return (
                        r.json()["choices"][0]["message"]["content"]
                        .strip()
                    )
                except Exception as e:
                    print("PARSE ERROR:", e, "BODY:", r.text[:400])
                    return "Произошла внутренняя ошибка при разборе ответа."

            # rate limit / временные ошибки → один короткий ретрай
            if r.status_code in (429, 500, 502, 503, 504) and i == 0:
                print("TEMP ERROR:", r.status_code, "BODY:", r.text[:400])
                backoff = min(0.7, max(0.3, remaining() - 0.5))
                if backoff > 0.3:
                    print("BACKOFF:", backoff, "sec")
                    time.sleep(backoff)
                    continue
                # если времени мало — сразу фолбэк
                return "Сервис перегружен. Попробуй ещё раз чуть позже."

            # другие коды — короткое сообщение пользователю
            print("API ERROR:", r.status_code, r.text[:400])
            return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."
        except Exception as e:
            print("REQ ERROR:", e)
            if i == 0 and remaining() > 1.0:
                time.sleep(0.3)
                continue
            break

    return "Сейчас высокая нагрузка, попробуй повторить запрос через минуту."


@app.get("/")
def root():
    """Проверка живости сервиса (health-check)."""
    return "ok", 200


@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
    # Любые не-POST запросы (health-check, тесты) — не должны падать с 400
    if request.method != "POST":
        return jsonify(
            {
                "version": "1.0",
                "session": {},
                "response": {
                    "text": "Навык на связи.",
                    "tts": "Навык на связи.",
                    "end_session": False,
                },
            }
        )

    data = request.get_json(silent=True) or {}
    req = data.get("request") or {}
    session = data.get("session") or {}
    utter = (req.get("original_utterance") or "").strip()

    if not utter:
        text = "Скажи, что ты хочешь спросить у GPT."
    else:
        text = ask_openai(utter)
        if not text:
            text = "Не расслышал. Повтори, пожалуйста."

    return jsonify(
        {
            "version": "1.0",
            "session": session,
            "response": {
                "text": text[:1024],
                "tts": text[:1024],
                "end_session": False,
            },
        }
    )


if __name__ == "__main__":
    # Локальный запуск (для отладки)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

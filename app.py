from flask import Flask, request, jsonify
import os
import requests
import time

app = Flask(__name__)

# Настройки окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Можно менять модель только через переменную окружения в Render:
# Например: gpt-5-mini, gpt-5-nano, gpt-4.1-mini, gpt-4o-mini
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Жёсткий лимит по времени на запрос к OpenAI (чтобы вписаться в 8 секунд Алисы)
HARD_DEADLINE_SEC = 4.5  # весь круг до ответа от OpenAI


def extract_text_from_response(resp_json: dict) -> str:
    """
    Достаём текст из формата Responses API:
    {
      "output": [
        {
          "content": [
            {
              "type": "output_text" | "text",
              "text": { "value": "Текст", ... }
            }
          ]
        }
      ]
    }
    """
    output = resp_json.get("output") or resp_json.get("outputs") or []
    if not output:
        return ""

    first_output = output[0]
    content = first_output.get("content") or []
    for item in content:
        t = item.get("type")
        if t in ("output_text", "text"):
            text_obj = item.get("text") or {}
            value = text_obj.get("value") or ""
            if value:
                return value.strip()

    return ""


def ask_openai(utter: str) -> str:
    """
    Один быстрый запрос к OpenAI Responses API без ретраев.
    Zадача — ответить быстро и стабильно, а не идеально длинно.
    """
    start = time.monotonic()

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # ВАЖНО:
    # - НЕ используем temperature, чтобы не ловить 400 на моделях без температуры.
    # - НЕ используем max_tokens / max_output_tokens — только max_completion_tokens.
    payload = {
        "model": MODEL,
        "input": utter,
         "max_output_tokens": 80,  # короткий ответ => быстрее, меньше ошибок
    }

    def remaining():
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    left = remaining()
    if left <= 0.3:
        return "Сейчас высокая нагрузка. Попробуй ещё раз."

    timeout = min(2.5, max(0.3, left))  # максимум 2.5 секунды на OpenAI

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("STATUS:", r.status_code)
        print("RAW BODY:", r.text[:400])

        if r.status_code == 200:
            resp_json = r.json()
            text = extract_text_from_response(resp_json)
            if text:
                return text

            # Если текст не вытащился — попробуем посмотреть статус
            status = resp_json.get("status")
            if status != "completed":
                reason = (resp_json.get("incomplete_details") or {}).get("reason")
                if reason == "max_completion_tokens":
                    return "Ответ получился слишком длинным и был обрезан. Попробуй спросить короче."
            return "Не удалось корректно обработать ответ от модели."

        # Временные/лимитные ошибки — просто короткий фолбэк
        if r.status_code in (429, 500, 502, 503, 504):
            return "Сервис перегружен. Попробуй ещё раз через минуту."

        # Остальное — показываем код
        return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."

    except Exception as e:
        print("REQ ERROR:", e)
        return "Не удалось связаться с моделью. Попробуй ещё раз позже."


@app.get("/")
def health():
    return "ok", 200


@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
    # GET/OPTIONS — хелсчек от Render или ручной заход из браузера
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
    req = data.get("request") or {}
    session = data.get("session") or {}
    utter = (req.get("original_utterance") or "").strip()

    if not utter:
        answer = "Скажи, что ты хочешь спросить у GPT."
    else:
        answer = ask_openai(utter)
        if not answer:
            answer = "Не расслышал, повтори, пожалуйста."

    return jsonify({
        "version": "1.0",
        "session": session,
        "response": {
            "text": answer[:1024],
            "tts": answer[:1024],
            "end_session": False
        }
    })


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

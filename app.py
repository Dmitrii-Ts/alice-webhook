from flask import Flask, request, jsonify
import os
import requests
import time

app = Flask(__name__)

# === Настройки ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # можно поменять в Render→Environment
HARD_DEADLINE_SEC = 9.0  # максимум на запрос к OpenAI, чтобы успеть в таймаут Алисы


def extract_text_from_response(resp_json: dict) -> str:
    """
    Достаём текст из формата Responses API:
    {
      "output": [
        {
          "content": [
            {
              "type": "output_text",
              "text": {
                "value": "Тут текст",
                ...
              }
            }
          ]
        }
      ]
    }
    """
    output = resp_json.get("output") or resp_json.get("outputs") or []
    if not output:
        return ""

    # Берём первый output
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
    """Запрос к новому OpenAI Responses API с ограничением по времени и ретраем."""
    start = time.monotonic()

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": utter,
        # ВАЖНО: temperature поддерживается не всеми моделями.
        # Если используешь o1 / o3 / some nano — убери эту строку.
        "max_output_tokens": 150,  # если будет мало — просто ответ обрежется
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
            print("RAW BODY:", r.text[:500])

            if r.status_code == 200:
                try:
                    resp_json = r.json()

                    # даже если status == "incomplete" из-за max_output_tokens —
                    # текст обычно всё равно есть в output
                    text = extract_text_from_response(resp_json)
                    if text:
                        return text

                    # если почему-то текста нет — пробуем дать аккуратный ответ
                    status = resp_json.get("status")
                    if status != "completed":
                        reason = (resp_json.get("incomplete_details") or {}).get("reason")
                        if reason == "max_output_tokens":
                            return "Ответ получился слишком длинным и был обрезан. Попробуй спросить короче."
                    return "Не удалось корректно разобрать ответ модели."
                except Exception as e:
                    print("PARSE ERROR:", e, r.text[:300])
                    return "Произошла ошибка при обработке ответа от GPT."

            # Временные ошибки → один ретрай
            if r.status_code in (429, 500, 502, 503, 504) and i == 0:
                print("TEMP ERROR:", r.status_code, r.text[:300])
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
    # Не-POST запросы (health-check от Render, ручной GET в браузере)
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

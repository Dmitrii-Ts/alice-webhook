from flask import Flask, request, jsonify
import os
import requests
import time

app = Flask(__name__)

# === Настройки окружения ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Рекомендуется задавать модель в Render → Environment,
# например:
#   OPENAI_MODEL = gpt-5-nano-2025-08-07
#   или (чуть умнее, но медленнее):
#   OPENAI_MODEL = gpt-4.1-2025-04-14
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

# Жёсткий лимит по времени на запрос к OpenAI (внутри ask_openai),
# чтобы уложиться в таймаут Алисы.
HARD_DEADLINE_SEC = 6.0


def extract_text_from_response(resp_json):
    """
    Универсально и безопасно достаём текст из Responses API.

    Никаких исключений наружу не выбрасываем — максимум возвращаем "".
    """
    try:
        # Если вдруг это не dict, а уже строка/что-то ещё
        if isinstance(resp_json, str):
            return resp_json.strip()

        if not isinstance(resp_json, dict):
            return ""

        # --- Основной путь: поле output / outputs ---
        outputs = resp_json.get("output") or resp_json.get("outputs")

        candidates = []
        if isinstance(outputs, list):
            candidates = outputs
        elif isinstance(outputs, dict):
            candidates = [outputs]

        for out in candidates:
            if not isinstance(out, dict):
                continue
            content = out.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    # 1) классический вариант: item["text"]["value"]
                    text_obj = item.get("text")
                    if isinstance(text_obj, dict):
                        val = text_obj.get("value")
                        if isinstance(val, str) and val.strip():
                            return val.strip()

                    # 2) альтернативный вариант: item["output_text"]["value"]
                    out_text_obj = item.get("output_text")
                    if isinstance(out_text_obj, dict):
                        val = out_text_obj.get("value")
                        if isinstance(val, str) and val.strip():
                            return val.strip()

                    # 3) иногда бывает просто строка в content
                    #    (на всякий случай)
                    direct_content = item.get("content")
                    if isinstance(direct_content, str) and direct_content.strip():
                        return direct_content.strip()

        # --- Фолбэк: рекурсивный поиск text.value или value по всему JSON ---

        def dfs(obj):
            # dict: смотрим text.value и value
            if isinstance(obj, dict):
                t_obj = obj.get("text")
                if isinstance(t_obj, dict):
                    v = t_obj.get("value")
                    if isinstance(v, str) and v.strip():
                        return v.strip()

                # некоторые структуры кладут текст прямо в value
                v = obj.get("value")
                if isinstance(v, str) and v.strip():
                    return v.strip()

                for v in obj.values():
                    res = dfs(v)
                    if res:
                        return res

            # list: обходим элементы
            elif isinstance(obj, list):
                for v in obj:
                    res = dfs(v)
                    if res:
                        return res

            # строки и всё остальное пропускаем
            return ""

        return dfs(resp_json) or ""
    except Exception as e:
        # Никогда не роняемся из-за парсинга
        print("PARSE ERROR in extract_text_from_response:", repr(e))
        return ""


def ask_openai(utter: str) -> str:
    """
    Один быстрый запрос к OpenAI Responses API без ретраев.
    Задача — ответить быстро и стабильно.
    """
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set")
        return "Ключ доступа к модели не настроен. Попробуй позже."

    start = time.monotonic()

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Важно:
    # - НЕ используем temperature (часть моделей его не поддерживает → 400).
    # - Используем max_output_tokens (как требует Responses API).
    payload = {
        "model": MODEL,
        "input": utter,
        "max_output_tokens": 120,  # 120 токенов ≈ 3–5 предложений
    }

    def remaining():
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    left = remaining()
    if left <= 0.3:
        return "Сейчас высокая нагрузка. Попробуй ещё раз."

    # Даём модели максимум 3.5 секунды, но не выходим за жёсткий бюджет
    timeout = min(3.5, max(0.3, left))

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print("STATUS:", r.status_code)
        print("RAW BODY:", r.text[:400])

        if r.status_code == 200:
            try:
                resp_json = r.json()
            except Exception as e:
                print("JSON PARSE ERROR:", repr(e))
                return "Не удалось обработать ответ от модели."

            text = extract_text_from_response(resp_json)
            if text:
                # Даже если status == "incomplete" из-за max_output_tokens —
                # просто отдаём то, что есть.
                return text

            # Если вообще не смогли достать текст
            return "Не удалось корректно обработать ответ от модели."

        # Временные/лимитные ошибки — короткий фолбэк
        if r.status_code in (429, 500, 502, 503, 504):
            return "Сервис перегружен. Попробуй ещё раз чуть позже."

        # Любой другой код — показываем код пользователю
        return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."

    except Exception as e:
        print("REQ ERROR:", repr(e))
        return "Не удалось связаться с моделью. Попробуй ещё раз позже."


@app.get("/")
def health():
    # Для UptimeRobot / Render health-check
    return "ok", 200


@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
    # GET/OPTIONS — health-check от Render/браузера/Диалогов
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
    if not isinstance(data, dict):
        data = {}

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
    # Локальный запуск (на Render используется gunicorn app:app)
    PORT = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=PORT)

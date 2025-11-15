from flask import Flask, request, jsonify
import os
import requests
import time
from typing import Any, Dict, List, Optional

app = Flask(__name__)

# Ключ и модель берём из переменных окружения (в Render → Environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Явное принудительное использование Responses API (если установлено в "1"/"true"/"yes")
OPENAI_USE_RESPONSES = os.getenv("OPENAI_USE_RESPONSES", "").lower()

# Жёсткий бюджет времени на запрос к OpenAI (чтобы уложиться в таймаут Алисы)
HARD_DEADLINE_SEC = 9.0  # секунд


def should_use_responses_api(model: str) -> bool:
    """
    Решение, какой endpoint использовать:
    - Если переменная OPENAI_USE_RESPONSES установлена явно (true/1/yes) — используем Responses API.
    - Иначе делаем эвристическую проверку по имени модели (новые модели обычно содержат 'gpt-4.1', 'gpt-5', 'gpt-4o' и т.п.).
    """
    if OPENAI_USE_RESPONSES in ("1", "true", "yes"):
        return True
    if OPENAI_USE_RESPONSES in ("0", "false", "no"):
        return False

    model_l = (model or "").lower()
    # Ключевые слова моделей, которые, как правило, поддерживают Responses API
    keywords = ["gpt-4.1", "gpt-5", "gpt-4o", "gpt-5-nano", "gpt-4o-mini", "gpt-4o-"]
    for k in keywords:
        if k in model_l:
            return True
    # по умолчанию оставляем попытку использовать Responses API для современных моделей,
    # но если модель явно старше (например, содержит "gpt-3"), не использовать.
    if "gpt-3" in model_l or "text-" in model_l:
        return False
    # Default: попытаться использовать Responses API
    return True


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    """
    Безопасно извлекает человекочитаемый текст из структуры ответа нового OpenAI Responses API
    или старого формата. Поддерживает:
      - новый формат: item['text'] = "string"
      - старый формат: item['text'] = {'value': 'string'} или {'text': 'string'}
    Имеется рекурсивный fallback, который ищет ключи 'text' или 'value' и возвращает первую найденную строку.
    Никогда не бросает исключение — в крайнем случае возвращает пустую строку.
    """
    def extract_any_text(node: Any) -> Optional[str]:
        if node is None:
            return None
        if isinstance(node, str):
            s = node.strip()
            return s if s else None
        if isinstance(node, dict):
            # приоритетные ключи
            for key in ("text", "value", "content", "message", "body"):
                if key in node:
                    val = node[key]
                    if isinstance(val, str):
                        v = val.strip()
                        if v:
                            return v
                    else:
                        candidate = extract_any_text(val)
                        if candidate:
                            return candidate
            # поиск во всех значениях словаря
            for v in node.values():
                candidate = extract_any_text(v)
                if candidate:
                    return candidate
            return None
        if isinstance(node, (list, tuple)):
            for elem in node:
                candidate = extract_any_text(elem)
                if candidate:
                    return candidate
            return None
        return None

    def extract_from_item(item: Any) -> Optional[str]:
        if not isinstance(item, dict):
            return extract_any_text(item)
        try:
            typ = item.get("type")
            if typ == "output_text":
                text_field = item.get("text")
                # новый формат: text — строка
                if isinstance(text_field, str):
                    t = text_field.strip()
                    if t:
                        return t
                # старый формат: text — dict с value или text
                if isinstance(text_field, dict):
                    for k in ("value", "text"):
                        if k in text_field:
                            v = text_field[k]
                            if isinstance(v, str):
                                s = v.strip()
                                if s:
                                    return s
                            else:
                                candidate = extract_any_text(v)
                                if candidate:
                                    return candidate
                    # рекурсивный поиск внутри text_field
                    candidate = extract_any_text(text_field)
                    if candidate:
                        return candidate
            # общий рекурсивный поиск (fallback)
            return extract_any_text(item)
        except Exception:
            return None

    try:
        results: List[str] = []
        outputs = resp.get("output") or resp.get("outputs") or resp.get("outputs_list") or []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            if isinstance(out, dict) and "content" in out and isinstance(out["content"], list):
                for item in out["content"]:
                    text_piece = extract_from_item(item)
                    if text_piece:
                        results.append(text_piece)
            else:
                candidate = extract_any_text(out)
                if candidate:
                    results.append(candidate)

        # запасной рекурсивный проход по всему ответу, если раньше ничего не нашли
        if not results:
            candidate = extract_any_text(resp)
            if candidate:
                results.append(candidate)

        joined = "\n\n".join(results).strip()
        return joined
    except Exception:
        return ""


def ask_openai(utter: str) -> str:
    """
    Отправка запроса в OpenAI с ограничением по времени и простым ретраем.
    Автоматически выбирает между /v1/responses и /v1/chat/completions в зависимости от модели
    (или переменной OPENAI_USE_RESPONSES).
    """
    start = time.monotonic()
    use_responses = should_use_responses_api(MODEL)

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
            if use_responses:
                url = "https://api.openai.com/v1/responses"
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL,
                    # Можно передавать "input" как строку или как список сообщений.
                    # Для лучшей совместимости используем список сообщений.
                    "input": [
                        {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                        {"role": "user", "content": utter},
                    ],
                    "temperature": 0.2,
                    "max_output_tokens": 150,
                }
            else:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                        {"role": "user", "content": utter},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150,
                }

            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            print("STATUS:", r.status_code, "URL:", url)

            if r.status_code == 200:
                try:
                    raw = r.json()
                except Exception as e:
                    print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                    return "Произошла внутренняя ошибка при разборе ответа."

                # Если используем Responses API — парсим универсальной функцией
                if use_responses:
                    text_from_any = extract_text_from_response(raw)
                    if text_from_any:
                        return text_from_any
                    # Если статус явно указан и не completed — вернуть ошибку/фолбэк
                    status = raw.get("status", "")
                    if status and status != "completed":
                        print("Responses API status not completed:", status)
                        return "Сервис временно недоступен. Попробуй ещё раз позже."
                    # иначе пробуем также парсинг старого формата на всякий случай
                    try:
                        choices = raw.get("choices")
                        if isinstance(choices, list) and choices:
                            msg = choices[0].get("message") or {}
                            content = msg.get("content")
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                            if isinstance(content, dict):
                                cf = extract_text_from_response({"output": [{"content": [content]}]})
                                if cf:
                                    return cf
                    except Exception:
                        pass
                    print("EMPTY PARSE for Responses API. RAW:", raw)
                    return "Произошла внутренняя ошибка при разборе ответа."

                # Если используем старый chat completions — пробуем привычный путь
                else:
                    try:
                        choices = raw.get("choices")
                        if isinstance(choices, list) and choices:
                            choice0 = choices[0]
                            # новая структура: choice0.message.content
                            if isinstance(choice0.get("message"), dict):
                                content = choice0["message"].get("content")
                                if isinstance(content, str) and content.strip():
                                    return content.strip()
                                # если content — dict (редко), пытаемся извлечь
                                if isinstance(content, dict):
                                    cf = extract_text_from_response({"output": [{"content": [content]}]})
                                    if cf:
                                        return cf
                            # старый вариант: choice0["text"]
                            if "text" in choice0 and isinstance(choice0["text"], str) and choice0["text"].strip():
                                return choice0["text"].strip()
                    except Exception as e:
                        print("PARSE CHAT COMPLETIONS ERROR:", e)
                    print("EMPTY PARSE for Chat Completions. RAW:", raw)
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

from flask import Flask, request, jsonify
import os
import requests
import time
from typing import Any, Dict, List, Optional

app = Flask(__name__)

# Ключ и модель берём из переменных окружения (в Render → Environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # поменяйте на вашу модель, например "gpt-5-mini"
# Жёсткий бюджет времени на запрос к OpenAI (чтобы уложиться в таймаут Алисы)
HARD_DEADLINE_SEC = 9.0  # секунд


def should_use_responses_api(model: str) -> bool:
    """
    Простая эвристика: современные модели (gpt-4.1, gpt-5, gpt-4o и т.п.) — использовать Responses API.
    """
    model_l = (model or "").lower()
    if not model_l:
        return True
    if "gpt-3" in model_l or model_l.startswith("text-"):
        return False
    # модели 5-mini/5.1-mini/4.1 и т.п. — считать современными
    keywords = ["gpt-4.1", "gpt-5", "gpt-4o", "gpt-5-mini", "gpt-5.1-mini", "gpt-5.1", "gpt-5-"]
    for k in keywords:
        if k in model_l:
            return True
    # по умолчанию — использовать Responses API
    return True


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    """
    Универсальный извлекатель текста (поддерживает новый Responses API и старый chat/completions).
    Возвращает пустую строку если ничего не найдено.
    """
    def extract_any_text(node: Any) -> Optional[str]:
        if node is None:
            return None
        if isinstance(node, str):
            s = node.strip()
            return s if s else None
        if isinstance(node, dict):
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
                if isinstance(text_field, str):
                    t = text_field.strip()
                    if t:
                        return t
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
                    candidate = extract_any_text(text_field)
                    if candidate:
                        return candidate
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

        if not results:
            candidate = extract_any_text(resp)
            if candidate:
                results.append(candidate)

        joined = "\n\n".join(results).strip()
        return joined
    except Exception:
        return ""


def parse_unsupported_params_from_message(msg: str) -> List[str]:
    """
    Попытаться извлечь имена неподдерживаемых параметров из текста ошибки OpenAI.
    Пример сообщения: "Unsupported parameter: 'temperature' is not supported with this model."
    Вернёт ['temperature'].
    """
    try:
        msg_l = msg.lower()
        keys: List[str] = []
        # простая эвристика: искать в кавычках или после слова 'unsupported parameter'
        if "unsupported parameter" in msg_l:
            # найти части вида 'temperature'
            parts = msg.split("'")
            for p in parts:
                pn = p.strip()
                if pn and pn.isidentifier():
                    keys.append(pn)
        # также искать явные слова temperature / max_output_tokens / max_tokens
        for candidate in ("temperature", "max_output_tokens", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
            if candidate in msg_l and candidate not in keys:
                keys.append(candidate)
        return keys
    except Exception:
        return []


def try_post_with_removing_params(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> requests.Response:
    """
    Выполнить POST, и при ошибке 400 с сообщением о неподдерживаемом параметре
    пробовать повторно, удаляя проблемные ключи из payload (до нескольких итераций).
    Возвращает последний Response.
    """
    max_retries = 3
    tried_removed = set()
    last_resp = None
    for attempt in range(max_retries):
        try:
            last_resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception:
            # пробелы сети — сразу вернуть ошибку наружу (будет обработано в вызове)
            raise
        if last_resp.status_code != 400:
            return last_resp
        # разбор тела ошибки
        try:
            err = last_resp.json().get("error", {})
            msg = err.get("message", "") or ""
        except Exception:
            msg = last_resp.text or ""
        unsupported = parse_unsupported_params_from_message(msg)
        # убрать уже удалённые или несуществующие ключи
        unsupported = [k for k in unsupported if k and k in payload and k not in tried_removed]
        if not unsupported:
            # ничего понятного для удаления — вернуть текущий ответ
            return last_resp
        # удаляем найденные параметры и будем повторять запрос
        for k in unsupported:
            tried_removed.add(k)
            payload.pop(k, None)
            print(f"Removed unsupported parameter '{k}' from payload and retrying.")
        # короткая пауза перед ретраем
        time.sleep(0.05)
    return last_resp


def ask_openai(utter: str) -> str:
    """
    Отправка запроса в OpenAI. Поддерживает автоматическое определение эндпоинта.
    Для новых моделей (например, 5-mini) при возникновении ошибок с неподдерживаемыми параметрами
    автоматически удаляет эти параметры и повторяет запрос.
    """
    start = time.monotonic()
    use_responses = should_use_responses_api(MODEL)

    def remaining() -> float:
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    attempts = 2
    for i in range(attempts):
        left = remaining()
        if left <= 0.2:
            break
        timeout = min(5.0, max(0.2, left))

        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            if use_responses:
                url = "https://api.openai.com/v1/responses"
                # Для максимальной совместимости с различными моделями:
                # - отправляем input как простую строку (многие модели принимают)
                # - добавляем параметры, которые могут быть удалены при ошибке
                payload: Dict[str, Any] = {
                    "model": MODEL,
                    "input": utter,
                    "temperature": 0.2,
                    "max_output_tokens": 150,
                }
            else:
                url = "https://api.openai.com/v1/chat/completions"
                payload = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                        {"role": "user", "content": utter},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150,
                }

            # Попытаться отправить и при необходимости удалить неподдерживаемые параметры
            r = try_post_with_removing_params(url, headers, payload, timeout)
            print("STATUS:", r.status_code, "URL:", url)

            if r.status_code == 200:
                try:
                    raw = r.json()
                except Exception as e:
                    print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                    return "Произошла внутренняя ошибка при разборе ответа."

                # Если Responses API — сначала использовать универсальный извлекатель
                if use_responses:
                    text_from_any = extract_text_from_response(raw)
                    if text_from_any:
                        return text_from_any
                    # если поле status есть и не completed — вернуть краткое сообщение
                    status = raw.get("status", "")
                    if status and status != "completed":
                        print("Responses API status not completed:", status)
                        return "Сервис временно недоступен. Попробуй ещё раз позже."
                    # в крайнем случае попытка посмотреть старый chat формат
                    try:
                        choices = raw.get("choices")
                        if isinstance(choices, list) and choices:
                            msg = choices[0].get("message") or {}
                            content = msg.get("content")
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                    except Exception:
                        pass
                    print("EMPTY PARSE for Responses API. RAW:", raw)
                    return "Произошла внутренняя ошибка при разборе ответа."
                else:
                    # старый chat/completions
                    try:
                        choices = raw.get("choices")
                        if isinstance(choices, list) and choices:
                            choice0 = choices[0]
                            if isinstance(choice0.get("message"), dict):
                                content = choice0["message"].get("content")
                                if isinstance(content, str) and content.strip():
                                    return content.strip()
                                if isinstance(content, dict):
                                    cf = extract_text_from_response({"output": [{"content": [content]}]})
                                    if cf:
                                        return cf
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
                    time.sleep(backoff)
                    continue
                return "Сервис перегружен. Попробуй ещё раз чуть позже."

            # прочие ошибки — вывести сообщение пользователю
            print("API ERROR:", r.status_code, r.text[:400])
            # если 400 — вернуть тело ошибки в лог и краткое сообщение пользователю
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

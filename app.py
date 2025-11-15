from flask import Flask, request, jsonify
import os
import requests
import time
import re
import json
import datetime
from typing import Any, Dict, List, Optional
import requests.exceptions as req_exc

app = Flask(__name__)

# Ключ и модель берём из переменных окружения (в Render → Environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # поменяйте на вашу модель, например "gpt-5-mini"

# Включить сохранение raw-ответов в /tmp (по умолчанию включено). Поставьте "0" чтобы отключить.
OPENAI_SAVE_RAW = os.getenv("OPENAI_SAVE_RAW", "1")

# Жёсткий бюджет времени на запрос к OpenAI (чтобы уложиться в таймаут Алисы)
HARD_DEADLINE_SEC = float(os.getenv("HARD_DEADLINE_SEC", "9.0"))

# Начальное число токенов для Responses API (можно поднять через окружение)
BASE_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_BASE_MAX_OUTPUT_TOKENS", "300"))
# Максимальное значение max_output_tokens, до которого можно автоматически увеличивать при retry
MAX_OUTPUT_TOKENS_CAP = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "1024"))


def should_use_responses_api(model: str) -> bool:
    model_l = (model or "").lower()
    if not model_l:
        return True
    if "gpt-3" in model_l or model_l.startswith("text-"):
        return False
    keywords = ["gpt-4.1", "gpt-5", "gpt-4o", "gpt-5-mini", "gpt-5.1-mini", "gpt-5.1", "gpt-5-"]
    for k in keywords:
        if k in model_l:
            return True
    return True


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    """
    Универсальный извлекатель текста, улучшенный фильтр для исключения служебных id'шек
    (например rs_... ) из результата.
    """
    def looks_like_id(s: str) -> bool:
        s_stripped = s.strip()
        if not s_stripped:
            return False
        s_low = s_stripped.lower()
        if re.fullmatch(r'rs_[0-9a-f]+', s_low):
            return True
        if re.fullmatch(r'resp_[0-9a-f]+', s_low):
            return True
        if re.fullmatch(r'[0-9a-f]{12,}', s_low):
            return True
        if len(s_stripped) <= 30 and ' ' not in s_stripped and re.fullmatch(r'[\w\-]+', s_stripped):
            return True
        return False

    def is_likely_user_text(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        s = s.strip()
        if not s:
            return False
        if re.search(r'[\u0400-\u04FF]', s):
            return True
        if ' ' in s and len(s) > 5:
            return True
        if re.search(r'[.,!?…]', s) and len(s) > 3:
            return True
        if len(s) >= 80:
            return True
        return False

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

    def extract_all_strings(node: Any, out: List[str]) -> None:
        if node is None:
            return
        if isinstance(node, str):
            s = node.strip()
            if s:
                out.append(s)
            return
        if isinstance(node, dict):
            for v in node.values():
                extract_all_strings(v, out)
            return
        if isinstance(node, (list, tuple)):
            for elem in node:
                extract_all_strings(elem, out)
            return
        return

    def extract_from_item(item: Any) -> Optional[str]:
        if not isinstance(item, dict):
            return extract_any_text(item)
        typ = item.get("type")
        if typ == "output_text":
            text_field = item.get("text")
            if isinstance(text_field, str):
                t = text_field.strip()
                if t and not looks_like_id(t):
                    return t
            if isinstance(text_field, dict):
                for k in ("value", "text"):
                    if k in text_field:
                        v = text_field[k]
                        if isinstance(v, str):
                            s = v.strip()
                            if s and not looks_like_id(s):
                                return s
                        else:
                            candidate = extract_any_text(v)
                            if candidate and not looks_like_id(candidate):
                                return candidate
                candidate = extract_any_text(text_field)
                if candidate and not looks_like_id(candidate):
                    return candidate
        candidate = extract_any_text(item)
        if candidate and not looks_like_id(candidate):
            return candidate
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

        # fallback: choices/message
        if not results:
            try:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0]
                    if isinstance(choice0.get("message"), dict):
                        content = choice0["message"].get("content")
                        if isinstance(content, str) and content.strip() and not looks_like_id(content.strip()):
                            results.append(content.strip())
                        elif isinstance(content, dict):
                            for k in ("value", "text"):
                                if k in content and isinstance(content[k], str):
                                    val = content[k].strip()
                                    if val and not looks_like_id(val):
                                        results.append(val)
                                        break
            except Exception:
                pass

        if not results:
            all_strings: List[str] = []
            extract_all_strings(resp, all_strings)
            filtered = [s for s in all_strings if not looks_like_id(s)]
            for s in filtered:
                if is_likely_user_text(s):
                    results.append(s)
                    break
            if not results and filtered:
                results.append(max(filtered, key=len))

        if not results:
            return ""
        joined = "\n\n".join(results).strip()
        return joined
    except Exception:
        return ""


def parse_unsupported_params_from_message(msg: str) -> List[str]:
    try:
        msg_l = (msg or "").lower()
        keys: List[str] = []
        if "unsupported parameter" in msg_l:
            parts = msg.split("'")
            for p in parts:
                pn = p.strip()
                if pn and pn.isidentifier():
                    keys.append(pn)
        for candidate in ("temperature", "max_output_tokens", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"):
            if candidate in msg_l and candidate not in keys:
                keys.append(candidate)
        return keys
    except Exception:
        return []


def try_post_with_removing_params(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> requests.Response:
    max_retries = 3
    tried_removed = set()
    last_resp = None
    for attempt in range(max_retries):
        try:
            last_resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception:
            raise
        if last_resp.status_code != 400:
            return last_resp
        try:
            err = last_resp.json().get("error", {})
            msg = err.get("message", "") or ""
        except Exception:
            msg = last_resp.text or ""
        unsupported = parse_unsupported_params_from_message(msg)
        unsupported = [k for k in unsupported if k and k in payload and k not in tried_removed]
        if not unsupported:
            return last_resp
        for k in unsupported:
            tried_removed.add(k)
            payload.pop(k, None)
            print(f"Removed unsupported parameter '{k}' from payload and retrying.")
        time.sleep(0.05)
    return last_resp


def _extract_partial_message_text(raw: Dict[str, Any]) -> Optional[str]:
    """
    Если модель вернула message (даже с status 'incomplete') и в message.content есть output_text —
    вернуть это частичное содержимое (полезно, когда модель остановилась по max_output_tokens).
    """
    try:
        outputs = raw.get("output") or []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            # message items содержат type == "message"
            if isinstance(out, dict) and out.get("type") == "message":
                content = out.get("content") or []
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "output_text":
                            tf = item.get("text")
                            if isinstance(tf, str):
                                s = tf.strip()
                                if s:
                                    parts.append(s)
                            elif isinstance(tf, dict):
                                v = tf.get("value") or tf.get("text")
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                    if parts:
                        # объединяем части (часто последняя строка может обрываться)
                        return "\n\n".join(parts).strip()
        # также проверить choices.message.content как fallback
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0.get("message"), dict):
                content = choice0["message"].get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, dict):
                    v = content.get("value") or content.get("text")
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    except Exception:
        pass
    return None


def ask_openai(utter: str) -> str:
    """
    Отправка запроса в OpenAI. Для Responses API реализована авто-поправка unsupported params,
    авто-retry на incomplete==max_output_tokens (увеличиваем max_output_tokens и пробуем снова) и
    возвращение частичного assistant message если он есть (даже при статусе 'incomplete').
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
                max_output_tokens = BASE_MAX_OUTPUT_TOKENS
                while True:
                    rem = remaining()
                    if rem <= 0.8:
                        print("Not enough remaining time to retry (remaining:", rem, "). Aborting retries.")
                        return "Сервис временно недоступен. Попробуй ещё раз позже."
                    per_req_timeout = max(0.8, min(12.0, rem - 0.25))

                    payload: Dict[str, Any] = {
                        "model": MODEL,
                        "input": utter,
                        "max_output_tokens": max_output_tokens,
                        # optional params that might be removed by try_post_with_removing_params
                        "temperature": 0.2,
                    }

                    try:
                        r = try_post_with_removing_params(url, headers, payload, per_req_timeout)
                    except req_exc.ReadTimeout:
                        print("Read timed out on attempt with max_output_tokens=", max_output_tokens, "remaining=", rem)
                        rem2 = remaining()
                        if rem2 > 1.0:
                            per_req_timeout2 = max(0.8, min(15.0, rem2 - 0.15))
                            try:
                                print("Trying one more attempt with slightly larger timeout:", per_req_timeout2)
                                r = try_post_with_removing_params(url, headers, payload, per_req_timeout2)
                            except Exception as e2:
                                print("Second attempt after ReadTimeout failed:", e2)
                                return "Сервис временно недоступен. Попробуй ещё раз позже."
                        else:
                            return "Сервис временно недоступен. Попробуй ещё раз позже."
                    except req_exc.ConnectionError as e:
                        print("Connection error on request:", e)
                        return "Сервис временно недоступен. Попробуйте позже."
                    except Exception as e:
                        print("Request error on request:", e)
                        return "Сервис временно недоступен. Попробуйте позже."

                    print("STATUS:", r.status_code, "URL:", url, "MODEL:", MODEL, "max_output_tokens:", max_output_tokens)

                    if r.status_code != 200:
                        break

                    try:
                        raw = r.json()
                    except Exception as e:
                        print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                        return "Произошла внутренняя ошибка при разборе ответа."

                    # --- DEBUG: сохранить сырой ответ для последующего анализа ---
                    try:
                        if str(OPENAI_SAVE_RAW).lower() in ("1", "true", "yes"):
                            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                            path = f"/tmp/openai_raw_{ts}.json"
                            with open(path, "w", encoding="utf-8") as _f:
                                json.dump(raw, _f, ensure_ascii=False, indent=2)
                            print(f"Saved raw OpenAI response to {path}")
                        try:
                            print("RAW (truncated):", str(raw)[:2000])
                        except Exception:
                            pass
                    except Exception as _e:
                        print("Failed to save raw response:", _e)
                    # --- /DEBUG ---

                    # Если есть частичный assistant message (даже при incomplete) — вернуть его пользователю
                    partial = _extract_partial_message_text(raw)
                    if partial:
                        # Добавим пометку, что это частичный ответ, если статус incomplete
                        status = raw.get("status", "")
                        incomplete = raw.get("incomplete_details")
                        reason = None
                        try:
                            if isinstance(incomplete, dict):
                                reason = incomplete.get("reason")
                        except Exception:
                            reason = None
                        if status == "incomplete":
                            # Возвращаем частичный текст (чтобы пользователь не получил пустой ответ), можно пометить/усл. добавить "..." при желании
                            return partial
                        else:
                            return partial

                    # Иначе, пытаемся извлечь текст общим парсером (например, если message/assistant полностью готов)
                    text_from_any = extract_text_from_response(raw)
                    if text_from_any:
                        return text_from_any

                    # Если статус incomplete и причина max_output_tokens — попробуем увеличить max_output_tokens и retry,
                    # но только если осталось время и не превысим кап.
                    status = raw.get("status", "")
                    incomplete = raw.get("incomplete_details")
                    reason = None
                    try:
                        if isinstance(incomplete, dict):
                            reason = incomplete.get("reason")
                    except Exception:
                        reason = None

                    if status == "incomplete" and reason == "max_output_tokens":
                        if remaining() > 1.5 and max_output_tokens < MAX_OUTPUT_TOKENS_CAP:
                            new_max = min(MAX_OUTPUT_TOKENS_CAP, max_output_tokens * 2)
                            if new_max == max_output_tokens:
                                print("Reached max cap for max_output_tokens; not retrying.")
                                return "Сервис временно недоступен. Попробуй ещё раз позже."
                            print(f"Response incomplete due to max_output_tokens; retrying with max_output_tokens={new_max}")
                            max_output_tokens = new_max
                            time.sleep(0.05)
                            continue
                        else:
                            print("Responses API status not completed:", status, "reason:", reason)
                            return "Сервис временно недоступен. Попробуй ещё раз позже."

                    print("EMPTY PARSE for Responses API. RAW:", raw)
                    return "Произошла внутренняя ошибка при разборе ответа."

            else:
                # старый chat/completions путь (без auto-retry по max_output_tokens)
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
                r = try_post_with_removing_params(url, headers, payload, timeout)
                print("STATUS:", r.status_code, "URL:", url)
                if r.status_code == 200:
                    try:
                        raw = r.json()
                    except Exception as e:
                        print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                        return "Произошла внутренняя ошибка при разборе ответа."
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

            try:
                if r is not None and hasattr(r, "status_code"):
                    if r.status_code in (429, 500, 502, 503, 504) and i == 0:
                        print("TEMP ERROR:", r.status_code, "BODY:", r.text[:400])
                        backoff = min(0.7, max(0.3, remaining() - 0.5))
                        if backoff > 0.3:
                            time.sleep(backoff)
                            continue
                        return "Сервис перегружен. Попробуй ещё раз чуть позже."
                    print("API ERROR:", r.status_code, r.text[:400])
                    return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."
            except Exception:
                return "Произошла внутренняя ошибка при разборе ответа."

        except Exception as e:
            print("REQ ERROR:", e)
            if i == 0 and remaining() > 1.0:
                time.sleep(0.3)
                continue
            break

    return "Сейчас высокая нагрузка, попробуй повторить запрос через минуту."


@app.get("/")
def root():
    return "ok", 200


@app.route("/alice", methods=["POST", "GET", "OPTIONS"])
def alice():
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

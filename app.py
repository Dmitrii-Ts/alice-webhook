from flask import Flask, request, jsonify, send_file, abort
import os
import requests
import time
import re
import json
import datetime
from typing import Any, Dict, List, Optional
import threading
import requests.adapters as req_adapters
import requests.exceptions as req_exc

# Try to use orjson if available (faster), fallback to json
try:
    import orjson as _orjson  # type: ignore
    def loads_bytes(b: bytes):
        return _orjson.loads(b)
    def dumps(obj):
        return _orjson.dumps(obj).decode()
except Exception:
    def loads_bytes(b: bytes):
        return json.loads(b.decode("utf-8"))
    def dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

app = Flask(__name__)

# Configuration from environment with conservative defaults for Starter render plan
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_SAVE_RAW = os.getenv("OPENAI_SAVE_RAW", "0")  # disabled by default for prod on small instances
HARD_DEADLINE_SEC = float(os.getenv("HARD_DEADLINE_SEC", "9.0"))
BASE_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_BASE_MAX_OUTPUT_TOKENS", "300"))
MAX_OUTPUT_TOKENS_CAP = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "512"))
# Limit parallel outbound requests to OpenAI to reduce memory/CPU spikes
OPENAI_MAX_PARALLEL = int(os.getenv("OPENAI_MAX_PARALLEL", "1"))

# Global session and semaphore for connection pooling + concurrency control
session = requests.Session()
adapter = req_adapters.HTTPAdapter(pool_connections=OPENAI_MAX_PARALLEL+2, pool_maxsize=OPENAI_MAX_PARALLEL+2)
session.mount("https://", adapter)
openai_sema = threading.Semaphore(OPENAI_MAX_PARALLEL)

# Simple heuristics to detect which endpoint to use
def should_use_responses_api(model: str) -> bool:
    m = (model or "").lower()
    if not m:
        return True
    if "gpt-3" in m or m.startswith("text-"):
        return False
    return True

# Lightweight extractor: focus on outputs -> message -> content -> output_text
def extract_text_from_response(resp: Dict[str, Any]) -> str:
    try:
        outputs = resp.get("output") or resp.get("outputs") or []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            if isinstance(out, dict) and out.get("type") == "message":
                content = out.get("content") or []
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "output_text":
                            txt = item.get("text")
                            if isinstance(txt, str) and txt.strip():
                                return txt.strip()
                            if isinstance(txt, dict):
                                v = txt.get("value") or txt.get("text")
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
        # fallback: choices.message.content
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            msg = c0.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, dict):
                v = content.get("value") or content.get("text")
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""
    except Exception:
        return ""

def _extract_partial_message_text(raw: Dict[str, Any]) -> Optional[str]:
    try:
        outputs = raw.get("output") or []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            if isinstance(out, dict) and out.get("type") == "message":
                content = out.get("content") or []
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "output_text":
                            tf = item.get("text")
                            if isinstance(tf, str) and tf.strip():
                                parts.append(tf.strip())
                            elif isinstance(tf, dict):
                                v = tf.get("value") or tf.get("text")
                                if isinstance(v, str) and v.strip():
                                    parts.append(v.strip())
                    if parts:
                        return "\n\n".join(parts).strip()
    except Exception:
        pass
    return None

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

# Use the shared session and a tuple timeout (connect, read)
def try_post_with_removing_params(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> requests.Response:
    """
    timeout is the read timeout; connect timeout is short (2s).
    Uses session.post and removes unsupported params on 400 with message hint.
    """
    connect_timeout = 2.0
    max_retries = 2
    tried_removed = set()
    last_resp = None
    for attempt in range(max_retries):
        try:
            with openai_sema:
                last_resp = session.post(url, headers=headers, json=payload, timeout=(connect_timeout, timeout))
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

def ask_openai(utter: str) -> str:
    start = time.monotonic()
    use_responses = should_use_responses_api(MODEL)

    def remaining() -> float:
        return HARD_DEADLINE_SEC - (time.monotonic() - start)

    # Conservative retry counts for Starter
    attempts = 1
    for i in range(attempts):
        left = remaining()
        if left <= 0.2:
            break
        # default read timeout per request
        timeout_read = min(8.0, max(1.0, left - 0.5))

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        if use_responses:
            url = "https://api.openai.com/v1/responses"
            max_output_tokens = BASE_MAX_OUTPUT_TOKENS
            # Single try-with-limited internal retry to avoid heavy loops
            while True:
                rem = remaining()
                if rem <= 0.8:
                    return "Сервис временно недоступен. Попробуй ещё раз позже."
                # per-request read timeout (connect is 2s inside try_post)
                per_req_timeout = max(1.0, min(timeout_read, rem - 0.25))

                payload: Dict[str, Any] = {
                    "model": MODEL,
                    "input": [
                        {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                        {"role": "user", "content": utter},
                    ],
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0.2,
                }

                try:
                    r = try_post_with_removing_params(url, headers, payload, per_req_timeout)
                except req_exc.ReadTimeout:
                    print("Read timed out on attempt with max_output_tokens=", max_output_tokens, "remaining=", rem)
                    return "Сервис временно недоступен. Попробуй ещё раз позже."
                except req_exc.ConnectionError as e:
                    print("Connection error on request:", e)
                    return "Сервис временно недоступен. Попробуй ещё раз позже."
                except Exception as e:
                    print("Request error on request:", e)
                    return "Сервис временно недоступен. Попробуй ещё раз позже."

                print("STATUS:", r.status_code, "URL:", url, "MODEL:", MODEL, "max_output_tokens:", max_output_tokens)

                if r.status_code != 200:
                    # handle transient errors
                    if r.status_code in (429, 500, 502, 503, 504):
                        return "Сервис перегружен. Попробуй ещё раз чуть позже."
                    print("API ERROR:", r.status_code, r.text[:400])
                    return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."

                try:
                    raw = r.json()
                except Exception as e:
                    print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                    return "Произошла внутренняя ошибка при разборе ответа."

                # Debug save only if explicitly enabled
                try:
                    if str(OPENAI_SAVE_RAW).lower() in ("1", "true", "yes"):
                        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                        path = f"/tmp/openai_raw_{ts}.json"
                        with open(path, "w", encoding="utf-8") as _f:
                            _f.write(dumps(raw))
                        print(f"Saved raw OpenAI response to {path}")
                        try:
                            print("RAW (truncated):", str(raw)[:2000])
                        except Exception:
                            pass
                except Exception as _e:
                    print("Failed to save raw response:", _e)

                # If there's a partial assistant message, return it (prefer user-visible text)
                partial = _extract_partial_message_text(raw)
                if partial:
                    return partial

                # Try the standard extractor
                text_from_any = extract_text_from_response(raw)
                if text_from_any:
                    return text_from_any

                # If incomplete due to max_output_tokens, allow one controlled increase if time permits
                status = raw.get("status", "")
                incomplete = raw.get("incomplete_details")
                reason = None
                try:
                    if isinstance(incomplete, dict):
                        reason = incomplete.get("reason")
                except Exception:
                    reason = None

                if status == "incomplete" and reason == "max_output_tokens":
                    if remaining() > 4.0 and max_output_tokens < MAX_OUTPUT_TOKENS_CAP:
                        new_max = min(MAX_OUTPUT_TOKENS_CAP, max_output_tokens * 2)
                        if new_max == max_output_tokens:
                            return "Сервис временно недоступен. Попробуй ещё раз позже."
                        print(f"Response incomplete due to max_output_tokens; retrying with max_output_tokens={new_max}")
                        max_output_tokens = new_max
                        # small pause
                        time.sleep(0.05)
                        continue
                    else:
                        # return partial if any, otherwise inform user
                        print("Responses API status not completed:", status, "reason:", reason)
                        return "Сервис временно недоступен. Попробуй ещё раз позже."

                # nothing parsed
                print("EMPTY PARSE for Responses API. RAW:", raw)
                return "Произошла внутренняя ошибка при разборе ответа."

        else:
            # fallback to chat/completions endpoint (rare path)
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Отвечай кратко и по сути, на русском."},
                    {"role": "user", "content": utter},
                ],
                "temperature": 0.2,
                "max_tokens": BASE_MAX_OUTPUT_TOKENS,
            }
            try:
                r = try_post_with_removing_params(url, headers, payload, timeout_read)
            except Exception as e:
                print("Request error:", e)
                return "Сервис временно недоступен. Попробуй ещё раз позже."

            print("STATUS:", r.status_code, "URL:", url)
            if r.status_code != 200:
                return f"Сервис временно недоступен ({r.status_code}). Попробуй ещё раз позже."
            try:
                raw = r.json()
            except Exception as e:
                print("PARSE ERROR JSON:", e, "BODY:", r.text[:400])
                return "Произошла внутренняя ошибка при разборе ответа."

            text = extract_text_from_response(raw)
            if text:
                return text
            return "Произошла внутренняя ошибка при разборе ответа."

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
    session_data = data.get("session") or {}
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
            "session": session_data,
            "response": {
                "text": text[:1024],
                "tts": text[:1024],
                "end_session": False,
            },
        }
    )

# Temporary debug endpoint to download latest raw (only if you enable RAW_DOWNLOAD_TOKEN in env)
@app.route("/debug/openai_raw/<token>", methods=["GET"])
def download_latest_raw(token: str):
    RAW_DOWNLOAD_TOKEN = os.getenv("RAW_DOWNLOAD_TOKEN", "")
    if not RAW_DOWNLOAD_TOKEN or token != RAW_DOWNLOAD_TOKEN:
        abort(403)
    import glob
    files = sorted(glob.glob("/tmp/openai_raw_*.json"), reverse=True)
    if not files:
        abort(404)
    latest = files[0]
    try:
        return send_file(latest, mimetype="application/json", as_attachment=True, download_name=os.path.basename(latest))
    except Exception:
        abort(500)

if __name__ == "__main__":
    bind_port = os.getenv("PORT", "8080")
    # recommend running with gunicorn in production:
    # gunicorn app:app --workers 1 --threads 2 -k gthread --timeout 30 --bind 0.0.0.0:$PORT
    app.run(host="0.0.0.0", port=int(bind_port))

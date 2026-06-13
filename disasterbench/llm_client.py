"""Real OpenAI-compatible HTTPS LLM client using standard library (no ssl import)."""
import urllib.request
import urllib.error
import json
import time
import re
from typing import Optional


def _redact_sensitive(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"sk-[A-Za-z0-9_\-*]{4,}", "sk-<redacted>", text)
    text = re.sub(r"tvly-[A-Za-z0-9_\-*]{4,}", "tvly-<redacted>", text)
    text = re.sub(r"s2k-[A-Za-z0-9_\-*]{4,}", "s2k-<redacted>", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9_\-./+=]{8,}", "Bearer <redacted>", text)
    return text

class LLMClient:
    def __init__(self, api_base: str, api_key: str, model_id: str,
                 temperature: float = 0.0, max_tokens: int = 2048):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.full_url = f"{self.api_base}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def chat_completion(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.full_url,
            data=payload_bytes,
            headers=self.headers,
            method='POST'
        )

        timeout = 120

        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if resp.status >= 400:
                        if attempt == 1:
                            raise RuntimeError(
                                f"LLM call failed for model {self.model_id}: HTTP {resp.status} {resp.reason} - {_redact_sensitive(raw)}"
                            )
                        time.sleep(2)
                        continue
                    data = json.loads(raw)
                    content = data["choices"][0]["message"]["content"]
                    return content.strip()
            except urllib.error.HTTPError as e:
                if attempt == 1:
                    body = _redact_sensitive(e.read().decode("utf-8"))
                    raise RuntimeError(
                        f"LLM call failed for model {self.model_id}: HTTP {e.code} {e.reason} - {body}"
                    )
                time.sleep(2)
            except (urllib.error.URLError, OSError) as e:
                if attempt == 1:
                    raise RuntimeError(f"LLM call failed for model {self.model_id}: {str(e)}")
                time.sleep(2)
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON response from LLM API for model {self.model_id}")

        raise RuntimeError(f"LLM call failed for model {self.model_id} after retries")

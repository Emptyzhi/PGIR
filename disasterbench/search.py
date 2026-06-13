"""Real Tavily search client – uses HTTPS GET requests."""
import urllib.request
import json
import time
from typing import List, Dict, Any

class SearchClient:
    def __init__(self, api_key: str, max_calls: int = 3):
        self.api_key = api_key
        self.max_calls = max_calls
        self.calls_made = 0
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        if self.calls_made >= self.max_calls:
            return []   # budget exhausted
        payload = json.dumps({
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = 60
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    self.calls_made += 1
                    return data.get("results", [])
            except Exception as e:
                if attempt == 1:
                    raise RuntimeError(f"Tavily search failed: {e}")
                time.sleep(2)
        return []

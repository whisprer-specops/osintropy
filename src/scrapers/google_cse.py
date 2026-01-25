"""scrapers/google_cse.py

Google Programmable Search Engine / Custom Search JSON API wrapper.

This is an *official* programmatic interface for Google search results.
It requires a Programmable Search Engine ID (cx) and an API key.

Free quota: 100 queries/day per project (at time of writing).

Docs:
  - https://developers.google.com/custom-search/v1/overview
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .base_scraper import BaseScraper
from config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


class GoogleCSEScraper(BaseScraper):
    def __init__(self, rate_limit: float = 1.0, timeout_s: float = 12.0):
        super().__init__(
            source_name="google_cse",
            base_url="https://www.googleapis.com/customsearch/v1",
            rate_limit=rate_limit,
        )
        self.timeout_s = timeout_s

    def search_person(self, first_name: str, last_name: str, location: Optional[str] = None) -> Dict[str, Any]:
        # Only run if configured
        if not Config.GOOGLE_CSE_API_KEY or not Config.GOOGLE_CSE_CX:
            return {
                "source": self.source_name,
                "records": [],
                "error": "missing_api_key",
            }

        self._respect_rate_limit()

        q = f'"{first_name} {last_name}"'
        if location:
            q += f' "{location}"'

        params = {
            "key": Config.GOOGLE_CSE_API_KEY,
            "cx": Config.GOOGLE_CSE_CX,
            "q": q,
            "num": min(10, Config.MAX_RESULTS_PER_SITE),
        }

        try:
            logger.info("Google CSE query: %s", q)
            r = requests.get(self.base_url, params=params, timeout=self.timeout_s)
            if r.status_code != 200:
                return {"source": self.source_name, "records": [], "error": f"HTTP {r.status_code}"}

            data = r.json()
            items = data.get("items") or []
            records: List[Dict[str, Any]] = []

            for it in items[: Config.MAX_RESULTS_PER_SITE]:
                if not isinstance(it, dict):
                    continue
                records.append(
                    {
                        "name": f"{first_name} {last_name}",
                        "source": self.source_name,
                        "raw": {
                            "title": it.get("title"),
                            "snippet": it.get("snippet"),
                            "link": it.get("link"),
                            "displayLink": it.get("displayLink"),
                        },
                    }
                )

            return {"source": self.source_name, "records": records, "error": None}

        except Exception as e:
            logger.exception("Google CSE error")
            return {"source": self.source_name, "records": [], "error": str(e)}

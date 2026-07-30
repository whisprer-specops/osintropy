"""scrapers/npi_registry.py

NPPES NPI Registry API (HHS/CMS) scraper.

This is a *public* programmatic API intended for real-time access to NPI data.
It will only return results for individuals/organizations that have an NPI
(i.e., healthcare providers). Useful as a free, legal source of names,
addresses, and sometimes organization/credential hints.

Docs:
  - https://npiregistry.cms.hhs.gov/api-page
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .base_scraper import BaseScraper
from config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


class NPIRegistryScraper(BaseScraper):
    def __init__(self, rate_limit: float = 1.0, timeout_s: float = 12.0):
        super().__init__(
            source_name="npi_registry",
            base_url="https://npiregistry.cms.hhs.gov/api/",
            rate_limit=rate_limit,
        )
        self.timeout_s = timeout_s

    def search_person(self, first_name: str, last_name: str, location: Optional[str] = None) -> Dict[str, Any]:
        self._respect_rate_limit()

        params: Dict[str, Any] = {
            "version": "2.1",
            "first_name": first_name,
            "last_name": last_name,
            "limit": Config.MAX_RESULTS_PER_SITE,
        }

        # Light parsing: allow "city" from location hint if it looks simple.
        if location:
            # If user passes "san francisco" that's city.
            params["city"] = location

        try:
            logger.info("NPI Registry search: %s", self.base_url)
            r = requests.get(self.base_url, params=params, timeout=self.timeout_s)
            if r.status_code != 200:
                return {
                    "source": self.source_name,
                    "records": [],
                    "error": f"HTTP {r.status_code}",
                }

            data = r.json()
            results = data.get("results") or []
            records: List[Dict[str, Any]] = []

            for item in results[: Config.MAX_RESULTS_PER_SITE]:
                basic = item.get("basic") or {}
                addresses = item.get("addresses") or []

                name_parts = [basic.get("first_name"), basic.get("middle_name"), basic.get("last_name")]
                name = " ".join([p for p in name_parts if p])

                # Prefer the "practice" address if available.
                addr = None
                for a in addresses:
                    if (a.get("address_purpose") or "").lower() == "location":
                        addr = a
                        break
                if addr is None and addresses:
                    addr = addresses[0]

                addr_str = None
                if isinstance(addr, dict):
                    parts = [
                        addr.get("address_1"),
                        addr.get("address_2"),
                        addr.get("city"),
                        addr.get("state"),
                        addr.get("postal_code"),
                    ]
                    addr_str = ", ".join([p for p in parts if p]) or None

                # Phones are present in address blocks sometimes
                phones: List[str] = []
                if isinstance(addr, dict) and addr.get("telephone_number"):
                    phones.append(str(addr.get("telephone_number")))

                if not name:
                    continue

                records.append(
                    {
                        "name": name,
                        "address": addr_str,
                        "phones": phones,
                        "source": self.source_name,
                        "raw": {
                            "enumeration_type": item.get("enumeration_type"),
                            "number": item.get("number"),
                            "taxonomies": item.get("taxonomies"),
                        },
                    }
                )

            return {"source": self.source_name, "records": records, "error": None}

        except Exception as e:
            logger.exception("NPI Registry error")
            return {"source": self.source_name, "records": [], "error": str(e)}

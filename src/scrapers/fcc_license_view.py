"""scrapers/fcc_license_view.py

FCC License View API scraper.

This uses the FCC "license-view" API, which is a U.S. government dataset intended
for programmatic access. It can return licensing records (e.g., amateur radio),
which often include a licensee name and a mailing address.

Endpoint (JSON):
  https://data.fcc.gov/api/license-view/basicSearch/getLicenses?searchValue=...&format=json
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_scraper import BaseScraper
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class FCCLicenseViewScraper(BaseScraper):
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.source_name = "fcc_license_view"
        self.base_url = "https://data.fcc.gov/api/license-view/basicSearch/getLicenses"
        self.timeout_s = float(getattr(self.config, "FCC_TIMEOUT_S", 30.0))

        # A conservative session: retries on transient errors, no "bot bypass" behavior.
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _build_search_value(self, first_name: str, last_name: str) -> str:
        first = (first_name or "").strip()
        last = (last_name or "").strip()
        if first and last:
            return f"{first} {last}"
        return (first or last).strip()

    def search_person(self, first_name: str, last_name: str, location: str) -> Dict[str, Any]:
        """
        Returns an envelope:
          {"source": "fcc_license_view", "records": [...], "error": "...optional..."}
        """
        search_value = self._build_search_value(first_name, last_name)
        if not search_value:
            return {"source": self.source_name, "records": [], "error": "missing_name"}

        params = {"searchValue": search_value, "format": "json"}

        try:
            logger.info("FCC License View search: %s", self.base_url)
            r = self.session.get(
                self.base_url,
                params=params,
                timeout=(5.0, self.timeout_s),  # (connect, read)
                allow_redirects=True,
                headers={"User-Agent": "osintropy/1.0 (+https://example.invalid)"},
            )
        except requests.RequestException as e:
            logger.error("FCC License View request error: %s", e)
            return {"source": self.source_name, "records": [], "error": str(e)}

        if r.status_code != 200:
            logger.error("FCC License View returned status %s", r.status_code)
            return {"source": self.source_name, "records": [], "error": f"HTTP {r.status_code}"}

        try:
            data = r.json()
        except ValueError:
            logger.error("FCC License View returned non-JSON response")
            return {"source": self.source_name, "records": [], "error": "non_json_response"}

        licenses = ((data.get("Licenses") or {}).get("License")) or []
        if isinstance(licenses, dict):
            licenses = [licenses]
        if not isinstance(licenses, list):
            licenses = []

        records: List[Dict[str, Any]] = []
        for lic in licenses:
            if not isinstance(lic, dict):
                continue

            name = (lic.get("licName") or lic.get("entityName") or "").strip()
            # Address fields vary; we stitch something reasonable.
            addr1 = (lic.get("licStreet") or lic.get("streetAddress") or "").strip()
            city = (lic.get("licCity") or lic.get("city") or "").strip()
            state = (lic.get("licState") or lic.get("state") or "").strip()
            zipc = (lic.get("licZip") or lic.get("zip") or "").strip()
            callsign = (lic.get("callsign") or lic.get("callSign") or lic.get("CallSign") or "").strip()

            full_address_parts = [p for p in [addr1, city, state, zipc] if p]
            full_address = ", ".join(full_address_parts).strip()

            rec: Dict[str, Any] = {
                "source": self.source_name,
                "full_name": name or f"{first_name} {last_name}".strip(),
            }
            if full_address:
                rec["address"] = full_address
            if city or state:
                rec["location"] = ", ".join([p for p in [city, state] if p])
            if callsign:
                rec["fcc_callsign"] = callsign

            records.append(rec)

        return {"source": self.source_name, "records": records}

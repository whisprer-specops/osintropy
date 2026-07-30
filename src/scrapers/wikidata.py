"""scrapers/wikidata.py

Wikidata programmatic lookups.

This uses Wikidata's MediaWiki Action API to search for entities by label
and then fetches entity JSON for a small set of properties that can sometimes
include employer/occupation and social handles.

Notes:
 - This is best for *notable* individuals.
 - It will often return nothing for ordinary private persons.

API docs:
  - https://www.wikidata.org/wiki/Wikidata:Data_access
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .base_scraper import BaseScraper
from config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


class WikidataScraper(BaseScraper):
    def __init__(self, rate_limit: float = 1.0, timeout_s: float = 12.0):
        super().__init__(
            source_name="wikidata",
            base_url="https://www.wikidata.org/w/api.php",
            rate_limit=rate_limit,
        )
        self.timeout_s = timeout_s

    def search_person(self, first_name: str, last_name: str, location: Optional[str] = None) -> Dict[str, Any]:
        self._respect_rate_limit()

        query = f"{first_name} {last_name}".strip()
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": min(5, Config.MAX_RESULTS_PER_SITE),
            "type": "item",
        }

        try:
            logger.info("Wikidata search: %s", query)
            r = requests.get(self.base_url, params=params, timeout=self.timeout_s)
            if r.status_code != 200:
                return {"source": self.source_name, "records": [], "error": f"HTTP {r.status_code}"}

            data = r.json()
            results = data.get("search") or []
            records: List[Dict[str, Any]] = []

            for hit in results:
                qid = hit.get("id")
                label = hit.get("label")
                if not qid or not label:
                    continue

                entity = self._fetch_entity(qid)
                if not entity:
                    continue

                # Pull a few potentially useful properties (best-effort)
                props = self._extract_props(entity)

                records.append(
                    {
                        "name": label,
                        "location": props.get("place_of_birth") or props.get("residence"),
                        "source": self.source_name,
                        "raw": {
                            "qid": qid,
                            "description": hit.get("description"),
                            **props,
                        },
                    }
                )

            return {"source": self.source_name, "records": records, "error": None}

        except Exception as e:
            logger.exception("Wikidata error")
            return {"source": self.source_name, "records": [], "error": str(e)}

    def _fetch_entity(self, qid: str) -> Optional[Dict[str, Any]]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        try:
            self._respect_rate_limit()
            r = requests.get(url, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            data = r.json()
            entities = data.get("entities") or {}
            ent = entities.get(qid)
            return ent if isinstance(ent, dict) else None
        except Exception:
            return None

    def _extract_props(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a small set of properties as human-readable strings.

        We deliberately keep this small and best-effort; Wikidata is huge.
        """

        def get_claim_ids(prop: str) -> List[str]:
            claims = (ent.get("claims") or {}).get(prop)
            if not isinstance(claims, list):
                return []
            out: List[str] = []
            for c in claims:
                dv = ((c.get("mainsnak") or {}).get("datavalue") or {})
                val = dv.get("value")
                if isinstance(val, dict) and val.get("id"):
                    out.append(str(val["id"]))
                elif isinstance(val, str):
                    out.append(val)
            return out

        def resolve_label(qid: str) -> Optional[str]:
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            try:
                self._respect_rate_limit()
                r = requests.get(url, timeout=self.timeout_s)
                if r.status_code != 200:
                    return None
                data = r.json()
                ent2 = (data.get("entities") or {}).get(qid) or {}
                labels = ent2.get("labels") or {}
                en = labels.get("en") or {}
                return en.get("value")
            except Exception:
                return None

        # Property IDs:
        # P106 occupation, P108 employer, P19 place of birth, P551 residence
        # P2002 Twitter, P2397 YouTube channel id, P2013 Facebook id, P2949 Wikipedia username
        props: Dict[str, Any] = {}

        occ_ids = get_claim_ids("P106")
        emp_ids = get_claim_ids("P108")
        pob_ids = get_claim_ids("P19")
        res_ids = get_claim_ids("P551")

        # Social handles sometimes are plain strings
        def get_string(prop: str) -> Optional[str]:
            claims = (ent.get("claims") or {}).get(prop)
            if not isinstance(claims, list) or not claims:
                return None
            dv = ((claims[0].get("mainsnak") or {}).get("datavalue") or {})
            val = dv.get("value")
            return str(val) if val else None

        props["occupation"] = [resolve_label(i) or i for i in occ_ids][:3] if occ_ids else None
        props["employer"] = [resolve_label(i) or i for i in emp_ids][:3] if emp_ids else None
        props["place_of_birth"] = resolve_label(pob_ids[0]) if pob_ids else None
        props["residence"] = resolve_label(res_ids[0]) if res_ids else None
        props["twitter"] = get_string("P2002")
        props["youtube_channel"] = get_string("P2397")
        props["facebook"] = get_string("P2013")
        props["wikipedia_username"] = get_string("P2949")

        # Drop empty keys
        props = {k: v for k, v in props.items() if v}
        if not props:
            return {"source": self.source_name, "records": [], "error": None}

        record: Dict[str, Any] = {"name": query, "source": self.source_name}
        # Map known keys into the record.
        if "employer" in props:
            record["employers"] = props["employer"] if isinstance(props["employer"], list) else [props["employer"]]
        if "occupation" in props:
            record["occupations"] = props["occupation"] if isinstance(props["occupation"], list) else [props["occupation"]]

        # Social handles
        social_profiles: Dict[str, str] = {}
        for k in ["twitter", "youtube_channel", "facebook", "wikipedia_username"]:
            v = props.get(k)
            if v:
                social_profiles[k] = str(v)
        if social_profiles:
            record["social_profiles"] = social_profiles

        # Residence / place of birth as location hints
        if "residence" in props:
            record["location"] = props["residence"]
        elif "place_of_birth" in props:
            record["location"] = props["place_of_birth"]

        return {"source": self.source_name, "records": [record], "error": None}


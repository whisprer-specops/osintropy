# =====================================
# aggregation/aggregator.py - Main Aggregator
# =====================================
"""
Main aggregation engine that combines all components.

Key behavior:
- Each scraper may return either:
    * an envelope dict: {"source": str, "records": [dict, ...], "error": str|None, ...}
    * a list of record dicts
    * a single record dict
- We normalize all those shapes into a flat List[Dict] of record dicts before merging.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from core.models import PersonRecord, Address, PhoneNumber
from core.database import Database
from scrapers import get_scraper
from aggregation.matcher import RecordMatcher
from analysis.risk_assessment import RiskAssessor
from utils.rate_limiter import RateLimiter
from config import Config


class OSINTAggregator:
    """Main aggregation engine."""

    def __init__(self, db_path: Optional[str] = None, include_brokers: bool = False) -> None:
        self.db = Database(db_path or Config.DATABASE_PATH)
        self.matcher = RecordMatcher()
        self.risk_assessor = RiskAssessor()
        self.rate_limiter = RateLimiter()
        self.include_brokers = include_brokers

        # Initialize scrapers
        self.scrapers = self._initialize_scrapers()

    def _initialize_scrapers(self) -> Dict[str, Any]:
        """Initialize all available scrapers."""
        scrapers: Dict[str, Any] = {}

        # Prefer sources that are designed for programmatic access.
        # (The people-broker websites often return 403 due to bot detection.)
        default_sites = [
            "npi_registry",
            "fcc_license_view",
            "wikidata",
            "google_cse",
        ]

        if self.include_brokers:
            default_sites.extend([
                "truepeoplesearch",
                "whitepages",
                "spokeo",
                "beenverified",
            ])

        for site in default_sites:
            try:
                scrapers[site] = get_scraper(site)
            except Exception as e:  # keep going even if a scraper fails to init
                print(f"Failed to initialize {site} scraper: {e}")

        return scrapers

    def _normalize_results(self, site: str, results_raw: Any) -> List[Dict[str, Any]]:
        """Normalize scraper outputs into a flat list of record dicts."""
        out: List[Dict[str, Any]] = []

        if results_raw is None:
            return out

        # Envelope dict (the normal shape in this project)
        if isinstance(results_raw, dict):
            records = results_raw.get("records")
            if isinstance(records, list):
                src = results_raw.get("source", site)
                for rec in records:
                    if isinstance(rec, dict):
                        r = dict(rec)
                        r.setdefault("source", src)
                        out.append(r)
                return out

            # Single record dict fallback
            r = dict(results_raw)
            r.setdefault("source", site)
            out.append(r)
            return out

        # List of dict records (or envelopes)
        if isinstance(results_raw, list):
            for item in results_raw:
                if isinstance(item, dict) and isinstance(item.get("records"), list):
                    src = item.get("source", site)
                    for rec in item["records"]:
                        if isinstance(rec, dict):
                            r = dict(rec)
                            r.setdefault("source", src)
                            out.append(r)
                    continue

                if isinstance(item, dict):
                    r = dict(item)
                    r.setdefault("source", site)
                    out.append(r)

            return out

        # Unknown type => ignore
        return out

    def search_person(
        self,
        first_name: str,
        last_name: str,
        location: Optional[str] = None,
        sites: Optional[List[str]] = None,
    ) -> Optional[PersonRecord]:
        """Search for a person across multiple sites."""
        if sites is None:
            sites = list(self.scrapers.keys())

        all_results: List[Dict[str, Any]] = []

        for site in sites:
            scraper = self.scrapers.get(site)
            if scraper is None:
                continue

            # Rate limiting (best-effort)
            self.rate_limiter.wait_if_needed(site)

            results_raw = scraper.search(
                first_name=first_name,
                last_name=last_name,
                location=location,
            )

            all_results.extend(self._normalize_results(site, results_raw))

        if not all_results:
            return None

        record = self._aggregate_results(all_results)

        # Calculate risk indicators
        record.risk_indicators = self.risk_assessor.assess(record)

        # Save to database
        self.db.save_record(record)

        return record

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> PersonRecord:
        """Aggregate search results into unified record."""
        record = self._find_or_create_record(results)

        for result in results:
            self._merge_result_into_record(result, record)

        return record

    def _find_or_create_record(self, results: List[Dict[str, Any]]) -> PersonRecord:
        """Find an existing record (if similar) or create a new one."""
        for result in results:
            existing = self.db.find_similar_records(result)
            if existing:
                best_match = self.matcher.find_best_match(result, existing)
                if best_match:
                    return best_match

        primary_id = self._generate_record_id(results[0])
        return PersonRecord(primary_id=primary_id)

    def _generate_record_id(self, data: Dict[str, Any]) -> str:
        """Generate a stable-ish ID for a record."""
        components: List[str] = []
        if data.get("name"):
            components.append(str(data["name"]))
        if data.get("phone"):
            components.append(str(data["phone"]))

        id_string = "|".join(components) if components else repr(sorted(data.items()))
        return hashlib.sha256(id_string.encode("utf-8")).hexdigest()[:16]

    def _merge_result_into_record(self, result: Dict[str, Any], record: PersonRecord) -> None:
        """Merge a single scraper record dict into the unified PersonRecord."""
        if not isinstance(result, dict):
            return

        src = result.get("source", "unknown")

        # Name
        name = result.get("name")
        if name:
            record.names.add(str(name))

        # Address/location
        addr_val = result.get("address") or result.get("location")
        if addr_val:
            record.addresses.append(Address(full_address=str(addr_val), sources={src}))

        # Phone(s)
        phone_val = result.get("phone")
        if phone_val:
            record.phone_numbers.append(PhoneNumber(number=str(phone_val), sources={src}))

        phones = result.get("phones")
        if isinstance(phones, list):
            for p in phones:
                if p:
                    record.phone_numbers.append(PhoneNumber(number=str(p), sources={src}))

        
        # Employers / occupations
        emp = result.get("employer") or result.get("employers")
        if isinstance(emp, str) and emp:
            record.employers.add(emp)
        elif isinstance(emp, list):
            for e in emp:
                if isinstance(e, str) and e:
                    record.employers.add(e)

        occ = result.get("occupation") or result.get("occupations")
        if isinstance(occ, str) and occ:
            record.occupations.add(occ)
        elif isinstance(occ, list):
            for o in occ:
                if isinstance(o, str) and o:
                    record.occupations.add(o)

        # Social handles / profiles
        socials = result.get("social_profiles")
        if isinstance(socials, dict):
            for k, v in socials.items():
                if k and v:
                    record.social_profiles[str(k)] = str(v)

        # Common wikidata-style keys
        for k in ["twitter", "facebook", "youtube_channel", "wikipedia_username", "instagram", "github", "linkedin"]:
            v = result.get(k)
            if v:
                record.social_profiles[k] = str(v)

        # URLs / mentions
        urls = result.get("urls")
        if isinstance(urls, list):
            for u in urls:
                if u:
                    record.urls.add(str(u))
        elif isinstance(urls, str) and urls:
            record.urls.add(urls)

        mentions = result.get("web_mentions")
        if isinstance(mentions, list):
            for m in mentions:
                if isinstance(m, dict):
                    record.web_mentions.append(m)

# Sources
        record.sources.add(src)

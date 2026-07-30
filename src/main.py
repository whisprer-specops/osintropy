"""
main.py - Main Entry Point

CLI entry point for the OSINT Aggregator project.

This file is intentionally defensive:
- If your real project modules exist (`aggregation.aggregator`, `analysis.report_generator`),
  it will use them.
- If they are missing (e.g., you only have this single file), it can run in `--demo` mode
  with lightweight stubs so the script is still executable and you can validate wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Optional real imports
# -----------------------------
_REAL_IMPORTS_OK = True
_IMPORT_ERR: Optional[BaseException] = None

try:
    from aggregation.aggregator import OSINTAggregator  # type: ignore
    from analysis.report_generator import ReportGenerator  # type: ignore
except BaseException as e:  # noqa: BLE001 (we want to catch anything import-related)
    _REAL_IMPORTS_OK = False
    _IMPORT_ERR = e


# -----------------------------
# Demo-only stubs (only used if real imports are missing and --demo is set)
# -----------------------------
@dataclass
class _DemoRecord:
    first_name: str
    last_name: str
    location: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class _DemoOSINTAggregator:
    """A tiny stand-in for wiring tests when you don't have the rest of the project."""

    def search_person(self, first_name: str, last_name: str, location: str) -> Optional[_DemoRecord]:
        # In demo mode, we return a deterministic example record.
        return _DemoRecord(
            first_name=first_name,
            last_name=last_name,
            location=location,
            sources=[
                {"type": "demo", "name": "stub_source", "note": "Replace with real sources once modules exist."}
            ],
            confidence_scores={"overall": 0.42},
        )


class _DemoReportGenerator:
    def generate(self, record: _DemoRecord) -> Dict[str, Any]:
        return {
            "person": {
                "first_name": record.first_name,
                "last_name": record.last_name,
                "location": record.location,
            },
            "sources": record.sources,
            "confidence_scores": record.confidence_scores,
            "generated_by": "demo_stub",
        }

    def save_report(self, report: Dict[str, Any], output_path: str | Path) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# CLI + main
# -----------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OSINT Aggregator - search and generate a report.")
    p.add_argument("--first-name", default="John", help="First name to search for.")
    p.add_argument("--last-name", default="Smith", help="Last name to search for.")
    p.add_argument("--location", default="New York", help="Location hint (city/region).")
    p.add_argument(
        "--output",
        default="output/john_smith_report.json",
        help="Output report path (JSON).",
    )
    p.add_argument(
        "--include-brokers",
        action="store_true",
        help="Include broker/people-search scrapers that frequently block automation (may return 403). Off by default.",
    )

    p.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo stubs if project modules are missing.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if _REAL_IMPORTS_OK:
        aggregator = OSINTAggregator(include_brokers=args.include_brokers)
        report_gen = ReportGenerator()
    else:
        if not args.demo:
            print("ERROR: Project modules could not be imported.", file=sys.stderr)
            print("  Missing imports: aggregation.aggregator / analysis.report_generator", file=sys.stderr)
            print(f"  Import error: {_IMPORT_ERR!r}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Fix options:", file=sys.stderr)
            print("  1) Run from the project root so packages resolve (python -m main).", file=sys.stderr)
            print("  2) Ensure packages have __init__.py and are on PYTHONPATH.", file=sys.stderr)
            print("  3) Or run this script with --demo to validate the CLI wiring.", file=sys.stderr)
            return 2

        aggregator = _DemoOSINTAggregator()
        report_gen = _DemoReportGenerator()

    record = aggregator.search_person(
        first_name=args.first_name,
        last_name=args.last_name,
        location=args.location,
    )

    if record:
        report = report_gen.generate(record)
        report_gen.save_report(report, args.output)

        # We try to be compatible with both the real record type and the demo record type.
        sources = getattr(record, "sources", []) or []
        confidence_scores = getattr(record, "confidence_scores", {}) or {}
        overall = float(confidence_scores.get("overall", 0.0))

        print(f"Found record with {len(sources)} sources")
        print(f"Confidence: {overall:.2%}")
        print(f"Report written to: {args.output}")
        return 0

    print("No records found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

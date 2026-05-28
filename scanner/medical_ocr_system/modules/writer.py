"""
writer.py  –  Write all pipeline outputs to disk.

Outputs produced for every run:
  <stem>_data.csv              – primary structured extraction (all fields)
  <stem>_sanity_report.txt     – sanity-check summary + flagged rows
  <stem>_inconsistencies.txt   – all detected inconsistencies
  <stem>_full_report.json      – machine-readable combined output
"""

from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# CSV  (primary extraction output)
# ──────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "field_name",
    "value",
    "exact_quote",
    "char_start",
    "char_end",
    "page_num",
    "sanity_status",
    "confidence",
    "notes",
]


def write_csv(fields, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for f in fields:
            writer.writerow(
                {
                    "field_name":    f.field_name,
                    "value":         f.value,
                    "exact_quote":   f.exact_quote,
                    "char_start":    f.char_start,
                    "char_end":      f.char_end,
                    "page_num":      f.page_num,
                    "sanity_status": f.sanity_status,
                    "confidence":    f.confidence,
                    "notes":         f.notes,
                }
            )

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Sanity report
# ──────────────────────────────────────────────────────────────────────────────

def write_sanity_report(
    fields,
    summary: dict,
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 72,
        "  SANITY CHECK REPORT",
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
        f"  Total fields extracted : {len(fields)}",
        f"  ✓  VERIFIED  : {summary.get('VERIFIED', 0)}",
        f"  ~  FUZZY     : {summary.get('FUZZY', 0)}  (near-match, possible OCR noise)",
        f"  ✗  NOT_FOUND : {summary.get('NOT_FOUND', 0)}  ← LIKELY HALLUCINATIONS",
        f"  -  SKIPPED   : {summary.get('SKIPPED', 0)}  (no quote provided)",
        "",
    ]

    not_found = [f for f in fields if f.sanity_status == "NOT_FOUND"]
    fuzzy     = [f for f in fields if f.sanity_status == "FUZZY"]

    if not_found:
        lines += [
            "─" * 72,
            "  ✗  HALLUCINATION CANDIDATES  (exact_quote not found in source)",
            "─" * 72,
        ]
        for f in not_found:
            lines += [
                f"  Field   : {f.field_name}",
                f"  Value   : {f.value}",
                f"  Quote   : {repr(f.exact_quote[:120])}",
                f"  Conf.   : {f.confidence}",
                "",
            ]

    if fuzzy:
        lines += [
            "─" * 72,
            "  ~  FUZZY MATCHES  (verify manually – may be OCR artefacts)",
            "─" * 72,
        ]
        for f in fuzzy:
            lines += [
                f"  Field   : {f.field_name}",
                f"  Value   : {f.value}",
                f"  Quote   : {repr(f.exact_quote[:120])}",
                f"  Chars   : [{f.char_start}:{f.char_end}]  Page {f.page_num}",
                "",
            ]

    lines += ["=" * 72, "  END OF SANITY REPORT", "=" * 72]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Inconsistency report
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_ICON = {"ERROR": "✗ ERROR", "WARNING": "⚠ WARNING", "INFO": "ℹ INFO"}


def write_inconsistency_report(
    inconsistencies,
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 72,
        "  INCONSISTENCY REPORT",
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
        f"  Total flags : {len(inconsistencies)}",
        f"  Errors      : {sum(1 for i in inconsistencies if i.severity == 'ERROR')}",
        f"  Warnings    : {sum(1 for i in inconsistencies if i.severity == 'WARNING')}",
        f"  Info        : {sum(1 for i in inconsistencies if i.severity == 'INFO')}",
        "",
    ]

    for idx, inc in enumerate(inconsistencies, start=1):
        icon = SEVERITY_ICON.get(inc.severity, inc.severity)
        lines += [
            "─" * 72,
            f"  [{idx}]  {icon}  –  {inc.rule}",
            f"  {inc.description}",
        ]
        if inc.field_names:
            lines.append(f"  Fields  : {', '.join(inc.field_names)}")
        for q in inc.quotes:
            lines.append(f"  Quote   : {repr(q[:120])}")
        lines.append("")

    if not inconsistencies:
        lines += ["  No inconsistencies detected.", ""]

    lines += ["=" * 72, "  END OF INCONSISTENCY REPORT", "=" * 72]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Full JSON output
# ──────────────────────────────────────────────────────────────────────────────

def write_json(
    fields,
    sanity_summary: dict,
    inconsistencies,
    ocr_source: str,
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ocr_source": ocr_source,
        "sanity_summary": sanity_summary,
        "inconsistency_count": len(inconsistencies),
        "fields": [
            {
                "field_name":    f.field_name,
                "value":         f.value,
                "exact_quote":   f.exact_quote,
                "char_start":    f.char_start,
                "char_end":      f.char_end,
                "page_num":      f.page_num,
                "sanity_status": f.sanity_status,
                "confidence":    f.confidence,
                "notes":         f.notes,
            }
            for f in fields
        ],
        "inconsistencies": [
            {
                "rule":        i.rule,
                "severity":    i.severity,
                "description": i.description,
                "field_names": i.field_names,
                "quotes":      i.quotes,
            }
            for i in inconsistencies
        ],
    }

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

"""
inconsistency_detector.py  –  Flag logical/clinical contradictions in a
medical report.

Two parallel sub-systems:
  1. Rule-based  (fast, deterministic) – checks date ordering, unit ranges,
                  duplicate field values that conflict, etc.
  2. LLM-based   (Claude API)          – semantic reasoning about contradictions

Both return a list of Inconsistency objects.  Results are merged and de-duped.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Inconsistency:
    rule: str            # short label, e.g. "DATE_ORDER" or "LLM_SEMANTIC"
    severity: str        # "ERROR" | "WARNING" | "INFO"
    description: str     # human-readable explanation
    field_names: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect(
    fields,               # List[ExtractedField]
    ocr_text: str,
    api_key: str | None = None,
    skip_llm: bool = False,
) -> List[Inconsistency]:
    """
    Run both detectors and return merged results.
    Set skip_llm=True for offline / unit-test mode.
    """
    issues: List[Inconsistency] = []
    issues.extend(_rule_based(fields))

    if not skip_llm:
        try:
            issues.extend(_llm_based(fields, ocr_text, api_key))
        except Exception as exc:
            issues.append(
                Inconsistency(
                    rule="LLM_ERROR",
                    severity="WARNING",
                    description=f"LLM inconsistency check failed: {exc}",
                )
            )

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Rule-based checks
# ──────────────────────────────────────────────────────────────────────────────

# Physiologically plausible ranges  (min, max)
VITAL_RANGES = {
    "heart_rate":        (20,   300),
    "hr":                (20,   300),
    "blood_pressure_systolic":  (50, 300),
    "sbp":               (50,   300),
    "blood_pressure_diastolic": (20, 200),
    "dbp":               (20,   200),
    "temperature":       (32.0, 42.5),  # °C
    "temp":              (32.0, 42.5),
    "spo2":              (50,   100),
    "oxygen_saturation": (50,   100),
    "respiratory_rate":  (4,    80),
    "rr":                (4,    80),
    "bmi":               (10,   80),
    "weight_kg":         (1,    500),
    "height_cm":         (30,   250),
}

DATE_FIELDS_ORDER = [
    ["date_of_birth", "encounter_date"],
    ["date_of_birth", "report_date"],
    ["admission_date", "discharge_date"],
    ["symptom_onset_date", "encounter_date"],
]


def _rule_based(fields) -> List[Inconsistency]:
    issues: List[Inconsistency] = []
    field_map: dict[str, list] = {}

    for f in fields:
        field_map.setdefault(f.field_name.lower(), []).append(f)

    # ── Duplicate / conflicting values for the same field ──
    for name, flist in field_map.items():
        if len(flist) > 1:
            values = [f.value.strip().lower() for f in flist]
            if len(set(values)) > 1:
                issues.append(
                    Inconsistency(
                        rule="DUPLICATE_CONFLICT",
                        severity="ERROR",
                        description=(
                            f"Field '{name}' appears {len(flist)} times with "
                            f"conflicting values: {list(set(values))}"
                        ),
                        field_names=[name],
                        quotes=[f.exact_quote for f in flist],
                    )
                )

    # ── Vital sign range checks ──
    for name, flist in field_map.items():
        key = name.replace(" ", "_")
        if key in VITAL_RANGES:
            lo, hi = VITAL_RANGES[key]
            for f in flist:
                try:
                    num = float(re.sub(r"[^\d.\-]", "", f.value))
                    if not (lo <= num <= hi):
                        issues.append(
                            Inconsistency(
                                rule="VITAL_OUT_OF_RANGE",
                                severity="ERROR",
                                description=(
                                    f"{name} value {num} is outside plausible range "
                                    f"[{lo}, {hi}]"
                                ),
                                field_names=[name],
                                quotes=[f.exact_quote],
                            )
                        )
                except (ValueError, TypeError):
                    pass

    # ── Date ordering checks ──
    for pair in DATE_FIELDS_ORDER:
        early_key, late_key = pair[0], pair[1]
        early_fields = field_map.get(early_key, [])
        late_fields  = field_map.get(late_key, [])
        if not early_fields or not late_fields:
            continue
        try:
            early_dt = _parse_date(early_fields[0].value)
            late_dt  = _parse_date(late_fields[0].value)
            if early_dt and late_dt and early_dt > late_dt:
                issues.append(
                    Inconsistency(
                        rule="DATE_ORDER",
                        severity="ERROR",
                        description=(
                            f"'{early_key}' ({early_fields[0].value}) is AFTER "
                            f"'{late_key}' ({late_fields[0].value})"
                        ),
                        field_names=[early_key, late_key],
                        quotes=[early_fields[0].exact_quote, late_fields[0].exact_quote],
                    )
                )
        except Exception:
            pass

    # ── Sex / gender vs. diagnosis check (simple heuristic) ──
    sex_fields = field_map.get("sex", field_map.get("gender", []))
    if sex_fields:
        sex_val = sex_fields[0].value.lower()
        all_diags = " ".join(
            f.value.lower()
            for fname, flist in field_map.items()
            if "diagnosis" in fname or "icd" in fname
            for f in flist
        )
        male_only   = ["prostate", "testicular", "epididymitis", "varicocele"]
        female_only = ["ovarian", "uterine", "endometriosis", "cervical", "eclampsia"]
        if "female" in sex_val or "f" == sex_val:
            for term in male_only:
                if term in all_diags:
                    issues.append(
                        Inconsistency(
                            rule="SEX_DIAGNOSIS_MISMATCH",
                            severity="WARNING",
                            description=f"Patient marked female but diagnosis contains '{term}'",
                            field_names=["sex"] + [
                                fn for fn in field_map if "diagnosis" in fn
                            ],
                        )
                    )
        elif "male" in sex_val or "m" == sex_val:
            for term in female_only:
                if term in all_diags:
                    issues.append(
                        Inconsistency(
                            rule="SEX_DIAGNOSIS_MISMATCH",
                            severity="WARNING",
                            description=f"Patient marked male but diagnosis contains '{term}'",
                            field_names=["sex"] + [
                                fn for fn in field_map if "diagnosis" in fn
                            ],
                        )
                    )

    return issues


def _parse_date(value: str):
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LLM-based semantic consistency check
# ──────────────────────────────────────────────────────────────────────────────

LLM_SYSTEM = """You are a clinical-data quality auditor.

You will receive:
  A) The extracted structured fields from a medical report (as JSON).
  B) The raw OCR text of the report.

Your task: identify LOGICAL, CLINICAL, or FACTUAL inconsistencies.

Return ONLY a valid JSON array (no markdown, no explanation).
Each element must have:
{
  "rule":        <short_snake_case label>,
  "severity":    <"ERROR" | "WARNING" | "INFO">,
  "description": <clear, concise explanation of the inconsistency>,
  "field_names": [<list of involved field_names>],
  "quotes":      [<relevant verbatim text snippets from the report>]
}

If there are NO inconsistencies, return an empty array: []

Examples of things to look for:
- Medication dose/route/frequency that is clinically unusual or dangerous
- Lab values that conflict with stated diagnosis
- Age / DOB inconsistencies
- Contradictory statements within the free text
- Abnormal lab values not mentioned in the assessment
- Drug allergies conflicting with prescribed medications
"""


def _llm_based(
    fields,
    ocr_text: str,
    api_key: str | None = None,
) -> List[Inconsistency]:
    from openai import OpenAI

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not set")

    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    fields_json = json.dumps(
        [
            {
                "field_name": f.field_name,
                "value": f.value,
                "exact_quote": f.exact_quote,
            }
            for f in fields
        ],
        indent=2,
    )

    user_msg = (
        "=== EXTRACTED FIELDS ===\n"
        f"{fields_json}\n\n"
        "=== RAW OCR TEXT ===\n"
        f"{ocr_text}"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip(), flags=re.MULTILINE)

    items = json.loads(cleaned)
    if not isinstance(items, list):
        items = []

    return [
        Inconsistency(
            rule=str(item.get("rule", "LLM_CHECK")),
            severity=str(item.get("severity", "WARNING")),
            description=str(item.get("description", "")),
            field_names=list(item.get("field_names", [])),
            quotes=list(item.get("quotes", [])),
        )
        for item in items
        if isinstance(item, dict)
    ]

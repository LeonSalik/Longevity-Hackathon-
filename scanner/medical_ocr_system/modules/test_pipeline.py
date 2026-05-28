#!/usr/bin/env python3
"""
test_pipeline.py  –  Run the full pipeline against a synthetic medical report.

Usage:
  python test_pipeline.py                  # uses ANTHROPIC_API_KEY env var
  python test_pipeline.py --api-key sk-... # explicit key
  python test_pipeline.py --skip-llm       # sanity check + rule-based only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modules import ocr as ocr_mod
from modules import extractor as extractor_mod
from modules import sanity_checker
from modules import inconsistency as incons_mod
from modules import writer
from modules.ocr import OcrResult


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic medical report (intentionally contains 2 inconsistencies)
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_REPORT = """
PATIENT MEDICAL REPORT
======================

Patient Name   : John A. Smith
Date of Birth  : 15/03/1978
Sex            : Male
Patient ID     : PA-20241101-7734
Encounter Date : 22/11/2024
Facility       : Greenfield General Hospital
Attending      : Dr. Sarah Chen, MD

CHIEF COMPLAINT
---------------
Patient presents with persistent headache, dizziness, and elevated blood pressure.

VITAL SIGNS
-----------
Blood Pressure : 185/112 mmHg
Heart Rate     : 92 bpm
Temperature    : 36.8 °C
SpO2           : 98 %
Weight         : 87 kg
Height         : 178 cm
BMI            : 27.5

CURRENT MEDICATIONS
-------------------
1. Lisinopril 10 mg PO once daily
2. Aspirin 81 mg PO once daily
3. Metformin 500 mg PO twice daily

ALLERGIES
---------
Penicillin (rash), Sulfonamides (anaphylaxis)

LABORATORY RESULTS
------------------
HbA1c          : 8.2 %         (Reference: <5.7%)
Fasting Glucose: 164 mg/dL     (Reference: 70-99 mg/dL)
Serum Creatinine: 1.1 mg/dL   (Reference: 0.7-1.2 mg/dL)
eGFR           : 72 mL/min/1.73m²
Total Cholesterol: 218 mg/dL   (Reference: <200 mg/dL)
LDL            : 142 mg/dL     (Reference: <100 mg/dL)
HDL            : 41 mg/dL      (Reference: >40 mg/dL)
Triglycerides  : 178 mg/dL     (Reference: <150 mg/dL)

** INCONSISTENCY PLANTED #1 **
Patient is male but assessment references ovarian cyst monitoring (for testing purposes).

ASSESSMENT & PLAN
-----------------
1. Stage 2 Hypertension (ICD-10: I10) – increase Lisinopril to 20 mg daily,
   add Amlodipine 5 mg once daily.
2. Type 2 Diabetes Mellitus (ICD-10: E11.9) – HbA1c poorly controlled;
   increase Metformin to 1000 mg twice daily, refer to endocrinology.
3. Hyperlipidaemia (ICD-10: E78.5) – initiate Atorvastatin 40 mg nightly.
4. Ovarian cyst follow-up – schedule pelvic ultrasound in 6 weeks.

** INCONSISTENCY PLANTED #2 **
Date below predates encounter date (for testing purposes):
Follow-up appointment: 10/11/2024

FOLLOW-UP
---------
Return in 4 weeks or sooner if BP > 180/110 or symptoms worsen.
Refer to dietitian for diabetic diet counselling.

Signed: Dr. Sarah Chen, MD
Date  : 22/11/2024
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip all LLM calls (rule-based checks only)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    out_dir = Path("output/test_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n══════════════════════════════════════════════════════════════")
    print("  Medical OCR Pipeline – Integration Test")
    print("══════════════════════════════════════════════════════════════\n")

    # ── Step 1: Mock OCR (skip actual OCR since we have raw text) ─────────────
    print("[1/5] OCR  –  using synthetic text (no file needed for this test)")
    ocr_result = OcrResult(
        full_text=SAMPLE_REPORT,
        page_map=[(1, 0, len(SAMPLE_REPORT))],
        source="synthetic",
    )
    print(f"      {len(ocr_result.full_text):,} characters\n")

    # ── Step 2: LLM extraction ────────────────────────────────────────────────
    if args.skip_llm or not api_key:
        if not api_key:
            print("[2/5] EXTRACT  –  SKIPPED (no API key)\n")
        else:
            print("[2/5] EXTRACT  –  SKIPPED (--skip-llm)\n")
        # Provide minimal mock fields so downstream steps can run
        from modules.extractor import ExtractedField
        extraction_fields = [
            ExtractedField("patient_name",    "John A. Smith",  "Patient Name   : John A. Smith"),
            ExtractedField("date_of_birth",   "15/03/1978",     "Date of Birth  : 15/03/1978"),
            ExtractedField("sex",             "Male",           "Sex            : Male"),
            ExtractedField("heart_rate",      "92",             "Heart Rate     : 92 bpm"),
            ExtractedField("spo2",            "98",             "SpO2           : 98 %"),
            ExtractedField("encounter_date",  "22/11/2024",     "Encounter Date : 22/11/2024"),
            ExtractedField("hba1c",           "8.2",            "HbA1c          : 8.2 %"),
            ExtractedField("allergy_1",       "Penicillin",     "Penicillin (rash)"),
            ExtractedField("hallucination_test", "Invented value", "THIS TEXT DOES NOT EXIST IN THE REPORT XYZ123"),
        ]
        model_name = "mock"
    else:
        print("[2/5] EXTRACT  –  Calling Claude API …")
        extraction = extractor_mod.extract(SAMPLE_REPORT, api_key=api_key)
        extraction_fields = extraction.fields
        model_name = extraction.model
        print(f"      {len(extraction_fields)} fields extracted  (model: {model_name})\n")
        for w in extraction.warnings:
            print(f"      ⚠  {w}")

    # ── Step 3: Sanity check ──────────────────────────────────────────────────
    print("[3/5] SANITY CHECK  –  Verifying quotes against source text …")
    sanity_summary = sanity_checker.run(extraction_fields, ocr_result)
    print(f"      VERIFIED={sanity_summary['VERIFIED']}  "
          f"FUZZY={sanity_summary['FUZZY']}  "
          f"NOT_FOUND={sanity_summary['NOT_FOUND']}  "
          f"SKIPPED={sanity_summary['SKIPPED']}")
    if sanity_summary["NOT_FOUND"]:
        not_found_fields = [f for f in extraction_fields if f.sanity_status == "NOT_FOUND"]
        for f in not_found_fields:
            print(f"      ✗ HALLUCINATION? field='{f.field_name}'  "
                  f"value='{f.value}'  quote={repr(f.exact_quote[:60])}")
    print()

    # ── Step 4: Inconsistency detection ──────────────────────────────────────
    print("[4/5] INCONSISTENCY DETECTION …")
    skip_llm_incons = args.skip_llm or not api_key
    inconsistencies = incons_mod.detect(
        extraction_fields,
        ocr_result.full_text,
        api_key=api_key,
        skip_llm=skip_llm_incons,
    )
    errors   = sum(1 for i in inconsistencies if i.severity == "ERROR")
    warnings = sum(1 for i in inconsistencies if i.severity == "WARNING")
    print(f"      {len(inconsistencies)} flags  (errors={errors}, warnings={warnings})")
    for inc in inconsistencies:
        icon = {"ERROR": "✗", "WARNING": "⚠", "INFO": "ℹ"}.get(inc.severity, "?")
        print(f"      {icon} [{inc.rule}] {inc.description[:80]}")
    print()

    # ── Step 5: Write outputs ─────────────────────────────────────────────────
    print("[5/5] WRITING outputs …")
    csv_path    = out_dir / "test_data.csv"
    sanity_path = out_dir / "test_sanity_report.txt"
    incons_path = out_dir / "test_inconsistencies.txt"
    json_path   = out_dir / "test_full_report.json"

    writer.write_csv(extraction_fields, csv_path)
    writer.write_sanity_report(extraction_fields, sanity_summary, sanity_path)
    writer.write_inconsistency_report(inconsistencies, incons_path)
    writer.write_json(extraction_fields, sanity_summary, inconsistencies,
                      ocr_result.source, json_path)

    print(f"\n  📄  CSV          → {csv_path}")
    print(f"  🔍  Sanity       → {sanity_path}")
    print(f"  ⚠   Inconsist.  → {incons_path}")
    print(f"  📦  Full JSON    → {json_path}")
    print("\n══════════════════════════════════════════════════════════════")
    print("  TEST COMPLETE")
    print("══════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()

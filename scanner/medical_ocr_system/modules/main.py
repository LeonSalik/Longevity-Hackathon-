#!/usr/bin/env python3
"""
main.py  –  Medical Report OCR + LLM Extraction Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage
─────
  python main.py <report_file> [options]

  python main.py patient_report.pdf
  python main.py scan.jpg --output-dir ./results --no-llm-inconsistency
  python main.py report.pdf --api-key sk-ant-...

Required environment variable (or --api-key flag):
  ANTHROPIC_API_KEY

Output files  (written to --output-dir, default: ./output/<stem>/)
────────────
  <stem>_data.csv               primary extraction CSV
  <stem>_sanity_report.txt      sanity-check results
  <stem>_inconsistencies.txt    inconsistency flags
  <stem>_full_report.json       full machine-readable output

Pipeline
────────
  1. OCR      – extract text (pdfplumber or Tesseract)
  2. Extract  – Claude API: structured fields + exact_quote per field
  3. Sanity   – verify every exact_quote against source text
  4. Incons.  – rule-based + LLM semantic checks  (parallel-ready)
  5. Write    – CSV, reports, JSON
"""

import argparse
import os
import sys
import time
import concurrent.futures
from pathlib import Path

# ── allow running without installing the package ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from modules import ocr as ocr_mod
from modules import extractor as extractor_mod
from modules import sanity_checker
from modules import inconsistency as incons_mod
from modules import writer


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Medical report OCR → LLM extraction → sanity check pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("report_file", help="PDF or image file to process")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: ./output/<stem>/)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)",
    )
    p.add_argument(
        "--no-llm-inconsistency",
        action="store_true",
        help="Skip LLM-based inconsistency check (run only rule-based)",
    )
    p.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only run OCR and save raw text; skip LLM and all checks",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print step-by-step progress",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args) -> None:
    input_path = Path(args.report_file)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stem = input_path.stem
    out_dir = Path(args.output_dir) if args.output_dir else Path("output") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    log = _make_logger(args.verbose)

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    log("OCR", f"Extracting text from {input_path.name} …")
    t0 = time.perf_counter()
    ocr_result = ocr_mod.extract(input_path)
    log("OCR", f"Done in {time.perf_counter()-t0:.1f}s  "
               f"({len(ocr_result.full_text):,} chars, "
               f"source={ocr_result.source}, "
               f"pages={len(ocr_result.page_map)})")

    for w in ocr_result.warnings:
        log("OCR", f"⚠  {w}")

    # Save raw OCR text for reference
    raw_txt_path = out_dir / f"{stem}_ocr_text.txt"
    raw_txt_path.write_text(ocr_result.full_text, encoding="utf-8")
    log("OCR", f"Raw text saved → {raw_txt_path}")

    if args.ocr_only:
        print(f"\n[DONE] OCR-only mode.  Output: {raw_txt_path}")
        return

    if not api_key:
        print(
            "[ERROR] No Anthropic API key found.\n"
            "Set ANTHROPIC_API_KEY or pass --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Step 2: LLM extraction  ───────────────────────────────────────────────
    log("EXTRACT", "Calling Claude API for structured extraction …")
    t0 = time.perf_counter()
    extraction = extractor_mod.extract(ocr_result.full_text, api_key=api_key)
    log("EXTRACT", f"Done in {time.perf_counter()-t0:.1f}s  "
                   f"({len(extraction.fields)} fields extracted, "
                   f"model={extraction.model})")

    for w in extraction.warnings:
        log("EXTRACT", f"⚠  {w}")

    if not extraction.fields:
        print("[WARNING] No fields were extracted.  Check the raw response in the JSON output.")

    # ── Steps 3 & 4 run concurrently ─────────────────────────────────────────
    log("PARALLEL", "Running sanity check + inconsistency detection concurrently …")
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        sanity_future = pool.submit(
            sanity_checker.run, extraction.fields, ocr_result
        )
        incons_future = pool.submit(
            incons_mod.detect,
            extraction.fields,
            ocr_result.full_text,
            api_key,
            args.no_llm_inconsistency,
        )
        sanity_summary  = sanity_future.result()
        inconsistencies = incons_future.result()

    log("PARALLEL", f"Done in {time.perf_counter()-t0:.1f}s")

    # ── Sanity summary ────────────────────────────────────────────────────────
    log("SANITY",
        f"VERIFIED={sanity_summary['VERIFIED']}  "
        f"FUZZY={sanity_summary['FUZZY']}  "
        f"NOT_FOUND={sanity_summary['NOT_FOUND']}  "
        f"SKIPPED={sanity_summary['SKIPPED']}")

    if sanity_summary["NOT_FOUND"] > 0:
        log("SANITY",
            f"⚠  {sanity_summary['NOT_FOUND']} field(s) could not be verified "
            f"→ possible hallucinations (see sanity report)")

    # ── Inconsistency summary ─────────────────────────────────────────────────
    errors   = sum(1 for i in inconsistencies if i.severity == "ERROR")
    warnings = sum(1 for i in inconsistencies if i.severity == "WARNING")
    log("INCONS", f"{len(inconsistencies)} flags  (errors={errors}, warnings={warnings})")

    # ── Step 5: Write outputs ─────────────────────────────────────────────────
    log("WRITE", "Writing output files …")

    csv_path     = out_dir / f"{stem}_data.csv"
    sanity_path  = out_dir / f"{stem}_sanity_report.txt"
    incons_path  = out_dir / f"{stem}_inconsistencies.txt"
    json_path    = out_dir / f"{stem}_full_report.json"

    writer.write_csv(extraction.fields, csv_path)
    writer.write_sanity_report(extraction.fields, sanity_summary, sanity_path)
    writer.write_inconsistency_report(inconsistencies, incons_path)
    writer.write_json(
        extraction.fields,
        sanity_summary,
        inconsistencies,
        ocr_result.source,
        json_path,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Input          : {input_path}")
    print(f"  OCR engine     : {ocr_result.source}")
    print(f"  Fields found   : {len(extraction.fields)}")
    print(f"  Sanity         : {sanity_summary}")
    print(f"  Inconsistencies: {len(inconsistencies)} flags")
    print()
    print(f"  📄  CSV          → {csv_path}")
    print(f"  🔍  Sanity       → {sanity_path}")
    print(f"  ⚠   Inconsist.  → {incons_path}")
    print(f"  📦  Full JSON    → {json_path}")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_logger(verbose: bool):
    def _log(tag: str, msg: str) -> None:
        if verbose:
            print(f"[{tag:8s}] {msg}")
    return _log


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)

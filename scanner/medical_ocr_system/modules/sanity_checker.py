"""
sanity_checker.py  –  Verify that every extracted field's exact_quote is
actually present in the OCR source text.

Status labels assigned to each ExtractedField:
  VERIFIED   – exact substring found in source text
  FUZZY      – near-match found above the similarity threshold (typo/OCR noise)
  NOT_FOUND  – no acceptable match → likely hallucination
  SKIPPED    – exact_quote was empty (field had no quote to verify)

char_start / char_end and page_num are populated for VERIFIED and FUZZY rows.
"""

from __future__ import annotations
import re
from typing import List, Tuple

from rapidfuzz import fuzz, process

# Minimum similarity score (0-100) to accept a fuzzy match
FUZZY_THRESHOLD = 82


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run(
    fields,             # List[ExtractedField]
    ocr_result,         # OcrResult from modules.ocr
) -> dict:
    """
    Mutate each ExtractedField in-place with sanity results.
    Return a summary dict.
    """
    summary = {"VERIFIED": 0, "FUZZY": 0, "NOT_FOUND": 0, "SKIPPED": 0}

    for f in fields:
        quote = f.exact_quote.strip()

        if not quote:
            f.sanity_status = "SKIPPED"
            summary["SKIPPED"] += 1
            continue

        status, start, end = _locate(quote, ocr_result.full_text)
        f.sanity_status = status
        f.char_start = start
        f.char_end = end
        if start >= 0:
            f.page_num = ocr_result.page_for_offset(start)
        summary[status] += 1

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Core location logic
# ──────────────────────────────────────────────────────────────────────────────

def _locate(
    quote: str, full_text: str
) -> Tuple[str, int, int]:
    """
    Return (status, char_start, char_end).
    Tries exact match first, then case-insensitive, then fuzzy sliding window.
    """
    # 1. Exact match
    idx = full_text.find(quote)
    if idx >= 0:
        return ("VERIFIED", idx, idx + len(quote))

    # 2. Case-insensitive match
    idx = full_text.lower().find(quote.lower())
    if idx >= 0:
        return ("VERIFIED", idx, idx + len(quote))

    # 3. Whitespace-normalised exact match
    norm_quote = _normalise_ws(quote)
    norm_text = _normalise_ws(full_text)
    idx = norm_text.find(norm_quote)
    if idx >= 0:
        # Map back to original text offset (best-effort)
        real_idx = _approx_original_offset(full_text, idx)
        return ("VERIFIED", real_idx, real_idx + len(quote))

    # 4. Fuzzy sliding-window search
    start, end, score = _fuzzy_search(quote, full_text)
    if score >= FUZZY_THRESHOLD:
        return ("FUZZY", start, end)

    return ("NOT_FOUND", -1, -1)


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _approx_original_offset(original: str, norm_offset: int) -> int:
    """
    Approximately map a character offset in the whitespace-normalised text
    back to the original text.  We count non-whitespace characters and
    whitespace runs as single characters in parallel.
    """
    orig_idx = 0
    norm_idx = 0
    in_ws = False
    for orig_idx, ch in enumerate(original):
        if norm_idx >= norm_offset:
            return orig_idx
        if ch.isspace():
            if not in_ws:
                norm_idx += 1  # whole run counts as 1 space
                in_ws = True
        else:
            norm_idx += 1
            in_ws = False
    return orig_idx


def _fuzzy_search(
    quote: str, text: str, window_multiplier: float = 1.5
) -> Tuple[int, int, float]:
    """
    Slide a window of size ~len(quote)*window_multiplier over text and score
    each candidate with partial_ratio.  Returns best (start, end, score).
    """
    qlen = len(quote)
    if qlen == 0:
        return (0, 0, 0.0)

    window = int(qlen * window_multiplier)
    best_score = 0.0
    best_start = 0
    best_end = qlen

    step = max(1, qlen // 4)   # stride to keep it fast
    for i in range(0, max(1, len(text) - window + 1), step):
        candidate = text[i : i + window]
        score = fuzz.partial_ratio(quote.lower(), candidate.lower())
        if score > best_score:
            best_score = score
            best_start = i
            best_end = i + qlen   # approximate end

    return (best_start, best_end, best_score)

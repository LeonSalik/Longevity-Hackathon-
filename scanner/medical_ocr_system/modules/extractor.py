"""
extractor.py – Βελτιωμένη έκδοση για Gemini
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import List
import time 

@dataclass
class ExtractedField:
    field_name: str
    value: str
    exact_quote: str          
    unit: str = ""                
    reference_range: str = ""     
    context_snippet: str = ""     
    confidence: str = "high"  
    notes: str = ""
    char_start: int = -1
    char_end: int = -1
    page_num: int = -1
    sanity_status: str = "PENDING"

@dataclass
class ExtractionResult:
    fields: List[ExtractedField] = field(default_factory=list)
    raw_response: str = ""
    model: str = ""
    warnings: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# ΠΟΛΥ ΠΙΟ ΙΣΧΥΡΟ SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert medical data extractor. Your ONLY job is to return a valid JSON array.

STRICT RULES:
- Return **ONLY** the JSON array. No explanations, no markdown, no ```json, no extra text.
- For every field you find in the document, create an object.
- Use EXACT text from the document for `exact_quote`.
- If a mandatory field is not found, still include it with value=null and confidence="low".

MANDATORY FIELDS TO LOOK FOR:
- subject_id, hadm_id, gender, age, date_of_birth
- icd9_code, short_title, long_title
- ast, alt, bilirubin, albumin, platelets, inr, creatinine, glucose, fibroscan, liver_stiffness

Extract ALL lab values, diagnoses, dates, patient info you can find.

Return format example:
[
  {
    "field_name": "gender",
    "value": "male",
    "unit": null,
    "reference_range": null,
    "exact_quote": "43-year-old male",
    "context_snippet": "Patient is a 43-year-old male with...",
    "confidence": "high",
    "notes": ""
  }
]"""

# ──────────────────────────────────────────────────────────────────────────────
def extract(ocr_text: str, api_key: str | None = None) -> ExtractionResult:
    max_retries = 3
    retry_delay = 3
    
    from openai import OpenAI

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No Google API key found.")

    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Καθαρίζουμε λίγο το OCR text πριν το στείλουμε
    cleaned_ocr = re.sub(r'[^a-zA-Z0-9\s\.\,\-\(\)\[\]\:\/]', ' ', ocr_text)  # αφαιρούμε σπασμένα σύμβολα
    cleaned_ocr = re.sub(r'\s+', ' ', cleaned_ocr)[:15000]  # περιορίζουμε μέγεθος

    user_msg = f"Extract structured medical data from this report:\n\n{cleaned_ocr}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                max_tokens=4096,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1}...")
                time.sleep(retry_delay)
            else:
                raise

    raw = response.choices[0].message.content.strip()
    result = ExtractionResult(raw_response=raw, model=response.model)

    print(f"DEBUG Raw first 400 chars:\n{raw[:400]}")
    print(f"DEBUG Raw last 200 chars:\n{raw[-200:]}")

    # Advanced JSON cleaning
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    # Βρίσκουμε το πρώτο JSON array
    match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        with open("llm_raw_response.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("💾 Saved raw response to llm_raw_response.txt")
    except:
        pass

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        result.warnings.append(f"JSON error: {exc}")
        print(f"❌ JSON Parse Failed: {exc}")
        return result

    if not isinstance(items, list):
        items = [items] if isinstance(items, dict) else []

    for item in items:
        if isinstance(item, dict):
            result.fields.append(
                ExtractedField(
                    field_name=str(item.get("field_name", "")),
                    value=str(item.get("value", "")) if item.get("value") is not None else "",
                    exact_quote=str(item.get("exact_quote", "")),
                    unit=str(item.get("unit", "")),
                    reference_range=str(item.get("reference_range", "")),
                    context_snippet=str(item.get("context_snippet", "")),
                    confidence=str(item.get("confidence", "medium")),
                    notes=str(item.get("notes", "")),
                )
            )

    print(f"✅ Extracted {len(result.fields)} fields")
    return result
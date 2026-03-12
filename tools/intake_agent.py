"""
intake_agent.py  -  Document Intake Agent for HOMEBASE

Accepts a warranty document, contractor invoice, work receipt, or inspection
report (PDF or image) and extracts structured data using Gemini 1.5 Flash
(multimodal). Maps the document to the closest matching registry item and
proposes field updates.

Extracted fields are surfaced for HITL review before any registry write occurs.
The caller is responsible for applying approved updates via registry_tools.update_item().

Returns:
    document_type:    warranty | invoice | receipt | inspection | unknown
    extracted_fields: dict of raw extracted data (date, contractor, cost, scope, etc.)
    proposed_item_id: registry item ID this document most likely relates to
    proposed_updates: dict of registry field changes to apply (description, status, etc.)
    confidence:       float 0.0-1.0 (match confidence)
    rationale:        one-sentence explanation of the match
    error:            str or None

Provider: Gemini 1.5 Flash (google-genai SDK direct -- bypasses langchain v1beta wrapper)
"""

import base64
import json
import os
import re
from typing import TypedDict

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class IntakeResult(TypedDict):
    document_type:    str
    extracted_fields: dict
    proposed_item_id: str
    proposed_updates: dict
    confidence:       float
    rationale:        str
    error:            str | None


_VALID_DOC_TYPES = {"warranty", "invoice", "receipt", "inspection", "unknown"}

_SYSTEM_PROMPT = (
    "You are a home management document intake assistant. "
    "You will receive an image or PDF of a home-related document "
    "(warranty, contractor invoice, work receipt, or inspection report) "
    "along with a list of open registry items.\n\n"
    "Your job is to:\n"
    "1. Identify the document type (warranty, invoice, receipt, inspection, or unknown)\n"
    "2. Extract all relevant structured fields from the document\n"
    "3. Identify which registry item this document most likely relates to\n"
    "4. Propose specific field updates for that registry item\n\n"
    "Respond ONLY with valid JSON -- no preamble, no markdown fences. "
    "Use exactly these keys:\n"
    '{\n'
    '  "document_type": "<warranty|invoice|receipt|inspection|unknown>",\n'
    '  "extracted_fields": {\n'
    '    "date": "<date of service or purchase>",\n'
    '    "contractor": "<contractor or vendor name>",\n'
    '    "cost": "<dollar amount>",\n'
    '    "scope": "<work performed or covered>",\n'
    '    "item_reference": "<any item ID or model number mentioned>",\n'
    '    "notes": "<any other relevant details>"\n'
    '  },\n'
    '  "proposed_item_id": "<registry item ID e.g. HV-001 or empty string>",\n'
    '  "proposed_updates": {\n'
    '    "description": "<updated description>",\n'
    '    "status": "<open|in_progress|closed>"\n'
    '  },\n'
    '  "confidence": <0.0 to 1.0>,\n'
    '  "rationale": "<one sentence match explanation>"\n'
    '}\n\n'
    "Matching guidelines:\n"
    "- HVAC docs match HV-* items; plumbing matches PLB-*; electrical matches EL-*; "
    "appliance matches APP-*; general matches GEN-*\n"
    "- Match by specific description when possible\n"
    "- If document explicitly references a registry ID, use it\n"
    "- No confident match: set proposed_item_id to empty string and confidence below 0.5\n"
    "- Completed work invoices/receipts: propose status closed\n"
    "- Warranties: propose status in_progress\n\n"
    "Confidence: 0.85-1.0 explicit match; 0.65-0.84 strong match; "
    "0.40-0.64 ambiguous; below 0.40 no match"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    return genai.Client(api_key=key)


def _encode_file(file_bytes: bytes, mime_type: str) -> dict:
    """Return a dict representing an inline_data blob for display/test purposes."""
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    return {"mime_type": mime_type, "data": b64}


def _format_registry_list(registry: list[dict]) -> str:
    """Format registry items as a compact list for the LLM context."""
    lines = ["Open registry items:"]
    for item in registry:
        lines.append(
            f"  {item['id']} [{item['category']}] {item['title']}"
            f" -- status: {item.get('status', 'open')}"
        )
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    mime_type: str,
    registry: list[dict],
    api_key: str | None = None,
) -> IntakeResult:
    """
    Process an uploaded document and propose registry item updates.

    Args:
        file_bytes: Raw bytes of the uploaded PDF or image file.
        mime_type:  MIME type (e.g. 'application/pdf', 'image/jpeg', 'image/png').
        registry:   Current list of registry items (from get_registry()).
        api_key:    Google API key. Falls back to GOOGLE_API_KEY env var.

    Returns:
        IntakeResult dict. Never raises -- errors captured in 'error' field.
    """
    if not file_bytes:
        return IntakeResult(
            document_type="unknown",
            extracted_fields={},
            proposed_item_id="",
            proposed_updates={},
            confidence=0.0,
            rationale="",
            error="No file content provided.",
        )

    if not registry:
        return IntakeResult(
            document_type="unknown",
            extracted_fields={},
            proposed_item_id="",
            proposed_updates={},
            confidence=0.0,
            rationale="",
            error="Registry is empty -- cannot match document to any item.",
        )

    try:
        client = _get_client(api_key=api_key)
        registry_context = _format_registry_list(registry)

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Content(parts=[
                    types.Part(text=_SYSTEM_PROMPT),
                    types.Part(text=registry_context),
                    types.Part(text="Document to analyze:"),
                    types.Part(inline_data=types.Blob(
                        mime_type=mime_type,
                        data=base64.standard_b64encode(file_bytes).decode("utf-8"),
                    )),
                ])
            ],
        )

        parsed = _parse_response(response.text)

        # Normalize document_type
        doc_type = str(parsed.get("document_type", "unknown")).lower().strip()
        if doc_type not in _VALID_DOC_TYPES:
            doc_type = "unknown"

        # Normalize confidence
        try:
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        extracted_fields = parsed.get("extracted_fields", {})
        if not isinstance(extracted_fields, dict):
            extracted_fields = {}

        proposed_updates = parsed.get("proposed_updates", {})
        if not isinstance(proposed_updates, dict):
            proposed_updates = {}

        # Only allow safe fields to be updated
        allowed_update_fields = {"title", "description", "status"}
        proposed_updates = {
            k: v for k, v in proposed_updates.items()
            if k in allowed_update_fields
        }

        proposed_item_id = str(parsed.get("proposed_item_id", "")).strip()

        # Validate proposed_item_id exists in registry
        valid_ids = {item["id"] for item in registry}
        if proposed_item_id and proposed_item_id not in valid_ids:
            proposed_item_id = ""
            confidence = min(confidence, 0.4)

        return IntakeResult(
            document_type=doc_type,
            extracted_fields=extracted_fields,
            proposed_item_id=proposed_item_id,
            proposed_updates=proposed_updates,
            confidence=confidence,
            rationale=str(parsed.get("rationale", "")).strip(),
            error=None,
        )

    except Exception as e:
        return IntakeResult(
            document_type="unknown",
            extracted_fields={},
            proposed_item_id="",
            proposed_updates={},
            confidence=0.0,
            rationale="",
            error=f"{type(e).__name__}: {e}",
        )
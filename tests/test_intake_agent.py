"""
tests/test_intake_agent.py  --  Unit tests for Document Intake Agent.

Covers:
  - Empty file guard
  - Empty registry guard
  - Happy paths: all four document types (warranty, invoice, receipt, inspection)
  - Confidence clamped to [0.0, 1.0]
  - Invalid document_type normalized to 'unknown'
  - proposed_updates restricted to allowed fields only
  - proposed_updates non-dict coerced to empty dict
  - extracted_fields non-dict coerced to empty dict
  - proposed_item_id validated against registry -- invalid ID cleared
  - confidence reduced when item_id invalid
  - Markdown fence stripping
  - Malformed / non-JSON LLM response handled gracefully
  - LLM exception handled gracefully (never raises)
  - error field includes exception type
  - API key passed to genai.Client
  - None API key falls back to env var
  - _format_registry_list output contains item IDs
  - _encode_file produces correct structure
  - _VALID_DOC_TYPES constant is complete
  - IntakeResult TypedDict structure
"""

import base64
import json
import pytest
from unittest.mock import MagicMock, patch

import tools.intake_agent as ia_module
from tools.intake_agent import (
    process_document,
    _format_registry_list,
    _encode_file,
    _VALID_DOC_TYPES,
    IntakeResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REGISTRY = [
    {"id": "HV-001", "category": "hvac",      "title": "Replace HVAC air filter",        "status": "open"},
    {"id": "APP-001","category": "appliance",  "title": "Dishwasher not draining fully",  "status": "open"},
    {"id": "PLB-002","category": "plumbing",   "title": "Inspect water heater anode rod", "status": "open"},
    {"id": "EL-001", "category": "electrical", "title": "Replace tripping GFCI outlet",   "status": "open"},
    {"id": "GEN-004","category": "general",    "title": "Clean gutters",                  "status": "open"},
]

SAMPLE_FILE_BYTES = b"%PDF-1.4 fake pdf content for testing purposes only"
SAMPLE_MIME = "application/pdf"


def make_llm_payload(
    doc_type="invoice",
    item_id="APP-001",
    confidence=0.88,
    rationale="Invoice matches dishwasher drain repair.",
    extracted_fields=None,
    proposed_updates=None,
) -> str:
    if extracted_fields is None:
        extracted_fields = {
            "date": "2025-11-15",
            "contractor": "AquaFix Plumbing",
            "cost": "$220",
            "scope": "Replaced drain pump assembly",
            "item_reference": "APP-001",
            "notes": "90-day labor warranty included",
        }
    if proposed_updates is None:
        proposed_updates = {
            "description": "Drain pump replaced by AquaFix Plumbing on 2025-11-15.",
            "status": "closed",
        }
    return json.dumps({
        "document_type": doc_type,
        "extracted_fields": extracted_fields,
        "proposed_item_id": item_id,
        "proposed_updates": proposed_updates,
        "confidence": confidence,
        "rationale": rationale,
    })


def mock_genai_client(response_text: str):
    """Patch genai.Client so generate_content returns response_text."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_model

    return patch("tools.intake_agent.genai.Client", return_value=mock_client)


def mock_genai_client_raises(exc: Exception):
    """Patch genai.Client so generate_content raises exc."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = exc

    mock_client = MagicMock()
    mock_client.models = mock_model

    return patch("tools.intake_agent.genai.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Constants and structure
# ---------------------------------------------------------------------------

class TestConstants:

    def test_valid_doc_types_complete(self):
        assert _VALID_DOC_TYPES == {"warranty", "invoice", "receipt", "inspection", "unknown"}

    def test_valid_doc_types_count(self):
        assert len(_VALID_DOC_TYPES) == 5

    def test_typedict_keys(self):
        keys = set(IntakeResult.__annotations__.keys())
        assert keys == {
            "document_type", "extracted_fields", "proposed_item_id",
            "proposed_updates", "confidence", "rationale", "error",
        }


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------

class TestInputGuards:

    def test_empty_file_bytes_returns_error(self):
        result = process_document(b"", SAMPLE_MIME, SAMPLE_REGISTRY)
        assert result["error"] is not None
        assert result["document_type"] == "unknown"
        assert result["confidence"] == 0.0

    def test_empty_registry_returns_error(self):
        result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, [])
        assert result["error"] is not None
        assert result["document_type"] == "unknown"

    def test_empty_file_returns_empty_proposed_item(self):
        result = process_document(b"", SAMPLE_MIME, SAMPLE_REGISTRY)
        assert result["proposed_item_id"] == ""

    def test_empty_file_returns_empty_proposed_updates(self):
        result = process_document(b"", SAMPLE_MIME, SAMPLE_REGISTRY)
        assert result["proposed_updates"] == {}

    def test_empty_file_returns_empty_extracted_fields(self):
        result = process_document(b"", SAMPLE_MIME, SAMPLE_REGISTRY)
        assert result["extracted_fields"] == {}


# ---------------------------------------------------------------------------
# Happy paths -- all document types
# ---------------------------------------------------------------------------

class TestHappyPaths:

    def test_invoice_document_type(self):
        with mock_genai_client(make_llm_payload(doc_type="invoice")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "invoice"
        assert result["error"] is None

    def test_warranty_document_type(self):
        with mock_genai_client(make_llm_payload(doc_type="warranty", item_id="HV-001", confidence=0.75,
                                                proposed_updates={"description": "HVAC warranty.", "status": "in_progress"})):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "warranty"
        assert result["error"] is None

    def test_receipt_document_type(self):
        with mock_genai_client(make_llm_payload(doc_type="receipt", item_id="GEN-004", confidence=0.70,
                                                proposed_updates={"description": "Gutters cleaned.", "status": "closed"})):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "receipt"
        assert result["error"] is None

    def test_inspection_document_type(self):
        with mock_genai_client(make_llm_payload(doc_type="inspection", item_id="EL-001", confidence=0.82,
                                                proposed_updates={"description": "GFCI inspected.", "status": "in_progress"})):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "inspection"
        assert result["error"] is None

    def test_proposed_item_id_returned(self):
        with mock_genai_client(make_llm_payload(item_id="APP-001")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_item_id"] == "APP-001"

    def test_confidence_returned(self):
        with mock_genai_client(make_llm_payload(confidence=0.88)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == pytest.approx(0.88)

    def test_rationale_returned(self):
        with mock_genai_client(make_llm_payload(rationale="Matches dishwasher drain issue.")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["rationale"] == "Matches dishwasher drain issue."

    def test_extracted_fields_returned(self):
        with mock_genai_client(make_llm_payload()):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["extracted_fields"]["contractor"] == "AquaFix Plumbing"

    def test_proposed_updates_returned(self):
        with mock_genai_client(make_llm_payload()):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_updates"]["status"] == "closed"

    def test_error_is_none_on_success(self):
        with mock_genai_client(make_llm_payload()):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------

class TestConfidenceNormalization:

    def test_confidence_above_one_clamped(self):
        with mock_genai_client(make_llm_payload(confidence=1.5)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == 1.0

    def test_confidence_below_zero_clamped(self):
        with mock_genai_client(make_llm_payload(confidence=-0.2)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == 0.0

    def test_confidence_string_coerced(self):
        payload = json.loads(make_llm_payload())
        payload["confidence"] = "0.77"
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == pytest.approx(0.77)

    def test_confidence_missing_key_defaults_to_half(self):
        payload = json.loads(make_llm_payload())
        del payload["confidence"]
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == pytest.approx(0.5)

    def test_confidence_null_defaults_to_half(self):
        payload = json.loads(make_llm_payload())
        payload["confidence"] = None
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Document type normalization
# ---------------------------------------------------------------------------

class TestDocumentTypeNormalization:

    def test_invalid_doc_type_normalized_to_unknown(self):
        with mock_genai_client(make_llm_payload(doc_type="purchase_order")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "unknown"

    def test_uppercase_doc_type_normalized(self):
        with mock_genai_client(make_llm_payload(doc_type="INVOICE")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "invoice"

    def test_empty_doc_type_normalized_to_unknown(self):
        with mock_genai_client(make_llm_payload(doc_type="")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "unknown"

    def test_missing_doc_type_key_normalized_to_unknown(self):
        payload = json.loads(make_llm_payload())
        del payload["document_type"]
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "unknown"


# ---------------------------------------------------------------------------
# Field sanitization
# ---------------------------------------------------------------------------

class TestFieldSanitization:

    def test_proposed_updates_restricted_to_allowed_fields(self):
        payload = json.loads(make_llm_payload())
        payload["proposed_updates"]["urgency"] = 0.9
        payload["proposed_updates"]["impact"] = 0.8
        payload["proposed_updates"]["id"] = "APP-999"
        payload["proposed_updates"]["category"] = "hvac"
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert "urgency" not in result["proposed_updates"]
        assert "impact" not in result["proposed_updates"]
        assert "id" not in result["proposed_updates"]
        assert "category" not in result["proposed_updates"]
        assert result["proposed_updates"]["status"] == "closed"

    def test_extracted_fields_non_dict_coerced_to_empty(self):
        payload = json.loads(make_llm_payload())
        payload["extracted_fields"] = "not a dict"
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["extracted_fields"] == {}

    def test_proposed_updates_non_dict_coerced_to_empty(self):
        payload = json.loads(make_llm_payload())
        payload["proposed_updates"] = ["status", "closed"]
        with mock_genai_client(json.dumps(payload)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_updates"] == {}


# ---------------------------------------------------------------------------
# Item ID validation
# ---------------------------------------------------------------------------

class TestItemIDValidation:

    def test_invalid_item_id_cleared(self):
        with mock_genai_client(make_llm_payload(item_id="APP-999", confidence=0.85)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_item_id"] == ""

    def test_invalid_item_id_reduces_confidence(self):
        with mock_genai_client(make_llm_payload(item_id="ZZ-999", confidence=0.90)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["confidence"] <= 0.4

    def test_valid_item_id_preserved(self):
        with mock_genai_client(make_llm_payload(item_id="PLB-002", confidence=0.80)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_item_id"] == "PLB-002"

    def test_empty_item_id_preserved_as_empty(self):
        with mock_genai_client(make_llm_payload(item_id="", confidence=0.35)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_item_id"] == ""

    def test_whitespace_item_id_cleared(self):
        with mock_genai_client(make_llm_payload(item_id="  ", confidence=0.5)):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["proposed_item_id"] == ""


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:

    def test_json_fence_stripped(self):
        payload = make_llm_payload(doc_type="invoice", item_id="APP-001")
        with mock_genai_client(f"```json\n{payload}\n```"):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "invoice"
        assert result["error"] is None

    def test_plain_fence_stripped(self):
        payload = make_llm_payload(doc_type="warranty", item_id="HV-001",
                                   proposed_updates={"status": "in_progress"})
        with mock_genai_client(f"```\n{payload}\n```"):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["document_type"] == "warranty"
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_malformed_json_returns_error(self):
        with mock_genai_client("not json at all here"):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["error"] is not None
        assert result["document_type"] == "unknown"

    def test_partial_json_returns_error(self):
        with mock_genai_client('{"document_type": "invoice"'):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["error"] is not None

    def test_llm_exception_returns_error(self):
        with mock_genai_client_raises(Exception("Network timeout")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["error"] is not None
        assert "Network timeout" in result["error"]

    def test_exception_never_raises(self):
        with mock_genai_client_raises(RuntimeError("Catastrophic failure")):
            try:
                result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
            except Exception as e:
                pytest.fail(f"process_document raised: {e}")
        assert result["error"] is not None

    def test_error_includes_exception_type(self):
        with mock_genai_client_raises(ValueError("Bad response")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert "ValueError" in result["error"]

    def test_error_returns_empty_collections(self):
        with mock_genai_client_raises(Exception("Fail")):
            result = process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="fake")
        assert result["extracted_fields"] == {}
        assert result["proposed_updates"] == {}
        assert result["proposed_item_id"] == ""
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

class TestAPIKeyHandling:

    def test_api_key_passed_to_client(self):
        with patch("tools.intake_agent.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text=make_llm_payload())
            process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key="test_google_key")
        call_kwargs = MockClient.call_args
        assert call_kwargs.kwargs.get("api_key") == "test_google_key"

    def test_none_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env_google_key")
        with patch("tools.intake_agent.genai.Client") as MockClient:
            instance = MockClient.return_value
            instance.models.generate_content.return_value = MagicMock(text=make_llm_payload())
            process_document(SAMPLE_FILE_BYTES, SAMPLE_MIME, SAMPLE_REGISTRY, api_key=None)
        call_kwargs = MockClient.call_args
        assert call_kwargs.kwargs.get("api_key") == "env_google_key"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_format_registry_list_contains_all_ids(self):
        output = _format_registry_list(SAMPLE_REGISTRY)
        for item in SAMPLE_REGISTRY:
            assert item["id"] in output

    def test_format_registry_list_contains_titles(self):
        output = _format_registry_list(SAMPLE_REGISTRY)
        assert "Dishwasher not draining fully" in output

    def test_format_registry_list_contains_categories(self):
        output = _format_registry_list(SAMPLE_REGISTRY)
        assert "appliance" in output
        assert "hvac" in output

    def test_format_registry_list_empty_registry(self):
        output = _format_registry_list([])
        assert "Open registry items" in output

    def test_encode_file_has_mime_type(self):
        result = _encode_file(b"test content", "image/jpeg")
        assert result["mime_type"] == "image/jpeg"

    def test_encode_file_has_data(self):
        result = _encode_file(b"test content", "application/pdf")
        assert "data" in result

    def test_encode_file_base64_decodable(self):
        content = b"test content bytes here"
        result = _encode_file(content, "application/pdf")
        decoded = base64.standard_b64decode(result["data"])
        assert decoded == content

    def test_encode_file_mime_type_pdf(self):
        result = _encode_file(b"data", "application/pdf")
        assert result["mime_type"] == "application/pdf"
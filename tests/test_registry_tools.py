"""
test_registry_tools.py  -  Unit tests for registry_tools.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.registry_tools import (
    get_registry,
    classify_item,
    classify_registry,
    get_item_detail,
    URGENCY_THRESHOLD,
    IMPACT_THRESHOLD,
    STALE_DAYS_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_items():
    return [
        {"id": "T-001", "category": "hvac",     "title": "HU/HI item",  "urgency": 0.8, "impact": 0.8, "days_since_update": 20, "status": "open"},
        {"id": "T-002", "category": "plumbing",  "title": "HU/LI item",  "urgency": 0.8, "impact": 0.3, "days_since_update": 5,  "status": "open"},
        {"id": "T-003", "category": "electrical","title": "LU/HI item",  "urgency": 0.2, "impact": 0.9, "days_since_update": 3,  "status": "open"},
        {"id": "T-004", "category": "general",   "title": "LU/LI item",  "urgency": 0.2, "impact": 0.2, "days_since_update": 2,  "status": "open"},
        {"id": "T-005", "category": "appliance", "title": "Stale HU/HI", "urgency": 0.7, "impact": 0.7, "days_since_update": 30, "status": "open"},
    ]


@pytest.fixture
def boundary_items():
    """Items sitting exactly on the threshold boundaries."""
    return [
        {"id": "B-001", "category": "general", "title": "Exact threshold both", "urgency": 0.6,  "impact": 0.6,  "days_since_update": 14, "status": "open"},
        {"id": "B-002", "category": "general", "title": "Just below both",       "urgency": 0.59, "impact": 0.59, "days_since_update": 13, "status": "open"},
        {"id": "B-003", "category": "general", "title": "Exact urgency only",    "urgency": 0.6,  "impact": 0.59, "days_since_update": 1,  "status": "open"},
        {"id": "B-004", "category": "general", "title": "Exact impact only",     "urgency": 0.59, "impact": 0.6,  "days_since_update": 1,  "status": "open"},
    ]


# ---------------------------------------------------------------------------
# get_registry
# ---------------------------------------------------------------------------

class TestGetRegistry:
    def test_returns_list(self):
        result = get_registry()
        assert isinstance(result, list)

    def test_not_empty(self):
        result = get_registry()
        assert len(result) > 0

    def test_required_fields_present(self):
        result = get_registry()
        required = {"id", "category", "title", "description", "urgency", "impact", "days_since_update", "status"}
        for item in result:
            assert required.issubset(item.keys()), f"Item {item.get('id')} missing fields"

    def test_urgency_impact_in_range(self):
        result = get_registry()
        for item in result:
            assert 0.0 <= item["urgency"] <= 1.0, f"{item['id']} urgency out of range"
            assert 0.0 <= item["impact"] <= 1.0, f"{item['id']} impact out of range"

    def test_ids_are_unique(self):
        result = get_registry()
        ids = [item["id"] for item in result]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in registry"


# ---------------------------------------------------------------------------
# classify_item
# ---------------------------------------------------------------------------

class TestClassifyItem:
    def test_hu_hi_classification(self):
        item = {"id": "X", "urgency": 0.8, "impact": 0.8, "days_since_update": 1, "status": "open"}
        result = classify_item(item)
        assert result["quadrant"] == "HU/HI"

    def test_hu_li_classification(self):
        item = {"id": "X", "urgency": 0.8, "impact": 0.3, "days_since_update": 1, "status": "open"}
        result = classify_item(item)
        assert result["quadrant"] == "HU/LI"

    def test_lu_hi_classification(self):
        item = {"id": "X", "urgency": 0.2, "impact": 0.9, "days_since_update": 1, "status": "open"}
        result = classify_item(item)
        assert result["quadrant"] == "LU/HI"

    def test_lu_li_classification(self):
        item = {"id": "X", "urgency": 0.2, "impact": 0.2, "days_since_update": 1, "status": "open"}
        result = classify_item(item)
        assert result["quadrant"] == "LU/LI"

    def test_original_fields_preserved(self):
        item = {"id": "X", "urgency": 0.8, "impact": 0.8, "days_since_update": 5, "status": "open", "title": "Test"}
        result = classify_item(item)
        assert result["id"] == "X"
        assert result["title"] == "Test"
        assert result["urgency"] == 0.8

    def test_quadrant_key_added(self):
        item = {"id": "X", "urgency": 0.5, "impact": 0.5, "days_since_update": 1, "status": "open"}
        result = classify_item(item)
        assert "quadrant" in result

    def test_boundary_both_at_threshold_is_hu_hi(self, boundary_items):
        result = classify_item(boundary_items[0])  # urgency=0.6, impact=0.6
        assert result["quadrant"] == "HU/HI"

    def test_boundary_just_below_both_is_lu_li(self, boundary_items):
        result = classify_item(boundary_items[1])  # urgency=0.59, impact=0.59
        assert result["quadrant"] == "LU/LI"

    def test_boundary_exact_urgency_only_is_hu_li(self, boundary_items):
        result = classify_item(boundary_items[2])  # urgency=0.6, impact=0.59
        assert result["quadrant"] == "HU/LI"

    def test_boundary_exact_impact_only_is_lu_hi(self, boundary_items):
        result = classify_item(boundary_items[3])  # urgency=0.59, impact=0.6
        assert result["quadrant"] == "LU/HI"


# ---------------------------------------------------------------------------
# classify_registry
# ---------------------------------------------------------------------------

class TestClassifyRegistry:
    def test_returns_expected_keys(self, sample_items):
        result = classify_registry(sample_items)
        assert {"hu_hi", "hu_li", "lu_hi", "lu_li", "stale_items", "all"}.issubset(result.keys())

    def test_all_items_classified(self, sample_items):
        result = classify_registry(sample_items)
        assert len(result["all"]) == len(sample_items)

    def test_all_items_have_quadrant(self, sample_items):
        result = classify_registry(sample_items)
        for item in result["all"]:
            assert "quadrant" in item

    def test_bucket_counts_correct(self, sample_items):
        result = classify_registry(sample_items)
        # T-001 HU/HI, T-002 HU/LI, T-003 LU/HI, T-004 LU/LI, T-005 HU/HI
        assert len(result["hu_hi"]) == 2
        assert len(result["hu_li"]) == 1
        assert len(result["lu_hi"]) == 1
        assert len(result["lu_li"]) == 1

    def test_stale_items_detected(self, sample_items):
        result = classify_registry(sample_items)
        # T-001 (20 days) and T-005 (30 days) are stale
        stale_ids = {item["id"] for item in result["stale_items"]}
        assert "T-001" in stale_ids
        assert "T-005" in stale_ids

    def test_non_stale_items_not_flagged(self, sample_items):
        result = classify_registry(sample_items)
        stale_ids = {item["id"] for item in result["stale_items"]}
        assert "T-002" not in stale_ids  # 5 days
        assert "T-003" not in stale_ids  # 3 days
        assert "T-004" not in stale_ids  # 2 days

    def test_stale_boundary_exactly_14_days(self):
        item = {"id": "S-001", "category": "general", "title": "Exactly stale", "urgency": 0.8, "impact": 0.8, "days_since_update": 14, "status": "open"}
        result = classify_registry([item])
        assert len(result["stale_items"]) == 1

    def test_stale_boundary_13_days_not_stale(self):
        item = {"id": "S-002", "category": "general", "title": "Just under stale", "urgency": 0.8, "impact": 0.8, "days_since_update": 13, "status": "open"}
        result = classify_registry([item])
        assert len(result["stale_items"]) == 0

    def test_empty_registry(self):
        result = classify_registry([])
        assert result["all"] == []
        assert result["hu_hi"] == []
        assert result["stale_items"] == []

    def test_buckets_sum_to_total(self, sample_items):
        result = classify_registry(sample_items)
        bucket_total = (
            len(result["hu_hi"]) +
            len(result["hu_li"]) +
            len(result["lu_hi"]) +
            len(result["lu_li"])
        )
        assert bucket_total == len(sample_items)


# ---------------------------------------------------------------------------
# get_item_detail
# ---------------------------------------------------------------------------

class TestGetItemDetail:
    def test_returns_correct_item(self, sample_items):
        result = get_item_detail("T-003", sample_items)
        assert result is not None
        assert result["id"] == "T-003"
        assert result["title"] == "LU/HI item"

    def test_returns_none_for_missing_id(self, sample_items):
        result = get_item_detail("DOES-NOT-EXIST", sample_items)
        assert result is None

    def test_returns_first_match(self, sample_items):
        # Registry shouldn't have dupes, but tool should return first match
        result = get_item_detail("T-001", sample_items)
        assert result["id"] == "T-001"

    def test_works_against_live_registry(self):
        registry = get_registry()
        first_id = registry[0]["id"]
        result = get_item_detail(first_id, registry)
        assert result is not None
        assert result["id"] == first_id
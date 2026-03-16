"""
test_duplicate_detector.py  -  Tests for TF-IDF duplicate detection.

Coverage:
- check_duplicates() threshold behavior (above, at, below)
- Empty registry returns no matches
- Exact title match returns high score
- Unrelated item returns no match
- Multi-match ranking (highest score first)
- Dual-channel scoring (full text + title-only)
- Status filter - closed items excluded by default
- has_duplicates() convenience function
- top_match() convenience function
- _build_corpus_text() helper
- Empty candidate text returns no matches
- Custom threshold respected
"""

import pytest

from tools.duplicate_detector import (
    check_duplicates,
    has_duplicates,
    top_match,
    _build_corpus_text,
    DEFAULT_THRESHOLD,
    DuplicateMatch,
)


def make_item(id, title, description="", category="general", status="open", urgency=0.5, impact=0.5):
    return {
        "id": id, "title": title, "description": description,
        "category": category, "status": status, "urgency": urgency,
        "impact": impact, "days_since_update": 5,
    }


HVAC_REGISTRY = [
    make_item("HV-001", "Replace HVAC air filter", "Filter is clogged and overdue for replacement", "hvac"),
    make_item("HV-002", "Annual AC tune-up", "Schedule seasonal AC maintenance before summer", "hvac"),
    make_item("PLB-001", "Leaking exterior hose bib", "Hose bib drips when not in use", "plumbing"),
    make_item("EL-001", "GFCI outlet tripping in garage", "Outlet trips intermittently under load", "electrical"),
    make_item("GEN-001", "Broken fence gate latch", "Gate latch broken, security concern", "general"),
]

MIXED_STATUS_REGISTRY = [
    make_item("HV-001", "Replace HVAC air filter", "Filter overdue", "hvac", status="open"),
    make_item("HV-002", "Old AC filter replaced", "Completed last month", "hvac", status="closed"),
    make_item("HV-003", "AC filter in progress", "Currently replacing", "hvac", status="in_progress"),
]


class TestBuildCorpusText:
    def test_combines_title_and_description(self):
        item = make_item("X-001", "Replace filter", "Air filter is clogged")
        result = _build_corpus_text(item)
        assert "Replace filter" in result
        assert "Air filter is clogged" in result

    def test_handles_missing_description(self):
        item = make_item("X-001", "Replace filter")
        item.pop("description", None)
        assert "Replace filter" in _build_corpus_text(item)

    def test_handles_none_description(self):
        item = make_item("X-001", "Replace filter", description=None)
        assert "Replace filter" in _build_corpus_text(item)

    def test_handles_empty_title_and_description(self):
        assert _build_corpus_text(make_item("X-001", "", "")) == ""


class TestCheckDuplicates:
    def test_empty_registry_returns_no_matches(self):
        assert check_duplicates("Replace HVAC filter", "Filter is clogged", registry=[]) == []

    def test_exact_title_match_returns_high_score(self):
        result = check_duplicates(
            "Replace HVAC air filter", "Filter is clogged and overdue for replacement",
            registry=HVAC_REGISTRY,
        )
        assert len(result) >= 1
        assert result[0]["item_id"] == "HV-001"
        assert result[0]["score"] >= 0.55

    def test_paraphrased_title_caught_by_dual_channel(self):
        result = check_duplicates(
            "HVAC Air Filter Replacement",
            "Air filter hasn't been changed in 6 weeks, system running warm",
            registry=HVAC_REGISTRY,
        )
        assert len(result) >= 1
        assert result[0]["item_id"] == "HV-001"

    def test_unrelated_item_returns_no_match(self):
        result = check_duplicates(
            "Repaint the kitchen ceiling", "Ceiling paint is peeling badly",
            registry=HVAC_REGISTRY,
        )
        assert result == []

    def test_results_ranked_highest_score_first(self):
        registry = [
            make_item("HV-001", "HVAC filter replacement needed urgently", "Replace filter now"),
            make_item("HV-002", "HVAC filter check", "Filter might need attention"),
        ]
        result = check_duplicates("HVAC air filter replacement", "Filter replacement overdue", registry=registry)
        if len(result) > 1:
            assert result[0]["score"] >= result[1]["score"]

    def test_score_pct_is_rounded_percentage(self):
        result = check_duplicates(
            "Replace HVAC air filter", "Filter is clogged and overdue for replacement",
            registry=HVAC_REGISTRY,
        )
        if result:
            assert result[0]["score_pct"] == round(result[0]["score"] * 100)

    def test_result_contains_required_fields(self):
        result = check_duplicates(
            "Replace HVAC air filter", "Filter is clogged and overdue for replacement",
            registry=HVAC_REGISTRY,
        )
        if result:
            for key in ("item_id", "title", "category", "status", "score", "score_pct"):
                assert key in result[0]

    def test_empty_candidate_text_returns_no_matches(self):
        assert check_duplicates("", "", registry=HVAC_REGISTRY) == []

    def test_score_is_float_between_0_and_1(self):
        result = check_duplicates("HVAC air filter replacement", "Filter overdue", registry=HVAC_REGISTRY, threshold=0.0)
        for m in result:
            assert 0.0 <= m["score"] <= 1.0


class TestThreshold:
    def test_default_threshold_is_0_55(self):
        assert DEFAULT_THRESHOLD == 0.55

    def test_low_threshold_returns_more_matches(self):
        high = check_duplicates("Replace HVAC filter", "Air filter", threshold=0.9, registry=HVAC_REGISTRY)
        low  = check_duplicates("Replace HVAC filter", "Air filter", threshold=0.3, registry=HVAC_REGISTRY)
        assert len(low) >= len(high)

    def test_high_threshold_returns_fewer_matches(self):
        low  = check_duplicates("Replace HVAC air filter", "Filter overdue", threshold=0.3, registry=HVAC_REGISTRY)
        high = check_duplicates("Replace HVAC air filter", "Filter overdue", threshold=0.95, registry=HVAC_REGISTRY)
        assert len(low) >= len(high)

    def test_threshold_0_0_returns_all_nonempty(self):
        result = check_duplicates("HVAC filter", "Filter", threshold=0.0, registry=HVAC_REGISTRY)
        assert len(result) >= 1

    def test_custom_threshold_respected(self):
        registry = [make_item("HV-001", "HVAC air filter replacement", "Replace the clogged filter")]
        above = check_duplicates("HVAC air filter replacement", "Replace clogged filter", threshold=0.5, registry=registry)
        below = check_duplicates("HVAC air filter replacement", "Replace clogged filter", threshold=0.99, registry=registry)
        assert len(above) >= len(below)


class TestStatusFilter:
    def test_closed_items_excluded_by_default(self):
        result = check_duplicates("Old AC filter replaced", "Completed last month", threshold=0.5, registry=MIXED_STATUS_REGISTRY)
        assert "HV-002" not in [m["item_id"] for m in result]

    def test_open_items_included_by_default(self):
        result = check_duplicates("Replace HVAC air filter", "Filter overdue", threshold=0.5, registry=MIXED_STATUS_REGISTRY)
        assert "HV-001" in [m["item_id"] for m in result]

    def test_in_progress_items_included_by_default(self):
        result = check_duplicates("AC filter in progress", "Currently replacing", threshold=0.5, registry=MIXED_STATUS_REGISTRY)
        assert "HV-003" in [m["item_id"] for m in result]

    def test_custom_status_filter_includes_closed(self):
        result = check_duplicates(
            "Old AC filter replaced", "Completed last month",
            threshold=0.5, status_filter=["open", "in_progress", "closed"],
            registry=MIXED_STATUS_REGISTRY,
        )
        assert "HV-002" in [m["item_id"] for m in result]

    def test_empty_registry_after_status_filter(self):
        closed_only = [make_item("HV-001", "Done item", "Already closed", status="closed")]
        assert check_duplicates("Done item", "Already closed", registry=closed_only) == []


class TestHasDuplicates:
    def test_returns_true_when_match_found(self):
        assert has_duplicates("Replace HVAC air filter", "Filter is clogged and overdue for replacement", registry=HVAC_REGISTRY) is True

    def test_returns_true_for_paraphrased_title(self):
        assert has_duplicates("HVAC Air Filter Replacement", "Air filter hasn't been changed in 6 weeks", registry=HVAC_REGISTRY) is True

    def test_returns_false_when_no_match(self):
        assert has_duplicates("Repaint kitchen ceiling", "Paint peeling from water damage", registry=HVAC_REGISTRY) is False

    def test_returns_false_for_empty_registry(self):
        assert has_duplicates("Replace HVAC filter", "Filter overdue", registry=[]) is False


class TestTopMatch:
    def test_returns_highest_scoring_match(self):
        match = top_match("Replace HVAC air filter", "Filter is clogged and overdue for replacement", registry=HVAC_REGISTRY)
        assert match is not None
        assert match["item_id"] == "HV-001"

    def test_returns_none_when_no_match(self):
        match = top_match("Repaint the kitchen ceiling", "Paint peeling from water damage", registry=HVAC_REGISTRY)
        assert match is None

    def test_returns_none_for_empty_registry(self):
        assert top_match("HVAC filter", "Filter overdue", registry=[]) is None

    def test_top_match_score_is_highest(self):
        all_matches = check_duplicates("Replace HVAC air filter", "Filter is clogged and overdue for replacement", threshold=0.3, registry=HVAC_REGISTRY)
        best = top_match("Replace HVAC air filter", "Filter is clogged and overdue for replacement", threshold=0.3, registry=HVAC_REGISTRY)
        if all_matches and best:
            assert best["score"] == all_matches[0]["score"]


class TestEdgeCases:
    def test_single_word_candidate(self):
        assert isinstance(check_duplicates("filter", "", registry=HVAC_REGISTRY), list)

    def test_very_long_description(self):
        assert isinstance(check_duplicates("Replace HVAC air filter", "HVAC air filter replacement " * 50, registry=HVAC_REGISTRY), list)

    def test_special_characters_in_candidate(self):
        assert isinstance(check_duplicates("HVAC filter (urgent!)", "Filter: needs replacement @ 90 days", registry=HVAC_REGISTRY), list)

    def test_registry_with_one_item(self):
        single = [make_item("HV-001", "Replace HVAC air filter", "Filter overdue")]
        assert isinstance(check_duplicates("HVAC filter replacement", "Air filter needs replacing", registry=single), list)

    def test_genuinely_new_item_not_flagged(self):
        result = check_duplicates("Back deck boards rotting", "Wood rotting safety hazard need replacement", registry=HVAC_REGISTRY)
        assert result == []
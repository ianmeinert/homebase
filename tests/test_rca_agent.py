"""
tests/test_rca_agent.py  —  Tests for cross-item RCA agent.
"""
import json
import pytest


def _make_rca_response(clusters=None, narrative="Systemic issue found.", recommendations=None, confidence=0.75):
    clusters = clusters or [{"cluster_id":"deferred_maintenance","label":"Deferred Maintenance","risk_factor":"Items repeatedly deprioritized","item_ids":["HV-001","PLB-001"],"severity":"high","confidence":0.8}]
    recommendations = recommendations or [{"priority":1,"action":"Schedule HVAC inspection","rationale":"Addresses highest urgency cluster","addresses_clusters":["deferred_maintenance"],"urgency":"immediate"}]
    return json.dumps({"confidence":confidence,"confidence_rationale":"Strong patterns.","pattern_clusters":clusters,"narrative":narrative,"recommendations":recommendations})


def _seed_registry(mock_llm):
    from tools.rca_agent import get_conn
    conn = get_conn()
    items = [
        ("HV-001","hvac","HVAC Filter","desc",0.8,0.7,20,"open"),
        ("PLB-001","plumbing","Leaky faucet","desc",0.6,0.5,5,"open"),
        ("EL-001","electrical","Outlet tripped","desc",0.7,0.8,15,"open"),
        ("APP-001","appliance","Washer vibrates","desc",0.5,0.6,30,"open"),
    ]
    conn.executemany("INSERT OR IGNORE INTO registry VALUES (?,?,?,?,?,?,?,?)", items)
    conn.commit()


def _seed_history(mock_llm):
    from tools.rca_agent import get_conn
    conn = get_conn()
    conn.execute("INSERT INTO run_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 ("r1","2025-01-01T10:00","weekly review","",4,'{"HU/HI":2}',1,1,"","[]","report"))
    conn.commit()


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestLoadFullRegistry:
    def test_returns_expected_keys(self, mock_llm):
        _seed_registry(mock_llm)
        from tools.rca_agent import _load_full_registry
        items = _load_full_registry()
        assert len(items) >= 4
        expected = {"id","category","title","description","urgency","impact","days_since_update","status"}
        assert expected.issubset(items[0].keys())

    def test_returns_closed_status(self, mock_llm):
        from tools.rca_agent import get_conn, _load_full_registry
        conn = get_conn()
        conn.execute("INSERT INTO registry VALUES (?,?,?,?,?,?,?,?)",
                     ("HV-999","hvac","Closed item","desc",0.5,0.5,10,"closed"))
        conn.commit()
        items = _load_full_registry()
        assert any(i["status"] == "closed" for i in items)

    def test_empty_returns_list(self, mock_llm):
        from tools.rca_agent import _load_full_registry
        assert isinstance(_load_full_registry(), list)


class TestLoadRunHistory:
    def test_returns_seeded_run(self, mock_llm):
        _seed_history(mock_llm)
        from tools.rca_agent import _load_run_history
        history = _load_run_history()
        assert any(h["run_id"] == "r1" for h in history)

    def test_respects_limit(self, mock_llm):
        from tools.rca_agent import get_conn, _load_run_history
        conn = get_conn()
        for i in range(15):
            conn.execute("INSERT INTO run_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (f"r{i}",f"2025-01-{i+1:02d}T10:00","test","",3,'{}',0,1,"","[]","r"))
        conn.commit()
        assert len(_load_run_history(limit=5)) == 5

    def test_quadrant_summary_is_dict(self, mock_llm):
        _seed_history(mock_llm)
        from tools.rca_agent import _load_run_history
        history = _load_run_history()
        run = next((h for h in history if h["run_id"] == "r1"), None)
        assert run is not None
        assert isinstance(run["quadrant_summary"], dict)

    def test_empty_returns_list(self, mock_llm):
        from tools.rca_agent import _load_run_history
        assert isinstance(_load_run_history(), list)


# ---------------------------------------------------------------------------
# run_rca integration tests
# ---------------------------------------------------------------------------

class TestRunRca:
    def test_returns_expected_keys(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        result = run_rca("root cause analysis")
        for key in ("clusters","narrative","recommendations","confidence","confidence_rationale","error","item_count","run_count"):
            assert key in result

    def test_no_error_on_success(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        assert run_rca("root cause analysis")["error"] is None

    def test_confidence_is_float_in_range(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response(confidence=0.82)})()
        from tools.rca_agent import run_rca
        conf = run_rca("analyze patterns")["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_clusters_have_required_fields(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        for cl in run_rca("root cause analysis")["clusters"]:
            for f in ("label","risk_factor","item_ids","severity","confidence"):
                assert f in cl

    def test_recommendations_have_required_fields(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        for rec in run_rca("systemic root cause")["recommendations"]:
            for f in ("priority","action","rationale","urgency"):
                assert f in rec

    def test_empty_registry_returns_error_dict(self, mock_llm):
        from tools.rca_agent import run_rca
        result = run_rca("root cause analysis")
        assert "error" in result and "clusters" in result

    def test_item_count_matches_seeded(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        assert run_rca("analyze patterns")["item_count"] >= 4

    def test_run_count_populated(self, mock_llm):
        _seed_registry(mock_llm)
        _seed_history(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        assert run_rca("root cause analysis")["run_count"] >= 1

    def test_parse_error_returns_error(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": "not json"})()
        from tools.rca_agent import run_rca
        result = run_rca("root cause analysis")
        assert result["error"] is not None
        assert "parse error" in result["error"].lower()

    def test_llm_exception_returns_error(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = RuntimeError("API down")
        from tools.rca_agent import run_rca
        result = run_rca("root cause analysis")
        assert result["error"] is not None
        mock_llm.invoke.side_effect = None

    def test_narrative_returned(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response(narrative="Systemic neglect across HVAC and plumbing.")})()
        from tools.rca_agent import run_rca
        assert len(run_rca("what's driving these issues")["narrative"]) > 10

    def test_confidence_rationale_is_string(self, mock_llm):
        _seed_registry(mock_llm)
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        assert isinstance(run_rca("root cause analysis")["confidence_rationale"], str)




# ---------------------------------------------------------------------------
# Category scoping tests
# ---------------------------------------------------------------------------

class TestCategoryScoping:
    def test_registry_filtered_by_category(self, mock_llm):
        from tools.rca_agent import _load_full_registry
        items = _load_full_registry(category="hvac")
        assert all(i["category"] == "hvac" for i in items)

    def test_registry_all_when_no_category(self, mock_llm):
        from tools.rca_agent import _load_full_registry
        items = _load_full_registry(category=None)
        cats = {i["category"] for i in items}
        assert len(cats) > 1  # seeded with multiple categories

    def test_run_rca_category_in_result(self, mock_llm):
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        result = run_rca("rca for hvac", category="hvac")
        assert result.get("category") == "hvac"

    def test_run_rca_no_category_is_none(self, mock_llm):
        mock_llm.invoke.side_effect = lambda *a, **kw: type("R", (), {"content": _make_rca_response()})()
        from tools.rca_agent import run_rca
        result = run_rca("root cause analysis")
        assert result.get("category") is None


class TestExtractRcaCategory:
    @pytest.mark.parametrize("phrase,expected", [
        ("rca for electrical", "electrical"),
        ("root cause analysis plumbing", "plumbing"),
        ("analyze hvac patterns", "hvac"),
        ("systemic issues in appliance", "appliance"),
        ("general category root cause", "general"),
        ("electric systems rca", "electrical"),
        ("appliances root cause", "appliance"),
        ("root cause analysis", None),
        ("weekly review", None),
    ])
    def test_extracts_category(self, phrase, expected):
        from tools.update_agent import extract_rca_category
        assert extract_rca_category(phrase) == expected

# ---------------------------------------------------------------------------
# Intent routing tests
# ---------------------------------------------------------------------------

class TestRcaIntentRouting:
    @pytest.mark.parametrize("phrase", [
        "root cause analysis",
        "run an rca",
        "analyze patterns across items",
        "what's driving these issues",
        "systemic root cause",
        "common factors across registry",
        "pattern analysis",
        "what's causing so many urgent items",
        "cross-item analysis",
    ])
    def test_rca_keywords_route_to_rca(self, phrase):
        from tools.update_agent import classify_input
        assert classify_input(phrase) == "rca", f"Expected rca for: {phrase!r}"

    @pytest.mark.parametrize("phrase", [
        "weekly review",
        "chart urgency by category",
        "add new plumbing item",
        "mark HV-001 in progress",
    ])
    def test_non_rca_phrases_not_rca(self, phrase):
        from tools.update_agent import classify_input
        assert classify_input(phrase) != "rca", f"Expected non-rca for: {phrase!r}"
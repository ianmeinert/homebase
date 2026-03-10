"""
test_subagent_tools.py  -  Unit tests for subagent_tools.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools.subagent_tools import (
    hvac_recommend,
    plumbing_recommend,
    electrical_recommend,
    appliance_recommend,
    general_recommend,
    route_to_subagent,
    CATEGORY_ROUTER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_item(id, category, title, urgency=0.7, impact=0.7, days=10):
    return {
        "id": id, "category": category, "title": title,
        "description": "Test item", "urgency": urgency,
        "impact": impact, "days_since_update": days,
        "status": "open", "quadrant": "HU/HI",
    }


REQUIRED_REC_FIELDS = {"item_id", "action", "estimated_effort", "estimated_cost", "priority_note", "agent"}


def assert_valid_recommendation(rec, expected_agent):
    assert REQUIRED_REC_FIELDS.issubset(rec.keys()), f"Missing fields: {REQUIRED_REC_FIELDS - rec.keys()}"
    assert rec["agent"] == expected_agent
    assert len(rec["action"]) > 0
    assert len(rec["estimated_effort"]) > 0
    assert len(rec["estimated_cost"]) > 0
    assert len(rec["priority_note"]) > 0


# ---------------------------------------------------------------------------
# HVAC recommendations
# ---------------------------------------------------------------------------

class TestHvacRecommend:
    def test_filter_item(self):
        item = make_item("H1", "hvac", "Replace HVAC air filter")
        rec = hvac_recommend(item)
        assert_valid_recommendation(rec, "HVACAgent")
        assert rec["item_id"] == "H1"
        assert "filter" in rec["action"].lower()

    def test_tune_up_item(self):
        item = make_item("H2", "hvac", "Schedule annual AC tune-up")
        rec = hvac_recommend(item)
        assert_valid_recommendation(rec, "HVACAgent")
        assert "tune" in rec["action"].lower() or "hvac" in rec["action"].lower()

    def test_generic_hvac_item(self):
        item = make_item("H3", "hvac", "Strange noise from unit")
        rec = hvac_recommend(item)
        assert_valid_recommendation(rec, "HVACAgent")


# ---------------------------------------------------------------------------
# Plumbing recommendations
# ---------------------------------------------------------------------------

class TestPlumbingRecommend:
    def test_drain_item(self):
        item = make_item("P1", "plumbing", "Fix slow drain in master bath")
        rec = plumbing_recommend(item)
        assert_valid_recommendation(rec, "PlumbingAgent")
        assert "drain" in rec["action"].lower()

    def test_water_heater_item(self):
        item = make_item("P2", "plumbing", "Inspect water heater anode rod")
        rec = plumbing_recommend(item)
        assert_valid_recommendation(rec, "PlumbingAgent")
        assert "anode" in rec["action"].lower() or "plumber" in rec["action"].lower()

    def test_hose_bib_item(self):
        item = make_item("P3", "plumbing", "Leaking exterior hose bib")
        rec = plumbing_recommend(item)
        assert_valid_recommendation(rec, "PlumbingAgent")
        assert "bib" in rec["action"].lower() or "hose" in rec["action"].lower()

    def test_generic_plumbing_item(self):
        item = make_item("P4", "plumbing", "Unknown plumbing issue")
        rec = plumbing_recommend(item)
        assert_valid_recommendation(rec, "PlumbingAgent")


# ---------------------------------------------------------------------------
# Electrical recommendations
# ---------------------------------------------------------------------------

class TestElectricalRecommend:
    def test_gfci_item(self):
        item = make_item("E1", "electrical", "Replace tripping GFCI outlet in garage")
        rec = electrical_recommend(item)
        assert_valid_recommendation(rec, "ElectricalAgent")
        assert "gfci" in rec["action"].lower() or "outlet" in rec["action"].lower()

    def test_motion_light_item(self):
        item = make_item("E2", "electrical", "Install outdoor motion sensor light")
        rec = electrical_recommend(item)
        assert_valid_recommendation(rec, "ElectricalAgent")
        assert "light" in rec["action"].lower() or "motion" in rec["action"].lower()

    def test_generic_electrical_item(self):
        item = make_item("E3", "electrical", "Flickering lights in bedroom")
        rec = electrical_recommend(item)
        assert_valid_recommendation(rec, "ElectricalAgent")


# ---------------------------------------------------------------------------
# Appliance recommendations
# ---------------------------------------------------------------------------

class TestApplianceRecommend:
    def test_dishwasher_item(self):
        item = make_item("A1", "appliance", "Dishwasher not draining fully")
        rec = appliance_recommend(item)
        assert_valid_recommendation(rec, "ApplianceAgent")
        assert "drain" in rec["action"].lower()

    def test_dryer_vent_item(self):
        item = make_item("A2", "appliance", "Dryer vent cleaning overdue")
        rec = appliance_recommend(item)
        assert_valid_recommendation(rec, "ApplianceAgent")
        assert "dryer" in rec["action"].lower() or "vent" in rec["action"].lower()

    def test_ice_maker_item(self):
        item = make_item("A3", "appliance", "Refrigerator ice maker stopped working")
        rec = appliance_recommend(item)
        assert_valid_recommendation(rec, "ApplianceAgent")
        assert "ice" in rec["action"].lower() or "water" in rec["action"].lower()

    def test_generic_appliance_item(self):
        item = make_item("A4", "appliance", "Oven not heating evenly")
        rec = appliance_recommend(item)
        assert_valid_recommendation(rec, "ApplianceAgent")


# ---------------------------------------------------------------------------
# General recommendations
# ---------------------------------------------------------------------------

class TestGeneralRecommend:
    def test_gutter_item(self):
        item = make_item("G1", "general", "Clean gutters")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")
        assert "gutter" in rec["action"].lower()

    def test_gate_latch_item(self):
        item = make_item("G2", "general", "Repair fence gate latch")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")
        assert "latch" in rec["action"].lower() or "gate" in rec["action"].lower()

    def test_caulk_item(self):
        item = make_item("G3", "general", "Caulk master bath shower")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")
        assert "caulk" in rec["action"].lower()

    def test_smoke_detector_item(self):
        item = make_item("G4", "general", "Test smoke and CO detectors")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")
        assert "detector" in rec["action"].lower() or "smoke" in rec["action"].lower()

    def test_paint_item(self):
        item = make_item("G5", "general", "Touch up exterior paint on trim")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")

    def test_generic_general_item(self):
        item = make_item("G6", "general", "Random maintenance task")
        rec = general_recommend(item)
        assert_valid_recommendation(rec, "GeneralAgent")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestRouteToSubagent:
    def test_routes_hvac(self):
        item = make_item("X1", "hvac", "Replace HVAC air filter")
        rec = route_to_subagent(item)
        assert rec["agent"] == "HVACAgent"

    def test_routes_plumbing(self):
        item = make_item("X2", "plumbing", "Fix slow drain")
        rec = route_to_subagent(item)
        assert rec["agent"] == "PlumbingAgent"

    def test_routes_electrical(self):
        item = make_item("X3", "electrical", "Replace GFCI outlet")
        rec = route_to_subagent(item)
        assert rec["agent"] == "ElectricalAgent"

    def test_routes_appliance(self):
        item = make_item("X4", "appliance", "Dryer vent cleaning overdue")
        rec = route_to_subagent(item)
        assert rec["agent"] == "ApplianceAgent"

    def test_routes_general(self):
        item = make_item("X5", "general", "Clean gutters")
        rec = route_to_subagent(item)
        assert rec["agent"] == "GeneralAgent"

    def test_unknown_category_falls_back_to_general(self):
        item = make_item("X6", "unknown_category", "Some task")
        rec = route_to_subagent(item)
        assert rec["agent"] == "GeneralAgent"

    def test_all_categories_covered_in_router(self):
        expected = {"hvac", "plumbing", "electrical", "appliance", "general"}
        assert expected.issubset(CATEGORY_ROUTER.keys())

    def test_item_id_preserved_in_recommendation(self):
        item = make_item("MYID-999", "hvac", "Replace HVAC air filter")
        rec = route_to_subagent(item)
        assert rec["item_id"] == "MYID-999"
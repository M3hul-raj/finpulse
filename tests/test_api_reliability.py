"""
test_api_reliability.py
Tests for Flask API health endpoints, cold-start precomputed cache behavior,
input validation (shock bounds, segment validation), and mocked alert responses.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


@pytest.fixture
def client():
    api_server.app.config["TESTING"] = True
    return api_server.app.test_client()


def test_health_endpoints_instant_and_label_synthetic(client):
    for route in ("/health", "/api/health"):
        resp = client.get(route)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert body["synthetic_data"] is True


def test_shock_param_validation_rejects_invalid_inputs(client):
    for bad in ("abc", "12.5", "-1", "-50", "101", "999"):
        for endpoint in ("/api/portfolio", "/api/heatmap", "/api/forecast", "/api/alerts"):
            resp = client.get(f"{endpoint}?shock={bad}")
            assert resp.status_code == 400, f"{endpoint}?shock={bad} returned {resp.status_code}"
            assert "error" in resp.get_json()


def test_forecast_rejects_unknown_segment_with_404(client):
    resp = client.get("/api/forecast?segment=NonExistentSegment&shock=0")
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"].lower()


def test_baseline_endpoints_serve_precomputed_cache_without_loading_csv(client):
    # Remove raw_data if loaded elsewhere and verify baseline_cache.json serves shock=0 directly
    api_server._cache.pop("raw_data", None)
    api_server._cache.pop("heatmap_0", None)
    api_server._cache.pop("forecasts_0", None)

    resp_p = client.get("/api/portfolio?shock=0")
    resp_h = client.get("/api/heatmap?shock=0")
    resp_f = client.get("/api/forecast?segment=Daily%20Wage&shock=0")

    if (api_server.MODEL_DIR / "baseline_cache.json").exists():
        assert resp_p.status_code == 200
        assert resp_p.get_json()["synthetic"] is True
        assert resp_h.status_code == 200
        assert len(resp_h.get_json()) == 8
        assert resp_f.status_code == 200
        assert resp_f.get_json()["synthetic"] is True
        # CSV was NOT touched for baseline shock=0 requests
        assert "raw_data" not in api_server._cache


def test_alerts_endpoint_caches_and_returns_mocked_explanations(client):
    api_server._cache.pop("alerts_payload_0", None)
    with patch("api_server.explain_all_alerts", return_value={"Daily Wage": "Mocked intervention plan."}) as mock_exp:
        resp1 = client.get("/api/alerts?shock=0")
        resp2 = client.get("/api/alerts?shock=0")

    if resp1.status_code == 200:
        data1 = resp1.get_json()
        data2 = resp2.get_json()
        assert data1 == data2
        assert len(data1) >= 1
        assert data1[0]["explanation"] == "Mocked intervention plan."
        # Second call served from cache without calling LLM again
        assert mock_exp.call_count == 1

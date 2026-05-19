"""API integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ongovoice.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_run_turn_timer(client: TestClient) -> None:
    r = client.post("/api/v1/turns", json={"text": "set a 5 min timer"})
    assert r.status_code == 200
    body = r.json()
    assert body["routing_target"] == "edge"
    assert body["intent"] == "timer"
    assert body["edge_resolved"] is True
    assert body["cost_usd"] == 0.0
    assert body["ttft_ms"] > 0


def test_run_turn_calendar(client: TestClient) -> None:
    r = client.post(
        "/api/v1/turns",
        json={"text": "what's on my calendar this afternoon"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["routing_target"] == "cloud"
    assert body["intent"] == "calendar"
    assert body["edge_resolved"] is False


def test_get_turn_404(client: TestClient) -> None:
    r = client.get("/api/v1/turns/no-such-turn")
    assert r.status_code == 404


def test_metrics_after_some_turns(client: TestClient) -> None:
    # Pre-load a few turns
    client.post("/api/v1/turns", json={"text": "set a timer"})
    client.post("/api/v1/turns", json={"text": "next song"})
    client.post("/api/v1/turns", json={"text": "what's on my calendar"})
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["turns_total"] >= 3
    # Two edge, one cloud → at least 60% edge resolved
    assert body["edge_resolved_pct"] >= 60.0


def test_validation_rejects_empty_text(client: TestClient) -> None:
    r = client.post("/api/v1/turns", json={"text": ""})
    assert r.status_code == 422


def test_validation_rejects_long_text(client: TestClient) -> None:
    r = client.post("/api/v1/turns", json={"text": "x" * 1000})
    assert r.status_code == 422

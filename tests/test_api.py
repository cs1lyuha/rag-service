import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragsvc.api import create_app

KB = Path(__file__).resolve().parent.parent / "kb"
U, A = {"x-api-key": "user-1"}, {"x-api-key": "admin-1"}


@pytest.fixture
def client():
    return TestClient(create_app(user_keys={"user-1"}, admin_key="admin-1", kb_dir=KB))


def test_health_and_documents(client):
    assert client.get("/health").json()["documents"] == 12
    docs = client.get("/documents", headers=U).json()
    assert {d["name"] for d in docs} >= {"livrare", "retur"}


def test_ask_with_citations(client):
    r = client.post("/ask", json={"question": "Cât costă livrarea standard?"}, headers=U).json()
    assert "50 de lei" in r["answer"] and not r["abstained"]
    assert r["citations"][0]["id"] == "livrare#costul-livrarii"


def test_ask_abstains_outside_docs(client):
    r = client.post("/ask", json={"question": "Livrați cu drona pe Lună?"}, headers=U).json()
    assert r["abstained"] and r["citations"] == []


def test_stream_sse(client):
    with client.stream("POST", "/ask/stream", json={"question": "Cât costă livrarea standard?"}, headers=U) as r:
        body = "".join(r.iter_text())
    events = [b for b in body.split("\n\n") if b]
    assert events[0].startswith("event: delta") and events[-1].startswith("event: done")
    done = json.loads(events[-1].split("data: ", 1)[1])
    assert done["citations"][0]["id"] == "livrare#costul-livrarii"


def test_auth(client):
    assert client.post("/ask", json={"question": "ceva"}).status_code == 401
    assert client.put("/documents/x", content="# X", headers=U).status_code == 401  # utilizatorul nu poate ingesta


def test_ingest_update_delete(client):
    doc = "# Program\n\n## Program magazine\nMagazinele TechNova sunt deschise duminica între 10:00 și 16:00."
    assert client.put("/documents/program", content=doc, headers=A).json()["status"] == "created"
    assert client.put("/documents/program", content=doc, headers=A).json()["status"] == "unchanged"
    r = client.post("/ask", json={"question": "Sunt deschise magazinele duminica?"}, headers=U).json()
    assert "duminica" in r["answer"] and r["citations"][0]["id"] == "program#program-magazine"
    assert client.delete("/documents/program", headers=A).status_code == 200
    assert client.delete("/documents/program", headers=A).status_code == 404


@pytest.mark.parametrize("payload", [{"question": "?"}, {"question": "x" * 1001}, {"question": "ok ok", "k": 50}])
def test_input_validation(client, payload):
    assert client.post("/ask", json=payload, headers=U).status_code == 422


def test_bad_document_name(client):
    assert client.put("/documents/..%2Fetc", content="# x", headers=A).status_code in (404, 422)

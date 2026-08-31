from fastapi.testclient import TestClient
from evidence_bureau.api import app
import json

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "model" in data
    assert "collection_count" in data

def test_ask():
    response = client.post(
        "/ask",
        json={
            "question": "What game is the simulation inspired by?",
            "stream": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert data["answer"]

    assert "trace_id" in data
    assert data["trace_id"]

    assert "retrieval" in data

    assert "retrieved_ids" in data["retrieval"]
    assert "reranked_ids" in data["retrieval"]

def test_empty_question_rejected():
    response = client.post(
        "/ask",
        json={
            "question": "",
            "stream": False,
        },
    )

    assert response.status_code == 422

def test_streaming_ask():
    import json

    with client.stream(
        "POST",
        "/ask",
        json={
            "question": "What game is the simulation inspired by?",
            "stream": True,
        },
    ) as response:

        assert response.status_code == 200

        events = []

        for line in response.iter_lines():

            if not line:
                continue

            if line.startswith("data: "):
                events.append(
                    line.removeprefix("data: ")
                )

    parsed_events = [
        json.loads(event)
        for event in events
    ]

    assert parsed_events

    assert parsed_events[0]["event"] == "start"

    event_types = [
        event["event"]
        for event in parsed_events
    ]

    assert "start" in event_types
    assert "token" in event_types
    assert "done" in event_types
from fastapi.testclient import TestClient
from evidence_bureau.api import app
from evidence_bureau.resources import collection
import pymupdf
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
            "question": "What is the capital of France?",
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


# --------------------------------------------------
# Document upload / processing tests
# --------------------------------------------------

def create_test_pdf() -> bytes:
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Evidence Bureau API test document. "
        "The capital of France is Paris."
    )

    pdf_bytes = doc.tobytes()
    doc.close()

    return pdf_bytes


def test_process_document_endpoint():
    pdf_bytes = create_test_pdf()

    response = client.post(
        "/process_document",
        files={
            "file": (
                "api_test.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["filename"].endswith("_api_test.pdf")
    assert data["pages"] == 1
    assert data["chunks_stored"] > 0
    assert data["source_id"]

    stored = collection.get(
        where={"source_id": data["source_id"]}
    )

    assert len(stored["ids"]) == data["chunks_stored"]
    assert len(stored["documents"]) == data["chunks_stored"]
    assert len(stored["metadatas"]) == data["chunks_stored"]

    for metadata in stored["metadatas"]:
        assert metadata["source_id"] == data["source_id"]
        assert metadata["filename"].endswith("_api_test.pdf")


def test_process_document_rejects_non_pdf():
    response = client.post(
        "/process_document",
        files={
            "file": (
                "test.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_process_document_rejects_large_file():
    large_file = b"x" * (26 * 1024 * 1024)

    response = client.post(
        "/process_document",
        files={
            "file": (
                "large.pdf",
                large_file,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]
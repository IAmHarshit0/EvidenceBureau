from pathlib import Path

import pymupdf

from evidence_bureau.process_doc import process_document
from evidence_bureau.resources import collection


def create_test_pdf(path: Path) -> None:
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Evidence Bureau integration test. "
        "The capital of France is Paris. "
        "This document is created dynamically for testing.",
    )

    doc.save(path)
    doc.close()


def test_process_document(tmp_path):
    pdf_path = tmp_path / "test_document.pdf"
    source_id = "test1234"

    create_test_pdf(pdf_path)

    result = process_document(
        pdf_path,
        source_id=source_id,
    )

    assert result["filename"] == "test_document.pdf"
    assert result["source_id"] == source_id
    assert result["pages"] == 1
    assert result["chunks_stored"] > 0

    stored = collection.get(
        where={"source_id": source_id}
    )

    assert len(stored["ids"]) == result["chunks_stored"]
    assert len(stored["documents"]) == result["chunks_stored"]
    assert len(stored["metadatas"]) == result["chunks_stored"]

    for metadata in stored["metadatas"]:
        assert metadata["source_id"] == source_id
        assert metadata["filename"] == "test_document.pdf"
from pathlib import Path
import json

chunk_path = Path("data/chunked_output.json")
with open(chunk_path, "r") as ch:
    data = json.load(ch)

def test_chunks():
    for page in data["pages"]:
        assert "chunks" in page
        assert "chunk_count" in page

def test_chunkMeta():
    for page in data["pages"]:
        for chunk in page["chunks"]:
            assert "chunk_id" in chunk

def test_chunkEmpty():
    for page in data["pages"]:
        for chunk in page["chunks"]:
            assert chunk["text"].strip()

def test_pageInfo():
    for page in data["pages"]:
        for chunk in page["chunks"]:
            assert chunk["chunk_id"].split("_")[1]

def test_chunkSize():
    for page in data["pages"]:
        for chunk in page["chunks"]:
            assert (len(chunk["text"]) <= 500)

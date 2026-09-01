# from pathlib import Path
# from src.evidence_bureau.ingestion import extract_pdf

# pdf_path = Path("data/doc.pdf")
# data = extract_pdf(pdf_path)

# def test_noOfPages():
#     assert data["numberOfPages"] == len(data["pages"])

# def test_structure():
#     for page in data["pages"]:
#         assert "page_number" in page
#         assert "text" in page
#         assert isinstance(page["page_number"], int)
#         assert isinstance(page["text"], list)

# def test_pageNumber():
#     pageNumbers = [
#         page["page_number"] 
#         for page in data["pages"]
#     ]
#     assert pageNumbers == list(range(1, data["numberOfPages"]+1))

# def test_pageText():
#     for page in data["pages"]:
#         assert len(page["text"]) > 0
    
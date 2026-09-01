import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_page_text(lines: list[str]) -> str:
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.isdigit():
            continue
        cleaned_lines.append(line_str)

    full_text = " ".join(cleaned_lines)
    full_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', full_text)
    return full_text


def chunk_pages(data: dict, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    """Takes extract_pdf()'s output dict, returns the same shape with chunks added per page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    processed_pages = []

    for page in data.get("pages", []):
        page_num = page["page_number"]
        page_text = clean_page_text(page.get("text", []))
        text_chunks = splitter.split_text(page_text) if page_text else []

        formatted_chunks = [
            {"chunk_id": f"page_{page_num}_chunk_{idx + 1}", "text": chunk_text}
            for idx, chunk_text in enumerate(text_chunks)
        ]

        processed_pages.append({
            "page_number": page_num,
            "chunk_count": len(formatted_chunks),
            "chunks": formatted_chunks
        })

    return {
        "filename": data.get("filename", "doc.pdf"),
        "numberOfPages": data.get("numberOfPages", len(processed_pages)),
        "pages": processed_pages
    }
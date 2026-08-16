import json
from pathlib import Path
import pymupdf


def extract_pdf(pdf_path: Path) -> dict:
    doc = pymupdf.open(pdf_path)

    pdf_data = {
        "filename": pdf_path.name,
        "numberOfPages": len(doc),
        "pages": [],
    }

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text", sort=True)
        text_lines = page_text.splitlines()

        pdf_data["pages"].append({
            "page_number": page_num + 1,
            "text": text_lines,
        })

    doc.close()

    return pdf_data


def save_json(data: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pdf_path = Path("../data/doc.pdf")
    output_path = Path("../data/output.json")

    data = extract_pdf(pdf_path)
    save_json(data, output_path)

    print(f"Extracted {data['numberOfPages']} pages.")
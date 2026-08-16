import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_path = Path("data/output.json")
output_path = Path("data/chunked_output.json")

def clean_page_text(lines):
    """
    Cleans raw line arrays from a PDF page: removes empty space,
    rejoins hyphenated words, and merges broken lines into continuous paragraphs.
    """
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()
        # Ignore empty lines and isolated page footer numbers
        if not line_str or line_str.isdigit():
            continue
        cleaned_lines.append(line_str)
        
    full_text = " ".join(cleaned_lines)
    
    # Fix words hyphenated across line breaks (e.g., "behav- ior" -> "behavior")
    full_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', full_text)
    
    return full_text

# Load input JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

processed_pages = []

# Process each page individually
for page in data.get("pages", []):
    page_num = page["page_number"]
    raw_text_array = page.get("text", [])
    
    # 1. Clean and merge page text into a single coherent string
    page_text = clean_page_text(raw_text_array)
    
    # 2. Split page text into chunks
    if page_text:
        text_chunks = splitter.split_text(page_text)
    else:
        text_chunks = []
        
    # 3. Format chunks with unique IDs
    formatted_chunks = [
        {
            "chunk_id": f"page_{page_num}_chunk_{idx + 1}",
            "text": chunk_text
        }
        for idx, chunk_text in enumerate(text_chunks)
    ]
    
    # 4. Append structured page data
    processed_pages.append({
        "page_number": page_num,
        "chunk_count": len(formatted_chunks),
        "chunks": formatted_chunks
    })

# Construct final output JSON
output_data = {
    "filename": data.get("filename", "doc.pdf"),
    "numberOfPages": data.get("numberOfPages", len(processed_pages)),
    "pages": processed_pages
}

# Save to a new JSON file
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f" Successfully processed and saved to {output_path}")
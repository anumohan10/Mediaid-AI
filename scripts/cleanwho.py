import json
import os
import re

INPUT_PATH = "data/who_data.json"
OUTPUT_PATH = "cleaned/who_data_cleaned.json"

def normalize_text(text):
    """
    Remove unnecessary whitespace and unicode artifacts.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.replace('\xa0', ' ')).strip()

def clean_who_data(input_path, output_path):
    # Open input in read mode
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = []
    for record in raw_data:
        url = record.get("url", "").strip()
        disease_name = normalize_text(record.get("disease_name", ""))
        sections = record.get("sections", [])

        if not disease_name or disease_name.lower() == "unknown":
            continue  # Skip records without a valid disease name

        cleaned_sections = []
        seen_headings = set()

        for sec in sections:
            heading = normalize_text(sec.get("heading", ""))
            content = normalize_text(sec.get("content", ""))

            if heading and content and heading not in seen_headings:
                seen_headings.add(heading)
                cleaned_sections.append({
                    "heading": heading,
                    "content": content
                })

        if cleaned_sections:
            cleaned_data.append({
                "url": url,
                "disease_name": disease_name,
                "sections": cleaned_sections
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to output file with proper encoding
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned {len(cleaned_data)} WHO records → saved to {output_path}")

if __name__ == "__main__":
    clean_who_data(INPUT_PATH, OUTPUT_PATH)

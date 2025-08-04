import json
import os
import re

INPUT_PATH = "data/cdc_data.json"
OUTPUT_PATH = "cleaned/cdc_data_cleaned.json"

def normalize_text(text):
    """
    Remove unnecessary whitespace and unicode artifacts.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text.replace('\xa0', ' ')).strip()

def clean_cdc_data(input_path, output_path):
    # Load raw CDC data
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = []
    for record in raw_data:
        url = record.get("url", "").strip()
        disease_name = normalize_text(record.get("disease_name", ""))
        sections = record.get("sections", [])

        if not disease_name or disease_name.lower() == "unknown":
            continue  # Skip if disease name is missing

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

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write cleaned data to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned {len(cleaned_data)} CDC records → saved to {output_path}")

if __name__ == "__main__":
    clean_cdc_data(INPUT_PATH, OUTPUT_PATH)

import pdfplumber
import csv
import os
from tqdm import tqdm
import re

pdf_path = os.path.expanduser("~/plumbgenius/ADA.pdf")
csv_path = os.path.expanduser("~/plumbgenius/ada_data.csv")

print(f"Extracting text from {pdf_path}...")
data = []
current_section = ""
try:
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Processing pages"):
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    # Match section numbers (e.g., "602.1", "604")
                    section_match = re.match(r"^(\d+\.\d+|\d+)\s", line)
                    if section_match:
                        current_section = section_match.group(1)
                        text_content = line[len(current_section):].strip()
                        if text_content:
                            data.append({"Section": current_section, "Text": text_content})
                    elif line and current_section:
                        data.append({"Section": current_section, "Text": line})
except Exception as e:
    print(f"Error processing {pdf_path}: {e}")

if not data:
    print("No data extracted! Check PDF content.")

print(f"Writing {len(data)} lines to {csv_path}...")
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Section", "Text"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Conversion complete! Saved to {csv_path}")
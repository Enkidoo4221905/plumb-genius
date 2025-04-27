import pdfplumber
import csv
import os
from tqdm import tqdm
import re
import glob
import json

pdf_dir = os.path.expanduser("~/plumbgenius/")
processed_log = os.path.join(pdf_dir, "processed_pdfs.json")
core_files = ["2021IPC.pdf", "ADA.pdf"]

# Load processed files
processed_files = {}
if os.path.exists(processed_log):
    with open(processed_log, "r") as f:
        processed_files = json.load(f)

# Find new PDFs
all_pdfs = glob.glob(f"{pdf_dir}/*.pdf")
new_pdfs = [f for f in all_pdfs if os.path.basename(f) not in core_files and os.path.basename(f) not in processed_files]

for pdf_path in new_pdfs:
    csv_path = os.path.splitext(pdf_path)[0] + "_data.csv"
    print(f"Processing new PDF: {pdf_path}...")
    data = []
    current_section = "Unknown"  # Default if no section found
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages, desc=f"Extracting {os.path.basename(pdf_path)}"):
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        section_match = re.match(r"^(\d+\.\d+|\d+)\s", line)
                        if section_match:
                            current_section = section_match.group(1)
                            text_content = line[len(current_section):].strip()
                            if text_content:
                                data.append({"Section": current_section, "Text": text_content})
                        else:
                            data.append({"Section": current_section, "Text": line})
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        continue

    if not data:
        print(f"No data extracted from {pdf_path}! Check content.")
        continue

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Section", "Text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    processed_files[os.path.basename(pdf_path)] = {"csv_path": csv_path}
    with open(processed_log, "w") as f:
        json.dump(processed_files, f)
    print(f"Saved to {csv_path}")
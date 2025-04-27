import pdfplumber
from tqdm import tqdm
import os
import re

pdf_path = os.path.expanduser("~/plumbgenius/exam_pdfs.pdf")
qa_pairs = []

print(f"Processing 63-page {pdf_path}...")
try:
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != 63:
            print(f"Warning: Expected 63 pages, found {len(pdf.pages)}")
        full_text = ""
        for page in tqdm(pdf.pages, desc="Extracting 63 pages", total=63):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        # Split into lines and process
        lines = full_text.split("\n")
        current_q = None
        options = []
        for i, line in enumerate(lines):
            line = line.strip()
            # Match numbered questions (e.g., "1.)", "2.)")
            if re.match(r"^\d+\.\)", line) and "?" in line:
                if current_q and options:
                    qa_pairs.append(f"{current_q} | Options: {' | '.join(options)}")
                current_q = line
                options = []
            # Look for options after a question
            elif current_q and line.startswith(("a.", "b.", "c.", "d.")):
                options.append(line)
            # Handle multi-line questions
            elif current_q and "?" in line and not line.startswith(("a.", "b.", "c.", "d.")):
                current_q += " " + line
        # Save last question
        if current_q and options:
            qa_pairs.append(f"{current_q} | Options: {' | '.join(options)}")
except Exception as e:
    print(f"Error processing {pdf_path}: {e}")

if not qa_pairs:
    print("No questions extracted! Check PDF content or script logic.")

output_path = os.path.expanduser("~/plumbgenius/exam_questions.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(qa_pairs))
print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
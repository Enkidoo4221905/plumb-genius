from pdf2csv.converter import convert
import os
import pandas as pd
import psutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Look in current folder
folder = "."
all_tables = []

# Check system resources before starting
logger.info(f"Memory available: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
logger.info(f"CPU usage: {psutil.cpu_percent()}%")

# Find and convert PDFs
pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
if not pdf_files:
    logger.warning("No PDF files found in the current directory.")
else:
    for file in pdf_files:
        pdf_path = os.path.join(folder, file)
        logger.info(f"Processing {file}...")
        try:
            tables = convert(pdf_path, output_format="csv")
            logger.info(f"Conversion completed for {file}")
            if tables:
                for i, table in enumerate(tables):
                    table["Source PDF"] = file
                    table["Table Number"] = i + 1
                    all_tables.append(table)
                logger.info(f"Extracted {len(tables)} table(s) from {file}")
            else:
                logger.info(f"No tables found in {file}")
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}", exc_info=True)

# Save to one CSV
if all_tables:
    try:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv("plumbing_data.csv", index=False)
        logger.info("Saved to plumbing_data.csv")
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}", exc_info=True)
else:
    logger.info("No tables found in any PDFs")

logger.info(f"Final memory available: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
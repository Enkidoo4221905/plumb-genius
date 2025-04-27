import logging
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to preprocess image
def preprocess_image(image):
    try:
        # Check if the image is a valid NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, but got {type(image)}")
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Return the processed grayscale image
        return gray
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# Function to extract text using OCR
def extract_text_from_image(image):
    try:
        # Preprocess image and extract text
        gray_image = preprocess_image(image)
        text = pytesseract.image_to_string(gray_image)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

# Function to process the PDF
def process_pdf(pdf_path, output_file, start_page=0, end_page=None):
    try:
        logger.info(f"Extracting text from {pdf_path} from page {start_page + 1} to {end_page}...")

        # Convert PDF pages to images
        images = convert_from_path(pdf_path, first_page=start_page + 1, last_page=end_page)

        # List to hold extracted text
        extracted_text = []

        # Iterate over pages and extract text using OCR
        for i, image in enumerate(tqdm(images, desc="Processing pages")):
            logger.info(f"Page {start_page + i + 1} is being processed with OCR.")
            image_np = np.array(image)  # Convert PIL image to NumPy array
            text = extract_text_from_image(image_np)
            extracted_text.append({"page": start_page + i + 1, "text": text})
        
        # Save the extracted text to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_text, f, ensure_ascii=False, indent=4)
        logger.info(f"Saving to {output_file}...")
        logger.info(f"Processed {len(extracted_text)} pages.")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

# Main function
def main():
    pdf_path = "2021IPC.pdf"
    chunk_size = 50
    total_pages = 150  # Adjust this to the number of pages in your PDF
    output_dir = "extracted_texts"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        output_file = os.path.join(output_dir, f"plumbing_text_{start_page + 1}_{end_page}.json")
        process_pdf(pdf_path, output_file, start_page=start_page, end_page=end_page)

if __name__ == "__main__":
    main()



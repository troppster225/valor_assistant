from google.cloud import storage
from PyPDF2 import PdfReader
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_bytes
import tempfile
import re
import io
import os
from typing import Optional

class PDFProcessor:
    def __init__(self):
        self.storage_client = storage.Client()

    def parse_and_store_pdf(self, file_content: bytes, filename: str, bucket_name: str) -> bool:
        """Parse PDF content with OCR fallback and enhanced error handling"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            
            print(f"\nProcessing PDF: {filename}")
            print(f"Input content size: {len(file_content)} bytes")
            
            # Store original PDF
            pdf_blob = bucket.blob(f"pdfs/{filename}")
            pdf_blob.upload_from_string(file_content)
            
            # Initialize text extraction
            extracted_text = ""
            
            # Try regular PDF text extraction first
            try:
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PdfReader(pdf_file)
                text_content = []
                
                print("Attempting regular PDF text extraction...")
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text = text.replace('\x00', '')
                            text = re.sub(r'\s+', ' ', text)
                            text = text.strip()
                            text_content.append(f"[Page {page_num + 1}]\n{text}")
                            print(f"Extracted {len(text)} characters from page {page_num + 1}")
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                
                extracted_text = '\n\n'.join(text_content)
            except Exception as e:
                print(f"Regular PDF extraction failed: {str(e)}")
            
            # If regular extraction yielded little or no text, try OCR
            if len(extracted_text.strip()) < 100:
                print("Limited text extracted, attempting OCR...")
                try:
                    with tempfile.TemporaryDirectory() as path:
                        images = convert_from_bytes(file_content, dpi=300, output_folder=path)
                        ocr_texts = []
                        
                        for page_num, image in enumerate(images):
                            print(f"Processing page {page_num + 1} with OCR...")
                            enhanced_image = self._enhance_image_for_ocr(image)
                            text = pytesseract.image_to_string(
                                enhanced_image,
                                config='--oem 3 --psm 6 -l eng'
                            )
                            
                            if text.strip():
                                text = self._clean_ocr_text(text)
                                ocr_texts.append(f"[Page {page_num + 1}]\n{text}")
                                print(f"OCR extracted {len(text)} characters from page {page_num + 1}")
                        
                        if ocr_texts:
                            extracted_text = '\n\n'.join(ocr_texts)
                            print("OCR processing completed successfully")
                        
                except Exception as e:
                    print(f"OCR processing failed: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # Final validation and storage
            if not extracted_text.strip():
                print("No text could be extracted from the document")
                return False
            
            # Store parsed text
            text_filename = f"parsed/{os.path.splitext(filename)[0]}.txt"
            text_blob = bucket.blob(text_filename)
            text_blob.upload_from_string(extracted_text)
            
            print(f"Successfully processed and stored {filename}")
            print(f"Total extracted text length: {len(extracted_text)} characters")
            return True
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        try:
            # Convert to grayscale
            image = image.convert('L')
            
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Resize if too small
            if image.width < 1000:
                ratio = 1000.0 / image.width
                new_size = (1000, int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return image

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up OCR artifacts and improve text quality"""
        try:
            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Fix common OCR mistakes
            replacements = {
                '|': 'I',
                '[': '(',
                ']': ')',
                '{': '(',
                '}': ')',
                'l.': '1.',
                'O0': '00',
                'S': '$'
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Fix common number formats
            text = re.sub(r'(\d+)\.(\d+)', r'\1.\2', text)
            text = re.sub(r'(\d{1,3}(?:,\d{3})+)(?!\d)', lambda m: m.group(1).replace(',', ''), text)
            
            # Fix common currency formats
            text = re.sub(r'([S$])(\d+)', r'$\2', text)
            
            # Clean up lines
            lines = text.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            text = '\n'.join(cleaned_lines)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error cleaning OCR text: {str(e)}")
            return text
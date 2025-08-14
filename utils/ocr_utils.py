#!/usr/bin/env python3
"""
OCR utilities for medical image processing.
Supports reading text from medical reports, prescriptions, and lab results.
"""

import os
import io
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import pdfplumber
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extracts text from a PDF (vector text PDFs like the synthetic ones)."""
        text_pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    text_pages.append(txt.strip())
        return "\n\n".join(text_pages).strip()

class MedicalOCR:
    """OCR processor for medical documents and images."""
    
    def __init__(self):
        """Initialize OCR processor."""
        self.ocr_engine = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engine with a stable default on macOS/Streamlit."""
    # Prefer Tesseract to avoid torch/EasyOCR native crashes
        try:
            import pytesseract
        # Hard-set tesseract binary on Homebrew macOS if env not set
            if not os.getenv("TESSERACT_CMD"):
                pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
            self.ocr_engine = pytesseract
            self.engine_type = 'tesseract'
            logger.info("✅ Tesseract OCR initialized successfully")
            return
        except Exception as e:
            logger.warning(f"Tesseract init failed: {e}")
        
        logger.error("❌ No OCR engine available. Please install pytesseract (and tesseract binary).")
        self.ocr_engine = None
        self.engine_type = None
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.ocr_engine is not None
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Robust preprocessing that avoids native crashes and huge memory spikes."""
        try:
        # Convert and downscale to a safe max dimension (helps stability & accuracy)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.copy()
            image.thumbnail((2200, 2200))  # clamp size

            import cv2, numpy as np
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Mild denoise; avoid heavy kernels
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Use Otsu instead of adaptiveThreshold (more stable across platforms)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return Image.fromarray(thresh)
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}. Using original image.")
            return image
    
    def extract_text_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR."""
        try:
            # Convert PIL image to numpy array
            import numpy as np
            image_array = np.array(image)
            
            # Extract text
            results = self.ocr_engine.readtext(image_array)
            
            # Combine all text
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low-confidence results
                    extracted_text.append(text)
            
            return ' '.join(extracted_text)
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """Extract text using Tesseract."""
        try:
            # Configure Tesseract for better medical text recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/() '
            
            text = self.ocr_engine.image_to_string(image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""
    
    def extract_text_from_image(self, image: Image.Image, preprocess: bool = True) -> Dict[str, str]:
        """Extract text from medical image."""
        if not self.is_available():
            return {
                'success': False,
                'text': '',
                'error': 'OCR engine not available'
            }
        
        try:
            # Preprocess image if requested
            if preprocess:
                processed_image = self.preprocess_image(image)
            else:
                img = image.copy()
                img.thumbnail((2200, 2200))  # clamp size
                processed_image = img

            # Extract text based on available engine
            if self.engine_type == 'easyocr':
                text = self.extract_text_easyocr(processed_image)
            elif self.engine_type == 'tesseract':
                text = self.extract_text_tesseract(processed_image)
            else:
                return {
                    'success': False,
                    'text': '',
                    'error': 'No OCR engine available'
                }
            
            # Clean and format text
            cleaned_text = self.clean_extracted_text(text)
            
            return {
                'success': True,
                'text': cleaned_text,
                'original_text': text,
                'engine': self.engine_type
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                'success': False,
                'text': '',
                'error': str(e)
            }
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and format extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors in medical text
        replacements = {
            '0': 'O',  # Zero to O in some medical contexts
            'l': 'I',  # lowercase l to I
            '|': 'I',  # pipe to I
        }
        
        # Apply replacements contextually (only for isolated characters)
        words = cleaned.split()
        for i, word in enumerate(words):
            if len(word) == 1 and word in replacements:
                words[i] = replacements[word]
        
        return ' '.join(words)
    
    def analyze_medical_document(self, text: str) -> Dict[str, List[str]]:
        """Analyze extracted text to identify medical document components."""
        if not text:
            return {}
        
        text_lower = text.lower()
        analysis = {}
        
        # Look for medications
        medication_keywords = ['mg', 'ml', 'tablet', 'capsule', 'syrup', 'injection', 'drops', 'ointment']
        medications = []
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in medication_keywords):
                medications.append(line.strip())
        if medications:
            analysis['medications'] = medications
        
        # Look for vital signs
        vital_patterns = ['bp:', 'blood pressure:', 'pulse:', 'temperature:', 'weight:', 'height:']
        vitals = []
        for line in lines:
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in vital_patterns):
                vitals.append(line.strip())
        if vitals:
            analysis['vitals'] = vitals
        
        # Look for diagnoses
        diagnosis_keywords = ['diagnosis:', 'diagnosed with', 'condition:', 'disease:', 'disorder:']
        diagnoses = []
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in diagnosis_keywords):
                diagnoses.append(line.strip())
        if diagnoses:
            analysis['diagnoses'] = diagnoses
        
        # Look for lab values
        lab_keywords = ['hemoglobin', 'glucose', 'cholesterol', 'creatinine', 'urea', 'hba1c']
        lab_values = []
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in lab_keywords):
                lab_values.append(line.strip())
        if lab_values:
            analysis['lab_values'] = lab_values
        
        return analysis

    
def create_ocr_interface():
    """Create Streamlit interface for OCR / PDF text extraction."""
    st.subheader("📄 Medical Document Reader")

    # Init OCR (for images). Do not hard-fail if missing.
    if 'ocr_processor' not in st.session_state:
        st.session_state.ocr_processor = MedicalOCR()
    ocr = st.session_state.ocr_processor

    # Status badge only (no early return)
    if ocr.is_available():
        st.success(f"OCR available: {ocr.engine_type}")
    else:
        st.info("📄 PDF text extraction is available. Image OCR is currently disabled.")

    uploaded_file = st.file_uploader(
        "Upload medical document (image or PDF)",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a prescription, report, or lab result (PDF preferred for synthetic docs)."
    )

    # Nothing uploaded yet
    if uploaded_file is None:
        return None

    # Determine extension safely
    try:
        filename = getattr(uploaded_file, "name", "") or ""
        suffix = os.path.splitext(filename)[1].lower()
    except Exception:
        suffix = ""

    # --- Handle PDF first (works without OCR) ---
    if suffix == ".pdf":
        try:
            with st.spinner("Reading PDF..."):
                pdf_bytes = uploaded_file.read()
                text = extract_text_from_pdf_bytes(pdf_bytes)  # uses pdfplumber
            if text:
                st.session_state.extracted_text = text
                st.session_state.document_analysis = ocr.analyze_medical_document(text)
                st.subheader("📝 Extracted Text")
                st.text_area("Text from document:", text, height=220)
                return text
            else:
                st.warning("No selectable text found in the PDF (might be a scan). You can try converting pages to images and OCR.")
                return None
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return None

    # --- Otherwise treat as image (OCR path) ---
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return None

    st.image(image, caption=filename or "Uploaded Image", use_container_width=True)

    preprocess = st.checkbox("Enhance image for better OCR", value=True)
    if st.button("Extract Text", type="primary"):
        if not ocr.is_available():
            st.error("OCR engine not available. Install EasyOCR or Tesseract to process images.")
            return None
        with st.spinner("Running OCR..."):
            result = ocr.extract_text_from_image(image, preprocess=preprocess)
        if result.get("success"):
            text = result.get("text", "")
            st.session_state.extracted_text = text
            st.session_state.document_analysis = ocr.analyze_medical_document(text)
            st.subheader("📝 Extracted Text")
            st.text_area("Text from document:", text, height=220)
            return text
        else:
            st.error(f"OCR failed: {result.get('error','unknown error')}")
            return None

    return None


# Test function for OCR
def test_ocr():
    """Test OCR functionality."""
    ocr = MedicalOCR()
    
    if ocr.is_available():
        print(f"✅ OCR available using: {ocr.engine_type}")
        return True
    else:
        print("❌ OCR not available")
        return False

if __name__ == "__main__":
    test_ocr()

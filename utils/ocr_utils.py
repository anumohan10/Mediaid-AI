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
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalOCR:
    """OCR processor for medical documents and images."""
    
    def __init__(self):
        """Initialize OCR processor."""
        self.ocr_engine = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize the OCR engine."""
        try:
            # Try EasyOCR first (more accurate for medical text)
            import easyocr
            self.ocr_engine = easyocr.Reader(['en'])
            self.engine_type = 'easyocr'
            logger.info("✅ EasyOCR initialized successfully")
        except ImportError:
            try:
                # Fallback to Tesseract
                import pytesseract
                self.ocr_engine = pytesseract
                self.engine_type = 'tesseract'
                logger.info("✅ Tesseract OCR initialized successfully")
            except ImportError:
                logger.error("❌ No OCR engine available. Please install easyocr or pytesseract")
                self.ocr_engine = None
                self.engine_type = None
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.ocr_engine is not None
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL
            processed_image = Image.fromarray(thresh)
            
            return processed_image
            
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
                processed_image = image
            
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
    """Create Streamlit interface for OCR functionality."""
    st.subheader("📄 Medical Document Reader")
    
    # Initialize OCR
    if 'ocr_processor' not in st.session_state:
        st.session_state.ocr_processor = MedicalOCR()
    
    ocr = st.session_state.ocr_processor
    
    if not ocr.is_available():
        st.error("❌ OCR engine not available. Please install required dependencies.")
        st.info("Run: `pip install easyocr pytesseract opencv-python`")
        return None
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload medical document (image)",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image of a medical report, prescription, or lab result"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_container_width=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            preprocess = st.checkbox("Enhance image for better OCR", value=True)
        with col2:
            if st.button("Extract Text", type="primary"):
                with st.spinner("Extracting text from image..."):
                    result = ocr.extract_text_from_image(image, preprocess=preprocess)
                
                if result['success']:
                    st.success("✅ Text extracted successfully!")
                    
                    # Store extracted text in session state
                    st.session_state.extracted_text = result['text']
                    st.session_state.document_analysis = ocr.analyze_medical_document(result['text'])
                    
                    # Display extracted text
                    st.subheader("📝 Extracted Text:")
                    st.text_area("Text from document:", result['text'], height=200)
                    
                    # Display analysis if available
                    if st.session_state.document_analysis:
                        st.subheader("🔍 Document Analysis:")
                        for category, items in st.session_state.document_analysis.items():
                            st.write(f"**{category.title()}:**")
                            for item in items:
                                st.write(f"- {item}")
                    
                    return result['text']
                    
                else:
                    st.error(f"❌ OCR failed: {result['error']}")
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

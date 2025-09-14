"""
Tesseract OCR Engine Implementation
A comprehensive OCR solution using Tesseract and OpenCV
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    """
    A comprehensive OCR engine using Tesseract and OpenCV
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the OCR engine
        
        Args:
            tesseract_path: Path to tesseract executable (auto-detected if None)
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try to find tesseract in common Windows locations
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.tesseract_path = path
                    break
        
        # Test tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized successfully at: {self.tesseract_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise RuntimeError("Tesseract not found. Please install Tesseract OCR.")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image], 
                        denoise: bool = True, 
                        enhance_contrast: bool = True,
                        remove_noise: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            denoise: Apply denoising
            enhance_contrast: Enhance contrast
            remove_noise: Remove noise
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy array
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        if denoise:
            gray = cv2.medianBlur(gray, 3)
        
        # Enhance contrast
        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Remove noise
        if remove_noise:
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return gray
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    lang: str = 'eng',
                    config: str = '--psm 6',
                    preprocess: bool = True) -> str:
        """
        Extract text from image using Tesseract
        
        Args:
            image: Input image
            lang: Language code (e.g., 'eng', 'fra', 'spa')
            config: Tesseract configuration
            preprocess: Whether to preprocess the image
            
        Returns:
            Extracted text
        """
        try:
            if preprocess:
                processed_img = self.preprocess_image(image)
            else:
                if isinstance(image, str):
                    processed_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                elif isinstance(image, Image.Image):
                    processed_img = np.array(image.convert('L'))
                else:
                    processed_img = image
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, lang=lang, config=config)
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def extract_text_with_confidence(self, image: Union[str, np.ndarray, Image.Image],
                                   lang: str = 'eng',
                                   config: str = '--psm 6',
                                   preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text with confidence score
        
        Args:
            image: Input image
            lang: Language code
            config: Tesseract configuration
            preprocess: Whether to preprocess the image
            
        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        try:
            if preprocess:
                processed_img = self.preprocess_image(image)
            else:
                if isinstance(image, str):
                    processed_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                elif isinstance(image, Image.Image):
                    processed_img = np.array(image.convert('L'))
                else:
                    processed_img = image
            
            # Get detailed data
            data = pytesseract.image_to_data(processed_img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, lang=lang, config=config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text.strip(), avg_confidence
        
        except Exception as e:
            logger.error(f"Error extracting text with confidence: {e}")
            return "", 0.0
    
    def extract_text_boxes(self, image: Union[str, np.ndarray, Image.Image],
                          lang: str = 'eng',
                          config: str = '--psm 6',
                          preprocess: bool = True) -> List[Dict]:
        """
        Extract text with bounding boxes
        
        Args:
            image: Input image
            lang: Language code
            config: Tesseract configuration
            preprocess: Whether to preprocess the image
            
        Returns:
            List of dictionaries containing text, confidence, and bounding box info
        """
        try:
            if preprocess:
                processed_img = self.preprocess_image(image)
            else:
                if isinstance(image, str):
                    processed_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                elif isinstance(image, Image.Image):
                    processed_img = np.array(image.convert('L'))
                else:
                    processed_img = image
            
            # Get detailed data
            data = pytesseract.image_to_data(processed_img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            
            # Process results
            results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:  # Only include non-empty text
                    results.append({
                        'text': text,
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error extracting text boxes: {e}")
            return []
    
    def draw_text_boxes(self, image: Union[str, np.ndarray, Image.Image],
                       text_boxes: List[Dict],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected text
        
        Args:
            image: Input image
            text_boxes: List of text box dictionaries
            color: BGR color for boxes
            thickness: Box thickness
            
        Returns:
            Image with drawn boxes
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        for box in text_boxes:
            bbox = box['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Draw text
            cv2.putText(img, box['text'], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available Tesseract languages
        
        Returns:
            List of language codes
        """
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.error(f"Error getting languages: {e}")
            return []
    
    def batch_process(self, image_paths: List[str],
                     output_dir: str = "ocr_output",
                     lang: str = 'eng',
                     config: str = '--psm 6',
                     preprocess: bool = True) -> Dict[str, str]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            lang: Language code
            config: Tesseract configuration
            preprocess: Whether to preprocess images
            
        Returns:
            Dictionary mapping file paths to extracted text
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                text = self.extract_text(image_path, lang=lang, config=config, preprocess=preprocess)
                results[image_path] = text
                
                # Save text to file
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = os.path.join(output_dir, f"{base_name}_text.txt")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = ""
        
        return results


def main():
    """
    Example usage of the OCR engine
    """
    # Initialize OCR engine
    ocr = OCREngine()
    
    # Print available languages
    print("Available languages:", ocr.get_available_languages())
    
    # Example: Process an image (you'll need to provide an actual image path)
    # image_path = "path/to/your/image.jpg"
    # text = ocr.extract_text(image_path)
    # print("Extracted text:", text)
    
    print("OCR Engine initialized successfully!")
    print("Usage example:")
    print("ocr = OCREngine()")
    print("text = ocr.extract_text('path/to/image.jpg')")


if __name__ == "__main__":
    main()

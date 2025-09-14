"""
Simple OCR Script
Input: Image file path
Output: Detected text with coordinates
"""

import sys
import os
from ocr_engine import OCREngine

def extract_text_with_coordinates(image_path):
    """
    Extract text with coordinates from an image
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of dictionaries with text and coordinates
    """
    # Initialize OCR engine
    ocr = OCREngine()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return []
    
    # Extract text boxes with coordinates
    text_boxes = ocr.extract_text_boxes(image_path)
    
    return text_boxes

def main():
    """
    Main function to run the simple OCR
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python simple_ocr.py <image_path>")
        print("Example: python simple_ocr.py sample_text.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Processing image: {image_path}")
    print("=" * 50)
    
    # Extract text with coordinates
    results = extract_text_with_coordinates(image_path)
    
    if not results:
        print("No text detected in the image.")
        return
    
    # Display results
    print("Detected Text with Coordinates:")
    print("-" * 50)
    
    for i, box in enumerate(results, 1):
        text = box['text']
        x = box['bbox']['x']
        y = box['bbox']['y']
        width = box['bbox']['width']
        height = box['bbox']['height']
        
        print(f"{i:2d}. Text: '{text}'")
        print(f"    Position: ({x}, {y})")
        print(f"    Size: {width} x {height}")
        print()

if __name__ == "__main__":
    main()

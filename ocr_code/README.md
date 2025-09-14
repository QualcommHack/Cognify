# Tesseract OCR Implementation for Python 3.10

A comprehensive OCR (Optical Character Recognition) solution using Tesseract and OpenCV, designed specifically for Python 3.10.

## Features

- **Easy-to-use OCR engine** with Tesseract integration
- **Image preprocessing** for better OCR accuracy
- **Multiple output formats**: plain text, text with confidence scores, and text with bounding boxes
- **Batch processing** for multiple images
- **Multiple language support**
- **Various PSM (Page Segmentation Mode) options**
- **Visual debugging** with bounding box visualization
- **Comprehensive error handling and logging**

## Installation

### 1. Install Tesseract OCR Engine

The Tesseract OCR engine has already been installed on your system via winget.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
from ocr_engine import OCREngine
ocr = OCREngine()
print("OCR Engine initialized successfully!")
```

## Quick Start

### Basic Text Extraction

```python
from ocr_engine import OCREngine

# Initialize OCR engine
ocr = OCREngine()

# Extract text from image
text = ocr.extract_text('path/to/your/image.jpg')
print(text)
```

### Text Extraction with Confidence

```python
# Extract text with confidence score
text, confidence = ocr.extract_text_with_confidence('image.jpg')
print(f"Text: {text}")
print(f"Confidence: {confidence:.2f}%")
```

### Text with Bounding Boxes

```python
# Extract text with bounding box information
text_boxes = ocr.extract_text_boxes('image.jpg')

for box in text_boxes:
    print(f"Text: {box['text']}")
    print(f"Confidence: {box['confidence']}%")
    print(f"Position: {box['bbox']}")
```

### Batch Processing

```python
# Process multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = ocr.batch_process(image_paths, output_dir="ocr_results")

for path, text in results.items():
    print(f"{path}: {text}")
```

## Advanced Usage

### Image Preprocessing

The OCR engine includes automatic image preprocessing to improve accuracy:

```python
# With preprocessing (default)
text = ocr.extract_text('image.jpg', preprocess=True)

# Without preprocessing
text = ocr.extract_text('image.jpg', preprocess=False)
```

### Different Languages

```python
# Extract text in different languages
text_english = ocr.extract_text('image.jpg', lang='eng')
text_french = ocr.extract_text('image.jpg', lang='fra')
text_spanish = ocr.extract_text('image.jpg', lang='spa')

# Check available languages
languages = ocr.get_available_languages()
print(f"Available languages: {languages}")
```

### Different PSM Modes

```python
# Different page segmentation modes
text_auto = ocr.extract_text('image.jpg', config='--psm 3')  # Fully automatic
text_block = ocr.extract_text('image.jpg', config='--psm 6')  # Uniform block
text_word = ocr.extract_text('image.jpg', config='--psm 8')   # Single word
text_line = ocr.extract_text('image.jpg', config='--psm 13')  # Single line
```

### Visual Debugging

```python
# Draw bounding boxes on image
text_boxes = ocr.extract_text_boxes('image.jpg')
img_with_boxes = ocr.draw_text_boxes('image.jpg', text_boxes)
cv2.imwrite('output_with_boxes.jpg', img_with_boxes)
```

## File Structure

```
ocrr/
├── ocr_engine.py          # Main OCR engine implementation
├── example_usage.py       # Comprehensive usage examples
├── test_ocr.py           # Test suite
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

## API Reference

### OCREngine Class

#### `__init__(tesseract_path=None)`
Initialize the OCR engine. If `tesseract_path` is not provided, it will auto-detect the Tesseract installation.

#### `extract_text(image, lang='eng', config='--psm 6', preprocess=True)`
Extract text from an image.

**Parameters:**
- `image`: Input image (file path, numpy array, or PIL Image)
- `lang`: Language code (default: 'eng')
- `config`: Tesseract configuration (default: '--psm 6')
- `preprocess`: Whether to preprocess the image (default: True)

**Returns:** Extracted text as string

#### `extract_text_with_confidence(image, lang='eng', config='--psm 6', preprocess=True)`
Extract text with confidence score.

**Returns:** Tuple of (text, average_confidence)

#### `extract_text_boxes(image, lang='eng', config='--psm 6', preprocess=True)`
Extract text with bounding box information.

**Returns:** List of dictionaries containing text, confidence, and bounding box data

#### `preprocess_image(image, denoise=True, enhance_contrast=True, remove_noise=True)`
Preprocess image for better OCR results.

**Returns:** Preprocessed image as numpy array

#### `batch_process(image_paths, output_dir="ocr_output", lang='eng', config='--psm 6', preprocess=True)`
Process multiple images in batch.

**Returns:** Dictionary mapping file paths to extracted text

## Running Examples

### Run Basic Examples
```bash
python example_usage.py
```

### Run Tests
```bash
python test_ocr.py
```

## Troubleshooting

### Common Issues

1. **Tesseract not found error**
   - Make sure Tesseract is installed and in your PATH
   - Or specify the path manually: `OCREngine(tesseract_path="C:/Program Files/Tesseract-OCR/tesseract.exe")`

2. **Low OCR accuracy**
   - Try different PSM modes
   - Enable preprocessing
   - Ensure good image quality
   - Use appropriate language setting

3. **Memory issues with large images**
   - Resize images before processing
   - Use batch processing for multiple images

### Performance Tips

- Use appropriate PSM mode for your use case
- Enable preprocessing for better accuracy
- Process images in batches for efficiency
- Consider image resolution (300 DPI is often optimal)

## Dependencies

- Python 3.10+
- pytesseract 0.3.10
- Pillow >= 9.0.0
- opencv-python >= 4.5.0
- numpy >= 1.21.0

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this OCR implementation.

## Support

For questions or issues, please check the troubleshooting section or create an issue in the project repository.

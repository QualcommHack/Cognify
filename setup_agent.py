"""
Setup script for Computer Use Agent
This script helps install dependencies and check system requirements
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def check_tesseract():
    """Check if Tesseract is installed"""
    print("Checking Tesseract OCR installation...")
    
    # Common Windows paths
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            print(f"✓ Tesseract found at: {path}")
            return True
    
    print("✗ Tesseract not found. Please install Tesseract OCR:")
    print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  Or use: winget install UB-Mannheim.TesseractOCR")
    return False

def check_genie_bundle():
    """Check if Genie bundle is available"""
    print("Checking Genie LLM bundle...")
    
    genie_path = "genie_bundle"
    genie_exe = os.path.join(genie_path, "genie-t2t-run.exe")
    config_file = os.path.join(genie_path, "genie_config.json")
    
    if os.path.exists(genie_exe) and os.path.exists(config_file):
        print("✓ Genie bundle found and configured")
        return True
    else:
        print("✗ Genie bundle not found or incomplete")
        print(f"  Expected: {genie_exe}")
        print(f"  Expected: {config_file}")
        return False

def main():
    """Main setup function"""
    print("=== Computer Use Agent Setup ===\n")
    
    success = True
    
    # Check all components
    success &= install_requirements()
    success &= check_tesseract()
    success &= check_genie_bundle()
    
    print("\n=== Setup Summary ===")
    if success:
        print("✓ All components are ready!")
        print("\nYou can now run the agent with:")
        print("  python computer_agent.py")
    else:
        print("✗ Some components need attention. Please fix the issues above.")
    
    return success

if __name__ == "__main__":
    main()

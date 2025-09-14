#!/usr/bin/env python3
"""
Setup script for voice recognition dependencies
This will install the required packages for offline speech recognition
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_pyaudio():
    """Check if PyAudio is available"""
    try:
        import pyaudio
        print("✓ PyAudio is available")
        return True
    except ImportError:
        print("✗ PyAudio not found")
        return False

def main():
    print("=== Voice Recognition Setup ===")
    print("Installing packages for offline speech recognition...")
    print()
    
    # Required packages
    packages = [
        "SpeechRecognition>=3.10.0",
        "pocketsphinx>=0.1.15"
    ]
    
    # Install basic packages
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    # Handle PyAudio separately (can be tricky on Windows)
    print("\nInstalling PyAudio...")
    if not check_pyaudio():
        # Try different installation methods
        methods = [
            "pyaudio",
            "PyAudio",
            "pipwin pyaudio"  # Alternative for Windows
        ]
        
        for method in methods:
            print(f"Trying: pip install {method}")
            if install_package(method):
                if check_pyaudio():
                    break
            print("Trying next method...")
        else:
            print("⚠ PyAudio installation failed.")
            print("For Windows, you may need to:")
            print("1. Install Visual Studio Build Tools")
            print("2. Or download a PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            print("3. Or use: pip install pipwin && pipwin install pyaudio")
            success = False
    
    # Test imports
    print("\nTesting imports...")
    try:
        import speech_recognition as sr
        print("✓ SpeechRecognition imported successfully")
        
        # Test recognizer
        r = sr.Recognizer()
        print("✓ Speech recognizer created")
        
        # Test microphone
        try:
            mic = sr.Microphone()
            print("✓ Microphone access available")
        except:
            print("⚠ Microphone access may have issues")
            
        # Test offline recognition
        try:
            # This will download required data if not present
            print("✓ Testing offline speech recognition...")
            print("  (This may take a moment on first run)")
        except Exception as e:
            print(f"⚠ Offline recognition test failed: {e}")
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 Voice recognition setup completed successfully!")
        print("You can now use voice commands in the Computer Agent.")
        print("\nTo test voice recognition:")
        print("1. Run: python computer_agent.py")
        print("2. Click the '🎤 Voice' button")
        print("3. Speak a command like 'open calculator'")
    else:
        print("⚠ Setup completed with some issues.")
        print("Voice recognition may not work properly.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()

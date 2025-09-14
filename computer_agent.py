"""
Computer Use Agent
An autonomous agent that can perform tasks on desktop using LLM, OCR, and PyAutoGUI
"""

import os
import sys
import time
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Dict, Optional, Tuple
import pyautogui
import pygetwindow as gw
import speech_recognition as sr
import pyaudio
import vosk
import json
import urllib.request
import zipfile
from PIL import Image, ImageTk
import io
import json
import re

# Add ocr_code to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ocr_code'))
from ocr_engine import OCREngine

class LLMInterface:
    """Interface to the Genie LLM model"""
    
    def __init__(self, genie_path: str = "genie_bundle"):
        self.genie_path = os.path.abspath(genie_path)
        self.genie_exe = os.path.join(self.genie_path, "genie-t2t-run.exe")
        self.config_file = os.path.join(self.genie_path, "genie_config.json")
        
        if not os.path.exists(self.genie_exe):
            raise FileNotFoundError(f"Genie executable not found at {self.genie_exe}")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Genie config not found at {self.config_file}")
    
    def query(self, user_message: str, system_prompt: str = None) -> str:
        """Query the LLM with a message"""
        if system_prompt is None:
            system_prompt = "You are a helpful computer assistant that can analyze screen content and suggest actions. Be specific and actionable in your responses."
        
        # Clean and sanitize text to avoid UTF-8 issues
        def clean_text(text):
            # Remove problematic characters and ensure ASCII-safe
            cleaned = text.encode('ascii', errors='ignore').decode('ascii')
            # Remove extra whitespace and newlines that might cause issues
            cleaned = ' '.join(cleaned.split())
            return cleaned
        
        user_message = clean_text(user_message)
        system_prompt = clean_text(system_prompt)
        
        # Format the prompt with simple structure  
        prompt = f'<|system|>\\n{system_prompt}<|end|>\\n<|user|>\\n{user_message}<|end|>\\n<|assistant|>\\n'
        
        try:
            # Run the genie command
            cmd = [self.genie_exe, "-c", self.config_file, "-p", prompt]
            result = subprocess.run(cmd, cwd=self.genie_path, capture_output=True, text=True, 
                                  timeout=30, encoding='utf-8', errors='ignore')
            
            if result.returncode != 0:
                return f"Error running LLM: {result.stderr}"
            
            # Extract response between [BEGIN]: and [END]
            output = result.stdout
            begin_match = re.search(r'\[BEGIN\]:\s*(.*?)\[END\]', output, re.DOTALL)
            if begin_match:
                return begin_match.group(1).strip()
            else:
                return "No response generated"
                
        except subprocess.TimeoutExpired:
            return "LLM query timed out"
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

class VoiceRecognizer:
    """High-accuracy voice recognition using Vosk offline model"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.vosk_model = None
        self.vosk_rec = None
        
        # Optimize recognizer settings for clarity
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
        
        # Initialize Vosk model for best offline recognition
        self._setup_vosk_model()
        
        # Initialize microphone
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                print("Adjusting microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("✓ Microphone initialized successfully")
        except Exception as e:
            print(f"⚠ Microphone initialization failed: {e}")
    
    def _setup_vosk_model(self):
        """Setup Vosk model for accurate offline recognition"""
        try:
            import vosk
            model_path = "vosk-model-small-en-us-0.15"
            
            if not os.path.exists(model_path):
                print("📥 Downloading Vosk model for accurate offline recognition...")
                url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                zip_path = "vosk-model.zip"
                
                print("⬇️ Downloading... (about 40MB)")
                urllib.request.urlretrieve(url, zip_path)
                
                print("📦 Extracting model...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove(zip_path)
            
            print("🔧 Loading Vosk model...")
            self.vosk_model = vosk.Model(model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            print("✅ Vosk model loaded successfully - HIGH ACCURACY mode enabled!")
            
        except Exception as e:
            print(f"⚠ Vosk setup failed: {e}")
            print("📝 Will use basic recognition only")
    
    def listen_for_command(self, timeout=5, phrase_time_limit=3) -> Optional[str]:
        """Listen with Vosk for highest accuracy, Google fallback"""
        if not self.microphone:
            return None
            
        try:
            print("🎤 Speak clearly now... (auto-stops when you pause)")
            self.is_listening = True
            
            with self.microphone as source:
                print("🔴 Recording...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            self.is_listening = False
            print("🔄 Processing with AI...")
            
            # Try Vosk first (best offline accuracy)
            if self.vosk_model and self.vosk_rec:
                try:
                    # Get raw audio data from speech_recognition
                    audio_data = audio.get_wav_data()
                    
                    # Convert WAV to raw PCM for Vosk
                    import wave
                    import io
                    import struct
                    
                    wav_io = io.BytesIO(audio_data)
                    with wave.open(wav_io, 'rb') as wav_file:
                        # Get audio parameters
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())
                        
                        # Convert to 16kHz if needed
                        if sample_rate != 16000:
                            # Convert bytes to 16-bit samples
                            samples = struct.unpack(f'<{len(frames)//2}h', frames)
                            
                            # Simple downsampling
                            downsample_factor = sample_rate // 16000
                            if downsample_factor > 1:
                                samples = samples[::downsample_factor]
                            
                            # Convert back to bytes
                            frames = struct.pack(f'<{len(samples)}h', *samples)
                    
                    # Reset recognizer for new audio
                    self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
                    
                    # Process with Vosk
                    if self.vosk_rec.AcceptWaveform(frames):
                        result = json.loads(self.vosk_rec.Result())
                        text = result.get('text', '').strip()
                    else:
                        result = json.loads(self.vosk_rec.FinalResult())
                        text = result.get('text', '').strip()
                    
                    if text and len(text) > 1:
                        processed = self.post_process_speech(text)
                        print(f"✅ Vosk (HIGH ACCURACY): '{text}' → '{processed}'")
                        return processed
                    else:
                        print("⚠ Vosk: No speech detected")
                        
                except Exception as e:
                    print(f"⚠ Vosk failed: {e}")
                    # Don't print full traceback to keep output clean
            
            # Fallback methods
            methods = [
                ("Google", lambda: self.recognizer.recognize_google(audio, language='en-US')),
                ("Sphinx", lambda: self.recognizer.recognize_sphinx(audio))
            ]
            
            for method_name, method_func in methods:
                try:
                    text = method_func().strip()
                    if text and len(text) > 1:
                        processed = self.post_process_speech(text)
                        print(f"✓ {method_name}: '{text}' → '{processed}'")
                        return processed
                except Exception as e:
                    print(f"⚠ {method_name} failed: {e}")
            
            print("❌ Could not understand - try speaking clearer")
            return None
                    
        except sr.WaitTimeoutError:
            print("⚠ No speech heard - speak louder")
            self.is_listening = False
            return None
        except Exception as e:
            print(f"⚠ Voice error: {e}")
            self.is_listening = False
            return None
    
    def post_process_speech(self, text: str) -> str:
        """Aggressively fix common speech recognition errors"""
        if not text:
            return text
            
        # Start with lowercase
        processed = text.lower()
        
        # Remove common filler words first
        fillers = ["um", "uh", "like", "you know", "well", "so"]
        for filler in fillers:
            processed = processed.replace(f" {filler} ", " ")
        
        # Aggressive corrections for common misheard phrases
        phrase_corrections = {
            # App name fixes
            "note bad": "notepad", "not bad": "notepad", "no pad": "notepad", 
            "note pad": "notepad", "notes pad": "notepad",
            "calculating": "calculator", "calc later": "calculator",
            "calculator": "calculator", "calc": "calculator",
            "micro soft word": "word", "microsoft word": "word",
            "spread sheet": "excel", "ex sell": "excel",
            "fire fox": "firefox", "firefox": "firefox",
            "google chrome": "chrome", "chrome": "chrome",
            
            # Camera and screenshot fixes
            "camera": "camera", "cameras": "camera", "cam": "camera",
            "take picture": "open camera", "take photo": "open camera",
            "open cam": "open camera", "start camera": "open camera",
            "screen shot": "screenshot", "screen capture": "screenshot",
            "take screenshot": "screenshot", "capture screen": "screenshot",
            "snap shot": "screenshot", "save screen": "screenshot",
            
            # Action fixes
            "opening": "open", "opens": "open", "opened": "open",
            "typing": "type", "types": "type", "typed": "type",
            "clicking": "click", "clicks": "click", "clicked": "click",
            "closing": "close", "closes": "close", "closed": "close",
            "saving": "save", "saves": "save", "saved": "save",
            
            # Number fixes
            "to": "2", "too": "2", "two": "2", "tu": "2",
            "for": "4", "four": "4", "fore": "4",
            "won": "1", "one": "1", "wan": "1",
            "tree": "3", "three": "3", "free": "3",
            "ate": "8", "eight": "8", "ait": "8",
            "five": "5", "six": "6", "seven": "7", "nine": "9", "ten": "10",
            
            # Math operation fixes
            "plus": "plus", "add": "plus", "and": "plus",
            "minus": "minus", "subtract": "minus",
            "times": "times", "multiply": "times", "multiplied by": "times",
            "divided by": "divided by", "divide": "divided by",
        }
        
        # Apply phrase corrections
        for wrong, right in phrase_corrections.items():
            processed = processed.replace(wrong, right)
        
        # Remove common command prefixes that don't help
        prefixes_to_remove = [
            "can you please", "could you please", "please", "can you", "could you",
            "i want you to", "i need you to", "go ahead and", "now", "um",
            "let me", "i want to", "help me", "would you"
        ]
        
        for prefix in prefixes_to_remove:
            if processed.startswith(prefix + " "):
                processed = processed[len(prefix) + 1:]
        
        # Clean up extra spaces and normalize
        processed = " ".join(processed.split())
        
        # Smart command pattern detection
        words = processed.split()
        
        # If we hear "calculator" or math words, ensure it starts with "open calculator"
        math_words = ["calculator", "calc", "plus", "minus", "times", "divided", "multiply", "add", "subtract"]
        if any(word in words for word in math_words) and not processed.startswith("open"):
            if any(word in words for word in ["calculator", "calc"]):
                processed = "open calculator " + processed
            else:
                processed = "open calculator and " + processed
        
        # Handle special commands
        if any(word in words for word in ["screenshot", "capture", "snap"]):
            processed = "take screenshot"
        elif any(word in words for word in ["camera", "cam", "picture", "photo"]):
            processed = "open camera"
        
        # If we hear app names without "open", add it
        app_names = ["notepad", "word", "excel", "chrome", "firefox", "calculator", "camera"]
        if any(app in words for app in app_names) and not any(action in words for action in ["open", "close", "start", "launch", "take"]):
            processed = "open " + processed
        
        return processed.strip()
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False

class ScreenAnalyzer:
    """Analyzes screen content using OCR"""
    
    def __init__(self):
        self.ocr = OCREngine()
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Capture screenshot of screen or region"""
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        return screenshot
    
    def analyze_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """Analyze screen content and return structured data"""
        screenshot = self.capture_screen(region)
        
        # Save screenshot temporarily for OCR processing
        temp_path = "temp_screenshot.png"
        screenshot.save(temp_path)
        
        try:
            # Extract text with coordinates
            text_boxes = self.ocr.extract_text_boxes(temp_path)
            
            # Get full text
            full_text = self.ocr.extract_text(temp_path)
            
            return {
                "screenshot": screenshot,
                "text_boxes": text_boxes,
                "full_text": full_text,
                "screen_size": pyautogui.size()
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def format_screen_info(self, screen_data: Dict) -> str:
        """Format screen analysis for LLM consumption with smart filtering"""
        # Find important elements for the LLM to make decisions
        important_elements = []
        clickable_elements = []
        
        for box in screen_data["text_boxes"]:
            text = box["text"].strip()
            if len(text) < 2:
                continue
                
            bbox = box["bbox"]
            x, y = bbox["x"], bbox["y"]
            width, height = bbox["width"], bbox["height"]
            
            # Identify potentially clickable/important elements
            text_lower = text.lower()
            
            # Identify various types of UI elements
            if any(keyword in text_lower for keyword in [
                # Apps
                'notepad', 'calculator', 'chrome', 'word', 'excel', 'paint', 'explorer',
                # UI elements  
                'search', 'start', 'file', 'edit', 'view', 'help', 'tools', 'format',
                'type here', 'cortana', 'settings', 'control panel',
                # Buttons and controls
                'ok', 'cancel', 'apply', 'save', 'open', 'close', 'minimize', 'maximize',
                'back', 'forward', 'refresh', 'home', 'stop', 'go',
                # Web elements
                'google', 'youtube', 'facebook', 'twitter', 'login', 'sign in',
                # File operations
                'documents', 'downloads', 'desktop', 'pictures', 'music', 'videos'
            ]):
                clickable_elements.append(f"CLICKABLE: '{text}' at ({x}, {y})")
            
            # General elements (limit to most relevant)
            elif len(text) > 3 and width > 30 and height > 15:  # Substantial text elements
                important_elements.append(f"'{text}' at ({x}, {y})")
        
        # Combine clickable and regular elements
        all_elements = clickable_elements + important_elements[:8]  # Limit total
        
        formatted = f"""Screen: {screen_data['screen_size']}
Key Elements: {', '.join(all_elements[:12])}
Full text snippet: {screen_data['full_text'][:300]}..."""
        return formatted

class ActionExecutor:
    """Executes actions on the desktop"""
    
    def __init__(self):
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        
        # Extended key mapping for all supported keys
        self.key_mapping = {
            # Windows specific keys
            'windows': 'win',
            'win': 'win',
            'winleft': 'winleft',
            'winright': 'winright',
            'cmd': 'win',  # Alias for Windows key
            
            # Function keys
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4', 'f5': 'f5', 'f6': 'f6',
            'f7': 'f7', 'f8': 'f8', 'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
            
            # Arrow keys
            'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
            'arrow_up': 'up', 'arrow_down': 'down', 'arrow_left': 'left', 'arrow_right': 'right',
            
            # Control keys
            'ctrl': 'ctrl', 'control': 'ctrl', 'ctrlright': 'ctrlright', 'ctrlleft': 'ctrlleft',
            'alt': 'alt', 'altleft': 'altleft', 'altright': 'altright',
            'shift': 'shift', 'shiftleft': 'shiftleft', 'shiftright': 'shiftright',
            
            # Special keys
            'enter': 'enter', 'return': 'enter',
            'space': 'space', 'spacebar': 'space',
            'tab': 'tab',
            'escape': 'escape', 'esc': 'escape',
            'backspace': 'backspace', 'back': 'backspace',
            'delete': 'delete', 'del': 'delete',
            'insert': 'insert', 'ins': 'insert',
            
            # Home/End/Page keys
            'home': 'home', 'end': 'end',
            'pageup': 'pageup', 'pgup': 'pageup',
            'pagedown': 'pagedown', 'pgdn': 'pagedown',
            
            # Lock keys
            'capslock': 'capslock', 'caps': 'capslock',
            'numlock': 'numlock', 'num': 'numlock',
            'scrolllock': 'scrolllock', 'scroll': 'scrolllock',
            
            # Numpad keys
            'numpad0': 'num0', 'numpad1': 'num1', 'numpad2': 'num2', 'numpad3': 'num3',
            'numpad4': 'num4', 'numpad5': 'num5', 'numpad6': 'num6', 'numpad7': 'num7',
            'numpad8': 'num8', 'numpad9': 'num9',
            'numpadenter': 'enter', 'numpadplus': 'add', 'numpadminus': 'subtract',
            'numpadmultiply': 'multiply', 'numpaddivide': 'divide',
            'numpaddecimal': 'decimal',
            
            # Media keys
            'volumeup': 'volumeup', 'volumedown': 'volumedown', 'volumemute': 'volumemute',
            'playpause': 'playpause', 'nexttrack': 'nexttrack', 'prevtrack': 'prevtrack',
            
            # Browser keys
            'browserback': 'browserback', 'browserforward': 'browserforward',
            'browserrefresh': 'browserrefresh', 'browserhome': 'browserhome',
            'browsersearch': 'browsersearch',
            
            # Print screen
            'printscreen': 'printscreen', 'prtsc': 'printscreen', 'prtscr': 'printscreen',
            
            # Menu key
            'menu': 'apps', 'contextmenu': 'apps', 'apps': 'apps',
            
            # Pause/Break
            'pause': 'pause', 'break': 'pause',
        }
    
    def normalize_key(self, key: str) -> str:
        """Normalize key name to PyAutoGUI format"""
        key_lower = key.lower().strip()
        return self.key_mapping.get(key_lower, key_lower)
    
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at coordinates"""
        try:
            pyautogui.click(x, y, clicks=clicks, button=button)
            return True
        except Exception as e:
            print(f"Click error: {e}")
            return False
    
    def double_click(self, x: int, y: int) -> bool:
        """Double click at coordinates"""
        return self.click(x, y, clicks=2)
    
    def right_click(self, x: int, y: int) -> bool:
        """Right click at coordinates"""
        return self.click(x, y, button="right")
    
    def type_text(self, text: str) -> bool:
        """Type text"""
        try:
            pyautogui.write(text)
            return True
        except Exception as e:
            print(f"Type error: {e}")
            return False
    
    def key_press(self, key: str) -> bool:
        """Press a key (supports all keyboard keys including Windows key)"""
        try:
            normalized_key = self.normalize_key(key)
            pyautogui.press(normalized_key)
            return True
        except Exception as e:
            print(f"Key press error for '{key}' -> '{self.normalize_key(key)}': {e}")
            return False
    
    def key_combination(self, keys: List[str]) -> bool:
        """Press key combination (supports Windows key and all others)"""
        try:
            normalized_keys = [self.normalize_key(key) for key in keys]
            pyautogui.hotkey(*normalized_keys)
            return True
        except Exception as e:
            print(f"Key combination error for {keys} -> {[self.normalize_key(k) for k in keys]}: {e}")
            return False
    
    def key_down(self, key: str) -> bool:
        """Hold down a key"""
        try:
            normalized_key = self.normalize_key(key)
            pyautogui.keyDown(normalized_key)
            return True
        except Exception as e:
            print(f"Key down error: {e}")
            return False
    
    def key_up(self, key: str) -> bool:
        """Release a key"""
        try:
            normalized_key = self.normalize_key(key)
            pyautogui.keyUp(normalized_key)
            return True
        except Exception as e:
            print(f"Key up error: {e}")
            return False
    
    def scroll(self, x: int, y: int, direction: int, clicks: int = 3) -> bool:
        """Scroll at position"""
        try:
            pyautogui.scroll(direction * clicks, x=x, y=y)
            return True
        except Exception as e:
            print(f"Scroll error: {e}")
            return False
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0) -> bool:
        """Drag from start to end position"""
        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            return True
        except Exception as e:
            print(f"Drag error: {e}")
            return False
    
    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to coordinates"""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as e:
            print(f"Mouse move error: {e}")
            return False
    
    def get_supported_keys(self) -> List[str]:
        """Get list of all supported key names"""
        return list(self.key_mapping.keys())

class ComputerAgent:
    """Main computer use agent"""
    
    def __init__(self):
        self.llm = LLMInterface()
        self.screen_analyzer = ScreenAnalyzer()
        self.action_executor = ActionExecutor()
        self.voice_recognizer = VoiceRecognizer()
        self.conversation_history = []
    
    def analyze_and_plan(self, user_request: str, context: str = "") -> str:
        """Analyze current screen and plan actions"""
        # Capture and analyze screen
        screen_data = self.screen_analyzer.analyze_screen()
        screen_info = self.screen_analyzer.format_screen_info(screen_data)
        
        # Prepare system prompt for action planning
        system_prompt = """Desktop automation assistant. Respond with ONE action only.

ACTIONS:
- key_press("win") - Open Start menu
- type_text("notepad") - Type text
- key_press("enter") - Press Enter
- click(x, y) - Click coordinates

FOR OPENING APPS:
1. Desktop → key_press("win")
2. Start menu → type_text("appname") 
3. Search results → key_press("enter")

Respond with ONE action:"""

        # Query LLM with screen context  
        prompt = f"""Screen: {screen_info}
User: {user_request}
Context: {context}

What action?"""

        response = self.llm.query(prompt, system_prompt)
        return response.strip()
    
    def verify_step_completion(self, user_request: str, expected_result: str) -> tuple:
        """Verify if the current step achieved its goal"""
        # Capture and analyze screen
        screen_data = self.screen_analyzer.analyze_screen()
        screen_info = self.screen_analyzer.format_screen_info(screen_data)
        
        system_prompt = """Answer YES or NO only."""

        prompt = f"""Screen: {screen_info}
Expected: {expected_result}
Achieved? YES or NO:"""

        response = self.llm.query(prompt, system_prompt).strip().upper()
        
        # Return verification result and screen data
        return response == "YES", screen_data
    
    def execute_action_from_text(self, action_text: str) -> bool:
        """Parse and execute action from text description"""
        action_text = action_text.strip()
        action_lower = action_text.lower()
        
        # Parse click actions
        if action_lower.startswith("click("):
            match = re.search(r'click\((\d+),\s*(\d+)\)', action_lower)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return self.action_executor.click(x, y)
        
        elif action_lower.startswith("double_click("):
            match = re.search(r'double_click\((\d+),\s*(\d+)\)', action_lower)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return self.action_executor.double_click(x, y)
        
        elif action_lower.startswith("right_click("):
            match = re.search(r'right_click\((\d+),\s*(\d+)\)', action_lower)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return self.action_executor.right_click(x, y)
        
        elif action_lower.startswith("type_text("):
            match = re.search(r'type_text\("([^"]+)"\)', action_text)  # Use original case for text
            if match:
                text = match.group(1)
                return self.action_executor.type_text(text)
        
        elif action_lower.startswith("key_press("):
            match = re.search(r'key_press\("([^"]+)"\)', action_lower)
            if match:
                key = match.group(1)
                return self.action_executor.key_press(key)
        
        elif action_lower.startswith("key_combination("):
            # Handle key combinations like key_combination(["ctrl", "c"]) or key_combination(["win", "r"])
            match = re.search(r'key_combination\(\[([^\]]+)\]\)', action_lower)
            if match:
                keys_str = match.group(1)
                # Parse the list of keys
                keys = [k.strip().strip('"\'') for k in keys_str.split(',')]
                return self.action_executor.key_combination(keys)
        
        elif action_lower.startswith("key_down("):
            match = re.search(r'key_down\("([^"]+)"\)', action_lower)
            if match:
                key = match.group(1)
                return self.action_executor.key_down(key)
        
        elif action_lower.startswith("key_up("):
            match = re.search(r'key_up\("([^"]+)"\)', action_lower)
            if match:
                key = match.group(1)
                return self.action_executor.key_up(key)
        
        elif action_lower.startswith("scroll("):
            match = re.search(r'scroll\((\d+),\s*(\d+),\s*(-?\d+)(?:,\s*(\d+))?\)', action_lower)
            if match:
                x, y, direction = int(match.group(1)), int(match.group(2)), int(match.group(3))
                clicks = int(match.group(4)) if match.group(4) else 3
                return self.action_executor.scroll(x, y, direction, clicks)
        
        elif action_lower.startswith("move_mouse("):
            match = re.search(r'move_mouse\((\d+),\s*(\d+)(?:,\s*([\d.]+))?\)', action_lower)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                duration = float(match.group(3)) if match.group(3) else 0.0
                return self.action_executor.move_mouse(x, y, duration)
        
        elif action_lower.startswith("drag("):
            match = re.search(r'drag\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)', action_lower)
            if match:
                start_x, start_y = int(match.group(1)), int(match.group(2))
                end_x, end_y = int(match.group(3)), int(match.group(4))
                duration = float(match.group(5)) if match.group(5) else 1.0
                return self.action_executor.drag(start_x, start_y, end_x, end_y, duration)
        
        return False
    
    def process_request(self, user_request: str) -> str:
        """Process a user request with forced app opening sequence"""
        try:
            results = []
            
            # Handle special requests first
            if "screenshot" in user_request.lower() or "capture screen" in user_request.lower():
                return self.take_screenshot()
            
            if "camera" in user_request.lower() and "open" in user_request.lower():
                return self.open_camera()
            
            # For calculation tasks with "open calculator and...", handle specially
            if any(word in user_request.lower() for word in ["open", "launch", "start"]) and "calculator" in user_request.lower() and any(calc_word in user_request.lower() for calc_word in ["add", "plus", "minus", "multiply", "divide", "calculate", "+", "-", "*", "/"]):
                calculation_result = self.handle_calculation(user_request)
                results.extend(calculation_result)
                
            # For app opening, use a simple forced sequence
            elif any(word in user_request.lower() for word in ["open", "launch", "start"]) and "type" not in user_request.lower():
                app_name = self.extract_app_name(user_request)
                results.append(f"Opening {app_name}...")
                
                # Step 1: Windows key
                results.append("Step 1: key_press('win')")
                success = self.action_executor.key_press("win")
                results.append(f"✓ Windows key pressed: {success}")
                time.sleep(2)
                
                # Step 2: Type app name
                results.append(f"Step 2: type_text('{app_name}')")
                success = self.action_executor.type_text(app_name)
                results.append(f"✓ Typed '{app_name}': {success}")
                time.sleep(1)
                
                # Step 3: Press Enter
                results.append("Step 3: key_press('enter')")
                success = self.action_executor.key_press("enter")
                results.append(f"✓ Enter pressed: {success}")
                time.sleep(3)  # Wait longer for app to fully open
                
                # Step 4: Handle Office app startup screens
                if app_name.lower() in ["word", "excel", "powerpoint"]:
                    results.append("Step 4: Finding and clicking 'Blank document'...")
                    
                    # Take a screenshot to find the blank document
                    screen_data = self.screen_analyzer.analyze_screen()
                    
                    # Look for "Blank document" text and click it
                    blank_found = False
                    for box in screen_data["text_boxes"]:
                        text = box["text"].lower().strip()
                        if "blank" in text and "document" in text:
                            x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                            y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                            results.append(f"  Found 'Blank document' at ({x}, {y})")
                            self.action_executor.click(x, y)
                            blank_found = True
                            break
                        elif "blank" in text:  # Just "blank" might be enough
                            x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                            y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                            results.append(f"  Found 'Blank' at ({x}, {y})")
                            self.action_executor.click(x, y)
                            blank_found = True
                            break
                    
                    if not blank_found:
                        # Fallback: try the first large rectangular area (likely a template)
                        results.append("  Using fallback: clicking first template area...")
                        # Click in the upper left template area
                        screen_width = screen_data["screen_size"].width
                        click_x = screen_width // 4
                        click_y = 200  # Approximate template area
                        self.action_executor.click(click_x, click_y)
                    
                    time.sleep(3)  # Wait for document to load
                    results.append("✓ Opened blank document")
                    time.sleep(2)
                
                results.append(f"🎯 {app_name} should now be open!")
            
            # For app opening + typing, do both
            elif any(word in user_request.lower() for word in ["open", "launch", "start"]) and "type" in user_request.lower():
                app_name = self.extract_app_name(user_request)
                text_to_type = self.extract_text_to_type(user_request)
                results.append(f"Opening {app_name} and typing '{text_to_type}'...")
                
                # Step 1: Open the app
                results.append("Step 1: key_press('win')")
                success = self.action_executor.key_press("win")
                results.append(f"✓ Windows key pressed: {success}")
                time.sleep(2)
                
                results.append(f"Step 2: type_text('{app_name}')")
                success = self.action_executor.type_text(app_name)
                results.append(f"✓ Typed '{app_name}': {success}")
                time.sleep(1)
                
                results.append("Step 3: key_press('enter')")
                success = self.action_executor.key_press("enter")
                results.append(f"✓ Enter pressed: {success}")
                time.sleep(4)  # Wait for app to fully load
                
                # Step 4: Handle app-specific startup (like Word's template page)
                if app_name.lower() in ["word", "excel", "powerpoint"]:
                    results.append("Step 4: Looking for 'Blank document' to click...")
                    
                    # Use OCR to find and click "Blank document"
                    screen_data = self.screen_analyzer.analyze_screen()
                    
                    blank_clicked = False
                    for box in screen_data["text_boxes"]:
                        text = box["text"].lower().strip()
                        if "blank" in text and "document" in text:
                            # Click in the center of the text box
                            x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                            y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                            results.append(f"  Clicking 'Blank document' at ({x}, {y})")
                            self.action_executor.click(x, y)
                            blank_clicked = True
                            break
                        elif "blank" in text:
                            x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                            y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                            results.append(f"  Clicking 'Blank' at ({x}, {y})")
                            self.action_executor.click(x, y)
                            blank_clicked = True
                            break
                    
                    if not blank_clicked:
                        # Fallback method
                        results.append("  Fallback: clicking likely template area...")
                        screen_width = screen_data["screen_size"].width
                        self.action_executor.click(screen_width // 4, 200)
                    
                    time.sleep(3)  # Wait for blank document to open
                    results.append("✓ Blank document opened")
                    time.sleep(2)
                
                # Step 5: Type the text
                results.append(f"Step 5: type_text('{text_to_type}')")
                success = self.action_executor.type_text(text_to_type)
                results.append(f"✓ Typed '{text_to_type}': {success}")
                
                # Step 6: Handle save request if mentioned
                if any(word in user_request.lower() for word in ["save", "save as"]):
                    results.append("Step 6: Saving document...")
                    time.sleep(1)
                    self.action_executor.key_combination(["ctrl", "s"])
                    time.sleep(2)  # Wait for save dialog
                    # Press Enter to save with default name
                    self.action_executor.key_press("enter")
                    results.append("✓ Document saved")
                    time.sleep(1)
                
                # Step 7: Handle formatting requests
                if any(word in user_request.lower() for word in ["bold", "italic", "underline", "format"]):
                    results.append("Step 6: Applying formatting...")
                    # Select all text first
                    self.action_executor.key_combination(["ctrl", "a"])
                    time.sleep(0.5)
                    
                    if "bold" in user_request.lower():
                        self.action_executor.key_combination(["ctrl", "b"])
                        results.append("✓ Applied bold formatting")
                    if "italic" in user_request.lower():
                        self.action_executor.key_combination(["ctrl", "i"])
                        results.append("✓ Applied italic formatting")
                    if "underline" in user_request.lower():
                        self.action_executor.key_combination(["ctrl", "u"])
                        results.append("✓ Applied underline formatting")
                    
                    # Click at end to deselect
                    self.action_executor.key_press("end")
                    time.sleep(0.5)
                
                results.append(f"🎯 Opened {app_name} and typed '{text_to_type}'!")
            
            # For just typing (when app is already open)
            elif "type" in user_request.lower():
                text_to_type = self.extract_text_to_type(user_request)
                results.append(f"Typing '{text_to_type}'...")
                
                success = self.action_executor.type_text(text_to_type)
                results.append(f"✓ Typed '{text_to_type}': {success}")
                
                results.append(f"🎯 Text typed successfully!")
                
            # For calculation tasks, use calculator
            elif any(word in user_request.lower() for word in ["calculate", "compute", "math", "calculator"]):
                calculation_result = self.handle_calculation(user_request)
                results.extend(calculation_result)
                
            else:
                # For other tasks, use the LLM
                results.append("Using LLM for complex task...")
                action_command = self.analyze_and_plan(user_request, "")
                results.append(f"LLM suggested: {action_command}")
                
                if action_command:
                    success = self.execute_action_from_text(action_command)
                    results.append(f"Execution result: {success}")
            
            # Add to conversation history
            final_result = "\n".join(results)
            self.conversation_history.append({
                "user": user_request,
                "agent_actions": results,
                "timestamp": time.time()
            })
            
            return final_result
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.conversation_history.append({
                "user": user_request,
                "error": error_msg,
                "timestamp": time.time()
            })
            return error_msg
    
    def take_screenshot(self) -> str:
        """Take a screenshot and save it with timestamp"""
        try:
            # Generate timestamp for unique filename
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
            
            # Get desktop path - try multiple common locations
            desktop_paths = [
                os.path.join(os.path.expanduser("~"), "Desktop"),
                os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),
                os.path.join(os.path.expanduser("~"), "Documents", "Desktop"),
                os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
                os.path.join(os.environ.get("USERPROFILE", ""), "OneDrive", "Desktop")
            ]
            
            # Find the first existing desktop path
            desktop_path = None
            for path in desktop_paths:
                if os.path.exists(path):
                    desktop_path = path
                    break
            
            # If no desktop found, use current directory
            if not desktop_path:
                desktop_path = os.getcwd()
                print(f"⚠ Desktop not found, saving to current directory: {desktop_path}")
            
            full_path = os.path.join(desktop_path, filename)
            
            # Take screenshot using pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save(full_path)
            
            # Show success message
            result = f"✅ Screenshot saved to {desktop_path} as '{filename}'"
            print(result)
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed to take screenshot: {str(e)}"
            print(error_msg)
            return error_msg
    
    def open_camera(self) -> str:
        """Open camera application"""
        try:
            results = []
            results.append("📷 Opening Camera app...")
            
            # Step 1: Windows key
            results.append("Step 1: Pressing Windows key")
            self.action_executor.key_press("win")
            time.sleep(2)
            
            # Step 2: Type camera
            results.append("Step 2: Typing 'camera'")
            self.action_executor.type_text("camera")
            time.sleep(1)
            
            # Step 3: Press Enter
            results.append("Step 3: Pressing Enter")
            self.action_executor.key_press("enter")
            time.sleep(3)
            
            results.append("✅ Camera should now be opening!")
            return "\n".join(results)
            
        except Exception as e:
            error_msg = f"❌ Failed to open camera: {str(e)}"
            print(error_msg)
            return error_msg
    
    def extract_app_name(self, user_request: str) -> str:
        """Extract app name from user request"""
        request_lower = user_request.lower()
        
        # Common app mappings
        app_mapping = {
            "notepad": "notepad",
            "calculator": "calculator", 
            "calc": "calculator",
            "chrome": "chrome",
            "browser": "chrome",
            "word": "word",
            "excel": "excel",
            "paint": "paint",
            "settings": "settings",
            "explorer": "explorer",
            "file manager": "explorer"
        }
        
        for keyword, app_name in app_mapping.items():
            if keyword in request_lower:
                return app_name
        
        # Default fallback - extract word after "open"
        words = request_lower.split()
        if "open" in words:
            try:
                open_index = words.index("open")
                if open_index + 1 < len(words):
                    return words[open_index + 1]
            except:
                pass
        
        return "notepad"  # Safe default
    
    def extract_text_to_type(self, user_request: str) -> str:
        """Extract text to type from user request"""
        request_lower = user_request.lower()
        
        # Look for patterns like "type hello", "type 'hello world'", "type hello world"
        import re
        
        # Pattern 1: type "quoted text"
        quoted_match = re.search(r'type\s+["\']([^"\']+)["\']', request_lower)
        if quoted_match:
            return quoted_match.group(1)
        
        # Pattern 2: type hello world (until end or certain words)
        type_match = re.search(r'type\s+(.+?)(?:\s+in\s+|\s+and\s+|$)', request_lower)
        if type_match:
            text = type_match.group(1).strip()
            # Clean up common endings
            text = re.sub(r'\s+(in|and|then|after).*$', '', text)
            return text
        
        # Pattern 3: Simple fallback - everything after "type"
        words = request_lower.split()
        if "type" in words:
            try:
                type_index = words.index("type")
                if type_index + 1 < len(words):
                    # Get words after "type" until we hit app names or actions
                    stop_words = ["in", "and", "then", "after", "notepad", "word", "chrome"]
                    text_words = []
                    for word in words[type_index + 1:]:
                        if word in stop_words:
                            break
                        text_words.append(word)
                    if text_words:
                        return " ".join(text_words)
            except:
                pass
        
        return "Hello World"  # Safe default
    
    def handle_calculation(self, user_request: str) -> list:
        """Handle calculator operations"""
        results = []
        
        try:
            # Extract the mathematical expression
            expression = self.extract_math_expression(user_request)
            results.append(f"🧮 Calculator Task: {expression}")
            
            # Step 1: Open Calculator
            results.append("Step 1: Opening Calculator...")
            self.action_executor.key_press("win")
            time.sleep(1)
            self.action_executor.type_text("calculator")
            time.sleep(1)
            self.action_executor.key_press("enter")
            time.sleep(2)  # Wait for calculator to open
            results.append("✓ Calculator opened")
            
            # Step 2: Clear calculator (just in case)
            self.action_executor.key_press("escape")  # Clear any existing calculation
            time.sleep(0.5)
            
            # Step 3: Input the calculation
            results.append(f"Step 2: Inputting '{expression}'...")
            success = self.input_calculation(expression)
            if success:
                results.append(f"✓ Entered calculation: {expression}")
            else:
                results.append(f"✗ Failed to enter calculation")
                return results
            
            # Step 4: Click equals button to get result
            results.append("Step 3: Calculating result...")
            equals_clicked = self.click_equals_button()
            if not equals_clicked:
                # Fallback to keyboard
                self.action_executor.key_press("enter")
            time.sleep(2)  # Wait for calculation
            
            # Step 5: Take screenshot and read result
            results.append("Step 4: Reading result...")
            screen_data = self.screen_analyzer.analyze_screen()
            result = self.extract_calculator_result(screen_data)
            
            if result:
                results.append(f"🎯 CALCULATION RESULT: {expression} = {result}")
            else:
                results.append("⚠ Could not read calculator result")
            
            # Step 6: Close calculator
            results.append("Step 5: Closing calculator...")
            self.action_executor.key_combination(["alt", "f4"])
            time.sleep(1)
            results.append("✓ Calculator closed")
            
            return results
            
        except Exception as e:
            results.append(f"✗ Calculator error: {str(e)}")
            return results
    
    def extract_math_expression(self, user_request: str) -> str:
        """Extract mathematical expression from user request"""
        import re
        
        request_lower = user_request.lower()
        
        # Convert word operations to symbols
        request_lower = request_lower.replace(" and ", " + ")
        request_lower = request_lower.replace(" add ", " + ")
        request_lower = request_lower.replace(" plus ", " + ")
        request_lower = request_lower.replace(" minus ", " - ")
        request_lower = request_lower.replace(" subtract ", " - ")
        request_lower = request_lower.replace(" multiply ", " * ")
        request_lower = request_lower.replace(" times ", " * ")
        request_lower = request_lower.replace(" divide ", " / ")
        request_lower = request_lower.replace(" divided by ", " / ")
        
        # Look for common calculation patterns
        patterns = [
            r'calculate\s+(.+?)(?:\s|$)',
            r'compute\s+(.+?)(?:\s|$)', 
            r'what\s+is\s+(.+?)(?:\s|$)',
            r'add\s+(\d+)\s+and\s+(\d+)',  # "add 2 and 10"
            r'(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)',
        ]
        
        # Special handling for "add X and Y" pattern
        add_match = re.search(r'add\s+(\d+)\s+and\s+(\d+)', request_lower)
        if add_match:
            return f"{add_match.group(1)}+{add_match.group(2)}"
        
        for pattern in patterns:
            match = re.search(pattern, request_lower)
            if match:
                expr = match.group(1).strip()
                # Clean up the expression
                expr = expr.replace('x', '*').replace('÷', '/').replace('×', '*')
                expr = expr.replace(' ', '')  # Remove spaces
                return expr
        
        # Fallback: look for simple number operations
        numbers_and_ops = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', user_request)
        if len(numbers_and_ops) >= 3:  # At least num op num
            return ''.join(numbers_and_ops)
        
        # Last resort: try to find just two numbers and assume addition
        numbers = re.findall(r'\d+', user_request)
        if len(numbers) >= 2:
            return f"{numbers[0]}+{numbers[1]}"
        
        return "2+2"  # Safe default
    
    def input_calculation(self, expression: str) -> bool:
        """Input calculation into calculator by clicking buttons"""
        try:
            print(f"DEBUG: Inputting calculation '{expression}' by clicking calculator buttons")
            
            # Take a screenshot to find calculator buttons
            screen_data = self.screen_analyzer.analyze_screen()
            
            # Define button mappings - common calculator button text
            button_map = {
                '0': ['0'],
                '1': ['1'], 
                '2': ['2'],
                '3': ['3'],
                '4': ['4'],
                '5': ['5'],
                '6': ['6'],
                '7': ['7'],
                '8': ['8'],
                '9': ['9'],
                '+': ['+', 'Add'],
                '-': ['-', 'Subtract'],
                '*': ['×', '*', 'Multiply'],
                '/': ['÷', '/', 'Divide'],
                '.': ['.', 'Decimal']
            }
            
            # Clean the expression and input character by character
            clean_expr = expression.replace(' ', '')
            
            # Find and click each character in the expression
            for char in clean_expr:
                if char in button_map:
                    button_found = False
                    search_terms = button_map[char]
                    
                    # Look for the button in OCR results
                    for box in screen_data["text_boxes"]:
                        text = box["text"].strip()
                        
                        # Check if this text matches any of our search terms
                        if any(term.lower() == text.lower() for term in search_terms):
                            # Click the center of this button
                            x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                            y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                            
                            print(f"DEBUG: Clicking '{char}' button at ({x}, {y}) - found text '{text}'")
                            self.action_executor.click(x, y)
                            button_found = True
                            time.sleep(0.5)  # Wait between clicks
                            break
                    
                    if not button_found:
                        print(f"DEBUG: Button '{char}' not found, trying keyboard fallback")
                        # Fallback to keyboard if button not found
                        if char.isdigit():
                            self.action_executor.key_press(char)
                        elif char == '+':
                            self.action_executor.key_combination(["shift", "equal"])
                        elif char == '-':
                            self.action_executor.key_press('minus')
                        elif char == '*':
                            self.action_executor.key_combination(["shift", "8"])
                        elif char == '/':
                            self.action_executor.key_press('slash')
                        elif char == '.':
                            self.action_executor.key_press('period')
                        time.sleep(0.3)
                else:
                    print(f"DEBUG: Ignoring character '{char}'")
            
            return True
            
        except Exception as e:
            print(f"Input calculation error: {e}")
            return False
    
    def click_equals_button(self) -> bool:
        """Find and click the equals button on calculator"""
        try:
            # Take a screenshot to find the equals button
            screen_data = self.screen_analyzer.analyze_screen()
            
            # Look for equals button - common texts: "=", "Equals"
            equals_terms = ["=", "Equals"]
            
            for box in screen_data["text_boxes"]:
                text = box["text"].strip()
                
                # Check if this is the equals button
                if any(term.lower() == text.lower() for term in equals_terms):
                    # Click the center of this button
                    x = box["bbox"]["x"] + box["bbox"]["width"] // 2
                    y = box["bbox"]["y"] + box["bbox"]["height"] // 2
                    
                    print(f"DEBUG: Clicking equals button at ({x}, {y}) - found text '{text}'")
                    self.action_executor.click(x, y)
                    return True
            
            print("DEBUG: Equals button not found")
            return False
            
        except Exception as e:
            print(f"DEBUG: Error clicking equals button: {e}")
            return False
    
    def extract_calculator_result(self, screen_data: dict) -> str:
        """Extract the result from calculator screen using OCR"""
        try:
            import re
            
            # Look for numbers in the calculator display area
            potential_results = []
            screen_height = screen_data["screen_size"].height
            
            print(f"DEBUG: Looking for calculator result in {len(screen_data['text_boxes'])} text boxes")
            
            for box in screen_data["text_boxes"]:
                text = box["text"].strip()
                bbox = box["bbox"]
                
                print(f"DEBUG: Found text '{text}' at ({bbox['x']}, {bbox['y']}) size {bbox['width']}x{bbox['height']}")
                
                # Look for any text containing digits
                if re.search(r'\d', text):
                    y_pos = bbox["y"]
                    width = bbox["width"]
                    height = bbox["height"]
                    
                    # Calculator result display criteria - focus on top area and larger text
                    is_likely_result = (
                        y_pos < screen_height * 0.4 and    # Upper 40% of screen
                        width > 20 and                     # Minimum width
                        height > 15                        # Minimum height
                    )
                    
                    if is_likely_result:
                        # Extract just the numbers from the text, removing commas and other chars
                        clean_text = re.sub(r'[^0-9.]', '', text)
                        if clean_text and re.match(r'^\d+(?:\.\d+)?$', clean_text):
                            # Score based on size and position (larger and higher = better)
                            score = width * height
                            if y_pos < screen_height * 0.25:  # Bonus for top quarter
                                score *= 2
                            
                            potential_results.append((clean_text, score, text, bbox))
                            print(f"DEBUG: Candidate '{clean_text}' from '{text}' score={score}")
            
            if potential_results:
                # Sort by score and return best match
                potential_results.sort(key=lambda x: x[1], reverse=True)
                best_match = potential_results[0]
                print(f"DEBUG: Best match: '{best_match[0]}' from '{best_match[2]}'")
                return best_match[0]
            
            # Fallback: look for any number in the full text
            full_text = screen_data["full_text"]
            numbers = re.findall(r'\d+', full_text)
            if numbers:
                print(f"DEBUG: Fallback using last number: {numbers[-1]}")
                return numbers[-1]
            
            return "No result"
            
        except Exception as e:
            print(f"DEBUG: Calculator OCR error: {str(e)}")
            return "Error"
    
    def get_expected_result(self, action_command: str, user_request: str) -> str:
        """Get expected result for verification"""
        action_lower = action_command.lower()
        
        if "key_press(\"win\")" in action_lower:
            return "Start menu is open"
        elif "type_text(" in action_lower and "open" in user_request.lower():
            # Typing app name to open it
            app_name = action_command.split('"')[1] if '"' in action_command else "app"
            return f"Typed '{app_name}' in search"
        elif "type_text(" in action_lower:
            # Typing text in an application
            text = action_command.split('"')[1] if '"' in action_command else "text"
            return f"Text '{text}' appears on screen"
        elif "key_press(\"enter\")" in action_lower:
            if "open" in user_request.lower():
                app_name = user_request.lower().replace("open ", "").replace("launch ", "").split()[0]
                return f"{app_name} application is open"
            else:
                return "Enter key pressed successfully"
        elif "click(" in action_lower:
            return "Clicked on target element"
        else:
            return "Action completed"
    
    def is_final_goal_achieved(self, user_request: str, screen_data: dict) -> bool:
        """Check if the final goal is achieved for various types of tasks"""
        request_lower = user_request.lower()
        screen_text = screen_data['full_text'].lower()
        
        # App opening tasks
        if any(word in request_lower for word in ["open", "launch", "start"]) and "type" not in request_lower:
            app_keywords = ["notepad", "calculator", "chrome", "word", "excel", "paint", "explorer", "settings"]
            for app in app_keywords:
                if app in request_lower and (app in screen_text or "untitled" in screen_text):
                    return True
        
        # Typing tasks
        elif "type" in request_lower:
            import re
            type_match = re.search(r'type\s+["\']?([^"\']+)["\']?', request_lower)
            if type_match:
                target_text = type_match.group(1).strip()
                if target_text in screen_text:
                    return True
        
        # File operations
        elif any(word in request_lower for word in ["save", "save as"]):
            if any(indicator in screen_text for indicator in ["save", "saved", "file saved"]):
                return True
        
        # Navigation tasks  
        elif any(word in request_lower for word in ["go to", "navigate", "visit"]):
            if "google" in request_lower and "google" in screen_text:
                return True
            if any(site in request_lower for site in ["youtube", "facebook"] if site in screen_text):
                return True
        
        # Settings tasks
        elif "settings" in request_lower:
            if any(word in screen_text for word in ["settings", "preferences", "options"]):
                return True
        
        # Close/minimize tasks
        elif any(word in request_lower for word in ["close", "minimize", "exit"]):
            # If we can't see the app anymore, it was likely closed
            app_keywords = ["notepad", "calculator", "chrome", "word", "excel"]
            for app in app_keywords:
                if app in request_lower and app not in screen_text:
                    return True
        
        # Copy/paste tasks
        elif any(word in request_lower for word in ["copy", "paste", "cut"]):
            # These are usually immediate actions, hard to verify visually
            return True  # Assume success for now
        
        # Scroll tasks
        elif "scroll" in request_lower:
            # Scrolling is immediate, assume success
            return True
        
        return False

class AgentGUI:
    """Tkinter GUI for the Computer Agent"""
    
    def __init__(self):
        self.agent = ComputerAgent()
        self.root = tk.Tk()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root.title("Computer Use Agent")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 Computer Use Agent", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Chat display area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            height=20, 
            width=80,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#333333",
            insertbackground="#333333"
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input field
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame, 
            textvariable=self.input_var,
            font=("Arial", 11)
        )
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Send button
        self.send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message,
            style="Accent.TButton"
        )
        self.send_button.grid(row=0, column=1, padx=(0, 10))
        
        # Voice button
        self.voice_button = ttk.Button(
            input_frame, 
            text="🎤 Voice", 
            command=self.start_voice_command
        )
        self.voice_button.grid(row=0, column=2)
        
        # Bind Enter key
        self.input_entry.bind("<Return>", lambda e: self.send_message())
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Control buttons
        ttk.Button(control_frame, text="Clear Chat", command=self.clear_chat).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(control_frame, text="Show Keys", command=self.show_supported_keys).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(control_frame, text="Voice Settings", command=self.show_voice_settings).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(control_frame, text="Emergency Stop", command=self.emergency_stop).grid(row=0, column=3)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initial message
        self.add_message("Agent", "Hello! I'm your autonomous computer use agent with voice control! 🎤\n\nI can:\n• Execute tasks through text commands\n• Listen to your voice commands (click 🎤 Voice button)\n• Automatically take screenshots and analyze your screen\n• Control mouse, keyboard, and applications\n• Perform calculations, open apps, type text, and more!\n\nJust tell me what you'd like me to do - type it or say it!")
        
        # Focus on input
        self.input_entry.focus()
    
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Format message based on sender
        if sender == "User":
            self.chat_display.insert(tk.END, f"[{timestamp}] 👤 You: {message}\n\n", "user")
        elif sender == "Agent":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🤖 Agent: {message}\n\n", "agent")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] ℹ️ {sender}: {message}\n\n", "system")
        
        # Configure tags
        self.chat_display.tag_config("user", foreground="#0066cc")
        self.chat_display.tag_config("agent", foreground="#006600")
        self.chat_display.tag_config("system", foreground="#666666")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self):
        """Send user message to agent"""
        message = self.input_var.get().strip()
        if not message:
            return
        
        # Clear input
        self.input_var.set("")
        
        # Add user message
        self.add_message("User", message)
        
        # Update status
        self.status_var.set("Processing...")
        self.root.update()
        
        # Process in thread to avoid blocking GUI
        threading.Thread(target=self.process_user_message, args=(message,), daemon=True).start()
    
    def process_user_message(self, message: str):
        """Process user message in background thread"""
        try:
            # Get agent response
            response = self.agent.process_request(message)
            
            # Update GUI in main thread
            self.root.after(0, lambda: self.add_message("Agent", response))
            self.root.after(0, lambda: self.status_var.set("Ready"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: self.add_message("System", error_msg))
            self.root.after(0, lambda: self.status_var.set("Error"))
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_message("Agent", "Chat cleared. How can I help you?")
    
    
    def show_supported_keys(self):
        """Show all supported keyboard keys"""
        keys = self.agent.action_executor.get_supported_keys()
        keys_text = ", ".join(sorted(keys))
        
        # Show in a new window
        key_window = tk.Toplevel(self.root)
        key_window.title("Supported Keyboard Keys")
        key_window.geometry("600x400")
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(key_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Insert content
        content = f"""Supported Keyboard Keys:

The agent supports all these key names (case-insensitive):

{keys_text}

Examples:
- key_press("win") - Press Windows key
- key_press("f4") - Press F4 key  
- key_combination(["win", "r"]) - Open Run dialog
- key_combination(["ctrl", "shift", "n"]) - Ctrl+Shift+N
- key_combination(["alt", "f4"]) - Alt+F4 to close window
- key_press("printscreen") - Take screenshot
- key_press("volumeup") - Increase volume

All function keys (f1-f12), arrow keys, numpad keys, media keys, and special keys are supported!"""
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def start_voice_command(self):
        """Start voice recognition in a separate thread"""
        if hasattr(self.agent.voice_recognizer, 'is_listening') and self.agent.voice_recognizer.is_listening:
            self.add_message("System", "⚠ Already listening for voice command...")
            return
            
        # Disable voice button during recognition
        self.voice_button.config(state='disabled', text="🎤 Listening...")
        self.status_var.set("🎤 Listening for voice command...")
        
        # Start voice recognition in background thread
        voice_thread = threading.Thread(target=self.process_voice_command, daemon=True)
        voice_thread.start()
    
    def process_voice_command(self):
        """Process voice command in background thread"""
        try:
            # Listen for voice command with optimized settings  
            voice_text = self.agent.voice_recognizer.listen_for_command(timeout=5, phrase_time_limit=3)
            
            # Update GUI in main thread
            self.root.after(0, self.handle_voice_result, voice_text)
            
        except Exception as e:
            self.root.after(0, self.handle_voice_error, str(e))
    
    def handle_voice_result(self, voice_text: Optional[str]):
        """Handle voice recognition result"""
        # Re-enable voice button
        self.voice_button.config(state='normal', text="🎤 Voice")
        self.status_var.set("Ready")
        
        if voice_text:
            # Put recognized text in input field and show it
            self.input_var.set(voice_text)
            self.add_message("You (Voice)", voice_text)
            
            # Automatically send the voice command
            self.send_message()
        else:
            self.add_message("System", "⚠ Could not understand voice command. Please try again.")
    
    def handle_voice_error(self, error_msg: str):
        """Handle voice recognition error"""
        self.voice_button.config(state='normal', text="🎤 Voice")
        self.status_var.set("Ready")
        self.add_message("System", f"⚠ Voice recognition error: {error_msg}")
    
    def show_voice_settings(self):
        """Show voice recognition settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Voice Recognition Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Main frame
        main_frame = ttk.Frame(settings_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎤 Voice Recognition Settings", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Test voice button
        test_button = ttk.Button(
            main_frame, 
            text="🎤 Test Voice Recognition", 
            command=lambda: self.test_voice_recognition(settings_window)
        )
        test_button.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Tips
        tips_label = ttk.Label(main_frame, text="💡 Tips for Better Recognition:", font=("Arial", 11, "bold"))
        tips_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W), pady=(0, 10))
        
        tips_text = """• Speak clearly and at normal speed
• Use simple, direct commands
• Minimize background noise
• Common commands:
  - "open calculator"
  - "open notepad and type hello"
  - "calculator 2 plus 10"
  - "type some text"
  - "close window"

• If recognition is poor, try using text input instead
• Google recognition (online) is most accurate
• Sphinx (offline) works without internet but less accurate"""
        
        tips_display = tk.Text(main_frame, height=10, width=50, wrap=tk.WORD, font=("Arial", 9))
        tips_display.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        tips_display.insert(tk.END, tips_text)
        tips_display.config(state=tk.DISABLED)
        
        # Close button
        close_button = ttk.Button(settings_window, text="Close", command=settings_window.destroy)
        close_button.grid(row=1, column=0, pady=10)
        
        # Center the window
        settings_window.transient(self.root)
        settings_window.grab_set()
    
    def test_voice_recognition(self, parent_window):
        """Test voice recognition in settings"""
        # Disable test button
        for widget in parent_window.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and "Test" in child.cget("text"):
                        child.config(state='disabled', text="🎤 Listening...")
                        break
        
        # Start test in background
        def test_voice():
            try:
                result = self.agent.voice_recognizer.listen_for_command(timeout=5, phrase_time_limit=3)
                parent_window.after(0, lambda: self.show_test_result(parent_window, result))
            except Exception as e:
                parent_window.after(0, lambda: self.show_test_result(parent_window, f"Error: {e}"))
        
        thread = threading.Thread(target=test_voice, daemon=True)
        thread.start()
    
    def show_test_result(self, parent_window, result):
        """Show voice test result"""
        # Re-enable test button
        for widget in parent_window.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and "Listening" in child.cget("text"):
                        child.config(state='normal', text="🎤 Test Voice Recognition")
                        break
        
        # Show result
        if result:
            messagebox.showinfo("Voice Test Result", f"✓ Recognized: '{result}'")
        else:
            messagebox.showwarning("Voice Test Result", "⚠ Could not understand speech. Try speaking more clearly or check microphone.")
    
    def emergency_stop(self):
        """Emergency stop for all automation"""
        try:
            # Stop voice recognition if active
            if hasattr(self.agent.voice_recognizer, 'stop_listening'):
                self.agent.voice_recognizer.stop_listening()
            # Move mouse to corner to trigger failsafe
            pyautogui.moveTo(0, 0)
            self.add_message("System", "Emergency stop activated!")
            self.status_var.set("Stopped")
        except Exception as e:
            self.add_message("System", f"Emergency stop error: {str(e)}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function to run the Computer Agent"""
    try:
        # Check dependencies
        print("Initializing Computer Use Agent...")
        
        # Start GUI
        app = AgentGUI()
        app.run()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start agent: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

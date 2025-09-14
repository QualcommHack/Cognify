Here's a shortened version of your README for easy copying to GitHub:

---

# Computer Use Agent

An autonomous desktop automation agent that combines LLM reasoning, OCR screen analysis, and PyAutoGUI to perform tasks through natural language commands.

## Features

* **Natural Language Interface**: Chat-based commands in plain English
* **Screen Analysis**: Uses OCR (Tesseract) to read screen content
* **LLM Reasoning**: Powered by Genie LLM for intelligent decision making
* **Desktop Automation**: Click, type, scroll, keyboard shortcuts
* **Real-time Feedback**: Chat interface shows agent thoughts

## Components

1. **LLM Interface**: Connects to Genie LLM bundle
2. **OCR Engine**: Uses Tesseract for text extraction
3. **Action Executor**: Controls mouse, keyboard, windows via PyAutoGUI
4. **GUI Interface**: Tkinter chat interface for interaction

## Requirements

### Software Dependencies

* Python 3.8+
* Tesseract OCR
* Genie LLM bundle (`genie_bundle/`)

### Python Packages

* pyautogui, pygetwindow
* Pillow, pytesseract
* opencv-python, numpy

## Setup

1. **Install Dependencies**:

   ```bash
   python setup_agent.py
   ```

2. **Install Tesseract OCR**:

   * [Download](https://github.com/UB-Mannheim/tesseract/wiki) or use `winget install UB-Mannheim.TesseractOCR`.

3. **Verify Setup**:

   ```bash
   python setup_agent.py
   ```

## Usage

1. **Start the Agent**:

   ```bash
   python computer_agent.py
   ```

2. **Give Commands**:

   * "Open Notepad and type 'Hello World'"
   * "Click the File menu and select Save As"
   * "Take a screenshot and tell me what you see"

## Available Actions

* **Mouse Actions**:

  * `click(x, y)`, `double_click(x, y)`, `drag(start_x, start_y, end_x, end_y)`

* **Keyboard Actions**:

  * `type_text("text")`, `key_press("key")`, `key_combination(["key1", "key2"])`

* **Windows Key**:

  * `key_press("win")`, `key_combination(["win", "r"])`

* **Scrolling**:

  * `scroll(x, y, direction, clicks)`

## Safety Features

* **Failsafe**: Mouse to top-left corner to stop automation
* **Emergency Stop**: Button in GUI to halt actions
* **Pause Control**: Adjustable delays between actions

## How It Works

1. **Screen Capture**: Takes screenshot of current screen
2. **OCR Analysis**: Extracts visible text with coordinates
3. **LLM Processing**: Sends screen context and user request to LLM
4. **Action Planning**: LLM suggests actions
5. **Execution**: PyAutoGUI carries out actions
6. **Feedback**: Results shown in chat interface

## Example Workflow

```
User: "Open Calculator and calculate 15 + 27"

Agent Analysis:
- Captures screen
- Finds Windows Start button at (10, 740)
- Plans: Click Start → Type "calculator" → Press Enter

Agent Actions:
1. click(10, 740)  # Click Start button
2. type_text("calculator")  # Search for calculator
3. key_press("enter")  # Open calculator
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure it's installed and in PATH.
2. **LLM not responding**: Verify `genie_bundle/` exists, and `genie-t2t-run.exe` is executable.
3. **Permission errors**: Run as admin if needed.
4. **OCR issues**: Ensure good contrast and check display settings.

### Performance Tips

* Close unnecessary apps to improve screenshot speed.
* Use specific regions for analysis when possible.
* Adjust PyAutoGUI pause settings for speed.

## Directory Structure

```
vettri/
├── computer_agent.py      # Main agent app
├── setup_agent.py         # Setup and dependency checker
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── genie_bundle/         # LLM model
└── ocr_code/             # OCR engine
    ├── ocr_engine.py
    └── simple_ocr.py
```

## Contributing

Feel free to extend the agent with additional capabilities, such as web browser automation, voice command support, multi-monitor support, etc.

## License

MIT License.

---

This version captures all critical details in a concise format, perfect for easy copying to GitHub! Let me know if you need more changes.



# Revoxx - Record Voices

This repository provides **Revoxx**, a graphical recording application for recording raw speech and generating datasets.

[![PyPI version](https://img.shields.io/pypi/v/revoxx)](https://pypi.org/project/revoxx/)
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
[![CI Status](https://github.com/icelandic-lt/revoxx/actions/workflows/build.yml/badge.svg)](https://github.com/icelandic-lt/revoxx/actions/workflows/build.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/icelandic-lt/revoxx)

## Overview

**Revoxx** has been created by [Grammatek ehf](https://www.grammatek.com) and is part of the [Icelandic Language Technology Programme](https://github.com/icelandic-lt/icelandic-lt).

- **Category:** [TTS](https://github.com/icelandic-lt/icelandic-lt/blob/main/doc/tts.md)
- **Domain:** Laptop/Workstation
- **Languages:** Python
- **Language Version/Dialect:**
  - Python: 3.10 - 3.13
- **Audience**: Developers, Researchers
- **Origins:** [Icelandic EmoSpeech scripts](https://github.com/icelandic-lt/emospeech-scripts)

## Status
![Production](https://img.shields.io/badge/Production-darkgreen)

## System Requirements
- **Operating System:** Linux/OS-X, should work on Windows
- **Python:** 3.10 - 3.13 with Tkinter support
- **Recording:** Audio Interface, good voice microphone and headphones
- **Linux:** Requires PortAudio library (`sudo apt-get install portaudio19-dev` on Ubuntu/Debian)
- **GUI:** Tkinter (usually included with Python, see installation notes below)

## Description

**Revoxx** is a graphical speech recorder specialized in recording TTS datasets quickly and reliably.<br>
You can use this project to create emotional / non-emotional voice recordings on a Workstation / Laptop with suitable audio equipment.
It has integrated support to easily transform raw recordings into datasets for training TTS voice models.<br>
This tool is especially useful for recording many short utterances - up to an utterance duration of approx. 30-45 secs each.
For longer texts, you need to split your input texts in appropriately sized chunks that would fit on the speaker screen.
<br>
**Revoxx** has been inspired by [Icelandic EmoSpeech scripts](https://github.com/icelandic-lt/emospeech-scripts), but has been vastly improved and is rewritten from scratch.<br>

**Screenshot:**

<img src="https://raw.githubusercontent.com/icelandic-lt/revoxx/main/doc/screenshot1.png" alt="screenshot1" width="100%"/>

We have condensed our experience from when we recorded [Talrómur 3](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/344),
the Icelandic emotional speech dataset, and created this tool to minimize hassle, valuable recording & post-processing time.

- **Revoxx** makes recording of speech **fast, reliable and convenient for the recording engineer and the voice talent**
  - Integrates all necessary tools to check if recordings & equipment meet your expected requirements
  - Automatically analyzes and validates audio equipment compatibility, including Sample Rate, Bit Depth, and I/O
    channel configurations
  - Supports unlimited re-recording while maintaining a complete **archive of raw recordings**, even for deleted content
  - Text size is automatically adjusted according to available screen real-estate
  - **Intuitive keyboard shortcuts** for accessing core functionalities
- Recordings are organized into **Recording Sessions**
  - Record emotional sessions for each speaker or record more traditional LJSpeech-style sessions
  - Seamless transitions between different recording sessions with automatic progress tracking: continue where you left-off
  - **Flag utterances** for re-recording or exclusion from export
  - Offers advanced search and navigation capabilities for utterances, with flexible sorting and ordering by label, emotion, text
    content, text length and recorded takes
  - Consistent audio settings & metadata for all recordings
- **Real-time monitoring** including toggable recording levels, mel spectrograms, maximum frequency detection, and more
  - Customizable **industry-standard presets for Peak/RMS levels**
  - Dedicated **Monitoring mode** for precise input calibration
- **Audio Editing** directly in the spectrogram view
  - Set position markers to play from any point in the recording
  - Create selection ranges for partial playback
  - Delete ranges with automatic crossfade
  - Insert new audio at marker position
  - Replace selected ranges with new recordings
  - Multiple undo/redo for all editing operations
- **Multi-Screen Support**
  - You can use multiple monitors to **separate recording view from speaker view**
  - We support Apple's [Sidecar](https://support.apple.com/en-us/102597) feature for a **convenient dual screen setup with an external iPad**
  - Each screen appearance can be individually configured
  - All screen layouts, placement & configuration are preserved at exit
- Export Dataset
  - Facilitates **batch export of multiple sessions** into T3 (Talrómur3) dataset format
  - Groups different recording sessions of the same speaker into a common dataset
  - Option to **skip rejected utterances** during export
  - **EBU R 128 loudness normalization** with selectable presets or custom LUFS target, true peak limiting
  - **Voice Activity Detection (VAD)** with speech segment timestamps
    - **OmniVAD** (included by default) - CPU-only, based on FireRedVAD DFSMN model via ncnn, no PyTorch required
    - **Silero VAD** (optional) - requires PyTorch, install with `pip install revoxx[silero]`
    - Both VAD backends can run simultaneously for comparison
- **ASR-based Recording Verification**
  - Automatically transcribe recordings via an **OpenAI-compatible ASR endpoint** (cloud or self-hosted) and compare against the script text
  - Spot misread utterances via configurable character-level similarity threshold
  - **Toggle** between script text and ASR transcription in the display with a single keystroke
  - **Manual override** for false negatives, direct navigation to mismatches

## Installation

<details>
<summary><b>Prerequisites</b></summary>

### Tkinter

Revoxx requires Tkinter for its graphical user interface. Tkinter is usually included with Python, but may need separate installation on some systems:

**macOS**: Tkinter should be included with Python.org installers and Homebrew Python, but integration issues can occur. If you encounter problems:
- For Homebrew Python: Try `brew install python-tk`
- For Python.org installers: Reinstall Python with the official installer
- Consider using a virtual environment with a fresh Python installation

**Linux**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

**Windows**: Tkinter is included with the standard Python installer.

**Verify Tkinter installation**:
```bash
python3 -c "import tkinter; print('Tkinter is installed')"
```

</details>

<details>
<summary><b>Basic Installation</b></summary>

### Using uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
uv pip install revoxx           # From PyPI (includes OmniVAD)
uv pip install .                # From source
uv pip install revoxx[silero]   # With additional Silero VAD support
```

### Using pip

```bash
pip install revoxx             # From PyPI (includes OmniVAD)
pip install .                  # From source
pip install revoxx[silero]     # With additional Silero VAD support
```

### From source

```bash
git clone https://github.com/icelandic-lt/revoxx.git
cd revoxx
# Then use either uv or pip as shown above
```

### Voice Activity Detection (VAD)

**OmniVAD** is included by default and requires no additional installation. It uses the FireRedVAD DFSMN model via ncnn and runs entirely on CPU.

To additionally install **Silero VAD** (requires PyTorch):

```bash
# Option 1: CPU-only PyTorch (recommended, smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install revoxx[silero]

# Option 2: Default PyTorch (often includes CUDA support > 2GB disk-space required)
pip install revoxx[silero]
```

Both VAD backends can be enabled simultaneously during export to produce separate result files for comparison.

</details>

<details>
<summary><b>Development Setup</b></summary>

### For development

#### Using uv (recommended)

```bash
git clone https://github.com/icelandic-lt/revoxx.git
cd revoxx

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e .[dev]
# With Silero VAD support:
uv pip install -e .[dev,silero]
```

#### Using pip (traditional)

```bash
git clone https://github.com/icelandic-lt/revoxx.git
cd revoxx

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]
# With Silero VAD support:
pip install -e .[dev,silero]
```

Development dependencies include:
- **black**: Code formatter (pinned to 25.x for Python 3.9 compatibility)
- **isort**: Import statement organizer
- **flake8**: Code linter
- **pytest**: Testing framework
- **pytest-cov**: Code coverage reporting

> **Note**: Black is pinned to version 25.x to ensure consistent formatting rules across all supported Python versions.

### Tcl/Tk for standalone Python builds

Standalone Python distributions (e.g. downloaded by `uv`) may not include Tcl/Tk data files. If Tkinter tests fail with `Cannot find a usable init.tcl`, set the `TCL_LIBRARY` environment variable to point to your installed Tcl library:

```bash
# macOS with Homebrew (ARM)
export TCL_LIBRARY=/opt/homebrew/opt/tcl-tk/lib/tcl9.0

# macOS with Homebrew (Intel)
export TCL_LIBRARY=/usr/local/opt/tcl-tk/lib/tcl9.0
```

In PyCharm: Run > Edit Configurations > Environment variables.

### Running code quality checks

```bash
# Format code
black revoxx/ scripts_module/ tests/

# Check code style
flake8 revoxx/ scripts_module/ tests/

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=revoxx --cov-report=html
```

</details>

## Running Revoxx

### After installation

Once installed, you can run Revoxx using:

```bash
revoxx
```

**macOS Note:** The first launch may take longer than usual as macOS verifies the application (Gatekeeper security check). Subsequent launches will be faster.

### During development (without installation)

Run as a Python module:

```bash
python -m revoxx
```

### In PyCharm or other IDEs

Configure your run configuration with:
- **Module name**: `revoxx` (not script path)
- **Working directory**: Project root directory

### Command-line tools

The package includes additional utilities:

```bash
revoxx-export    # Export sessions to dataset format
revoxx-vadiate   # Voice Activity Detection tool (uses OmniVAD by default)
```

The `revoxx-vadiate` tool uses OmniVAD by default. Use `--backend silero` to use Silero VAD instead (requires `pip install revoxx[silero]`).

### Command-line arguments

```bash
revoxx --help                    # Show all available options
revoxx --show-devices            # List available audio devices
revoxx --session path/to/session # Open specific session
```

## Usage

For a guide on using Revoxx, please see the [User Guide](https://github.com/icelandic-lt/revoxx/blob/main/revoxx/doc/USER_GUIDE.md).

## Prepare recordings

Before you start recording, you need to prepare an utterance script with the utterances you want to record. This can be simplified by using the "Import Text to Script" Dialog:

<img src="https://raw.githubusercontent.com/icelandic-lt/revoxx/main/doc/import_raw_text.png" alt="Raw text import dialog" width="50%"/>

This dialog takes an input script of raw text and converts it into an utterance script. You can redo this for the same input text as many times you want, e.g. if you want to use separate emotional levels for different speakers.

### Utterance script format

A script file follows Festival-style format. The script should be a simple text file with one utterance per line. The utterances can be in any language you want.

For a script with emotion levels:

```text
( <unique id> "<emotion-level>: <utterance>" )
```

For a script without emotion levels. This format was used for recording our non-emotional "addendas":

```text
( <unique id> "<utterance>" )
```

You can see for both formats an example in the directory [t3_scripts](https://github.com/icelandic-lt/revoxx/tree/main/t3_scripts).

The emotion levels can be from any monotonic numerical value range you want. If you want to follow Talrómur 3 conventions, you can use emotion intensity levels 1-5 and 6 emotions: neutral, happy, sad, angry, surprised, and helpful.
The emotion intensity levels are used to control the emotion intensity of the speech in combination with the specific emotion.
Neutral speech is treated as intensity level 0 at dataset export.

## Known Issues

### Linux: USB Audio Output Devices

On Linux systems, USB audio output devices (e.g., Focusrite Scarlett, Clarett+) may temporarily disappear from the device list during playback. This is a known issue with the interaction between PortAudio and ALSA/PulseAudio. 

**Symptoms:**
- Output device works for the first playback but fails on subsequent attempts
- Error message: "Output device 'DeviceName' disappeared from system"
- Device reappears after restarting the application
- Input devices (recording) are typically not affected

**Workarounds:**
- The application will automatically fall back to the system default audio output device
- If the issue persists, restart the application
- Consider updating to the latest version of sounddevice (>=0.5.1)
- Check that no other application is claiming exclusive access to the device

## Acknowledgements
This project is part of the program Language Technology for Icelandic. The program was funded by the Icelandic Ministry of Culture and Business Affairs.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/icelandic-lt/revoxx/blob/main/LICENSE) file for details.

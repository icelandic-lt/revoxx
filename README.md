# Revoxx - Record Voices

This repository provides **Revoxx**, a graphical voice recording application for recording raw speech and generating datasets.

![Version](https://img.shields.io/badge/Version-main-darkgreen)
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![CI Status](https://img.shields.io/badge/CI-[unavailable]-red)
![Docker](https://img.shields.io/badge/Docker-[unavailable]-red)

## Overview

**Revoxx** has been created by [Grammatek ehf](https://www.grammatek.com) and is part of the [Icelandic Language Technology Programme](https://github.com/icelandic-lt/icelandic-lt).

- **Category:** [TTS](https://github.com/icelandic-lt/icelandic-lt/blob/main/doc/tts.md)
- **Domain:** Laptop/Workstation
- **Languages:** Python
- **Language Version/Dialect:**
  - Python: 3.10
- **Audience**: Developers, Researchers
- **Origins:** [Icelandic EmoSpeech scripts](https://github.com/icelandic-lt/emospeech-scripts)

## Status
![Beta](https://img.shields.io/badge/Beta-darkviolet)

## System Requirements
- **Operating System:** Linux/OS-X, should work on Windows
- **Recording:** Audio Interface, good voice microphone and headphones

## Description

**Revoxx** is a graphical voice recorder specialized in recording TTS datasets quickly and reliably.<br>
You can use this project to create emotional / non-emotional voice recordings on a Workstation / Laptop with suitable audio equipment.
It has integrated support to easily transform raw recordings into datasets for training TTS voice models.

**Revoxx** is a vastly improved version of the [Icelandic EmoSpeech scripts](https://github.com/icelandic-lt/emospeech-scripts),
written from scratch.<br>

We have condensed our experience from when we recorded [Talrómur 3](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/344),
the Icelandic emotional speech dataset, and created this tool to minimize hassle, valuable recording & post-processing time.

- **Revoxx** makes recording of speech fast, reliable and convenient for the recording engineer and the voice talent
- Integrates all necessary tools to check if recordings & equipment meet your expected requirements
- Realtime feedback for recording levels, mel spectrograms, max. recorded frequency, etc.
- Probes audio equipment for supported settings (e.g. Sample Rate, Bit Depth, Input & Output channels)
- Recordings are organized into sessions, with consistent audio settings & metadata for all recordings
- Choose from different industry-standard presets for target Peak / RMS levels and use Monitoring mode to calibrate your inputs
- All utterances can be re-recorded as many times as you want, all raw recordings are preserved, even when deleted
- Most important functionality is available via easy to remember keyboard shortcuts

## Installation

### From PyPI (later, if available)

```bash
pip install revoxx
```

### From source

Clone the repository and install in development mode:

```bash
git clone https://github.com/icelandic-lt/revoxx.git
cd revoxx
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### For development

Install with development dependencies:

```bash
pip install -e .[dev]
```

This installs additional tools for development:
- **black**: Code formatter
- **isort**: Import statement organizer
- **flake8**: Code linter
- **pytest**: Testing framework
- **pytest-cov**: Code coverage reporting

#### Running code quality checks

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

## Running Revoxx

### After installation

Once installed, you can run Revoxx using:

```bash
revoxx
```

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
revoxx-vadiate   # Voice Activity Detection tool
```

### Command-line arguments

```bash
revoxx --help                    # Show all available options
revoxx --show-devices            # List available audio devices
revoxx --session path/to/session # Open specific session
```

## Prepare recordings

Before you start recording, you should prepare a script with the utterances you want to record.
The script should be a simple text file with one utterance per line. The utterances can be in any language you want.

A script file follows Festival-style and has the following possible two formats:

For a script with emotion levels:

```text
( <unique id> "<emotion-level>: <utterance>" )
```

For a script without emotion levels. This format was used for recording our non-emotional "addendas":

```text
( <unique id> "<utterance>" )
```

You can see for both formats an example in the directory [scripts](scripts).

The emotion levels can be from any monotonic numerical value range you want. If you want to follow Talrómur 3 dataset conventions, you can use emotion levels 0-5 for 6 emotions: neutral, happy, sad, angry, surprised, and helpful.
The emotion levels are used to control the emotion intensity of the speech in combination with the specific emotion.
Neutral speech corresponds to emotion level 0.

## Record dataset

to be defined

## Acknowledgements
This project is part of the program Language Technology for Icelandic. The program was funded by the Icelandic Ministry of Culture and Business Affairs.

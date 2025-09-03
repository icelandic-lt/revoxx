# Revoxx User Guide

## Quick Start Guide

This guide will walk you through using Revoxx for speech recording.

## 1. Prepare Your Recording Script

Before you can start recording, you need to prepare a script containing all the utterances you want to record. Revoxx uses the Festival-style script format for organizing utterances.

### Using the Import Tool

The easiest way to create a script is through the built-in import tool:

1. Navigate to **Tools → Import Text to Script** in the menu bar.
2. Select your input text and output script file paths.
3. Configure the text splitting options:
   - **Split by**: Choose how to divide your text into utterances:
     - **Lines**: Each line in the input file becomes one utterance
     - **Sentences**: Text is split at sentence boundaries (periods, question marks, exclamation points)
     - **Paragraphs**: Each paragraph (separated by blank lines) becomes one utterance
   - **Maximum characters**: Set a character limit for each utterance. The tool displays statistics for your input file showing the character count distribution to help you choose an appropriate limit.
4. Configure additional import settings:
   - **Emotion levels**: If you're recording emotional speech with different intensities, you can choose between two methods:
     - **Fixed levels**: Define specific emotion levels (e.g., 1-5) that will be assigned sequentially or randomly
     - **Normal distribution**: Set parameters for a gaussian distribution:
       - **Mean**: The center of the distribution (average emotion level)
       - **Standard deviation**: How spread out the emotion levels should be
       - **Minimum/Maximum**: The bounds for the emotion levels
       - The tool displays a graph showing the actual distribution of emotion levels that will be assigned to your utterances based on these parameters
   - **Output location**: Select where the generated script file will be saved.
5. Click **Import** to generate your script file in the correct format.

### Script Format Examples

**Emotional speech:**
```
( spk1_happy_001 "happy-3: I love sunny days!" )
( spk1_sad_002 "sad-4: The rain makes me melancholy." )
```

**Neutral speech:**
```
( spk1_001 "The weather forecast predicts sunshine." )
( spk1_002 "Tomorrow will be partly cloudy." )
```

## 2. Create a New Recording Session

Once you have prepared your script, you can create a new recording session:

1. Select **File → New Session...** from the menu to open the session configuration dialog.
2. Configure the various session settings:

   ### Speaker Information
   - **Speaker Name**: Enter the full name of the voice talent who will be recording.
   - **Gender**: Select the appropriate gender option (M/F/X).
   - **Emotion**: Choose either a specific emotion (happy, sad, angry, etc.) or select "neutral" for non-emotional recordings.

   ### Script Selection
   - Click **Browse** to select your prepared script file.
   - The script file determines which utterances will appear during the recording session.

   ### Session Storage
   - **Base Directory**: Choose the parent folder where all your recording sessions will be stored.
   - **Session Directory**: You can enter a descriptive name for this session - or a subfolder with the speaker name and emotion will automatically be created.

   ### Audio Configuration
   - **Input Device**: Select your audio interface from the dropdown list.
   - **Sample Rate**: Choose at least 48000 Hz for professional quality recordings (44100 Hz is also acceptable).
   - **Bit Depth**: Select 24-bit for optimal dynamic range.
   - **Recording Format**: FLAC format is recommended as it provides losslessly compressed audio.

3. After configuring all settings, click **OK** to start your recording session.

## 3. Main Recording Interface

### Selecting Recording Standards

Before calibrating, select the appropriate recording standard for your project. Navigate to **Settings → Level Meter Preset** and choose from these industry standards:

- **EBU R128 broadcast** (default): For broadcast content with -23 LUFS integrated loudness
- **ACX/Audible audiobook**: For audiobook recording with RMS levels between -23 to -18 dB
- **Podcast standard**: For podcast production targeting -16 to -14 LUFS
- **Film dialog recording**: For film/video dialog with peaks between -27 to -20 dBFS
- **Music vocal recording**: For music production with peaks between -18 to -12 dBFS

Each preset automatically configures the level meter with appropriate target ranges, warning thresholds, and measurement windows specific to that recording standard.

### Before Recording - Calibration

It's important to calibrate your input levels before starting the recording session:

1. Press **M** to enter Monitor Mode. In this mode, you can see the input levels without recording.
2. Ask the speaker to test their voice at the volume they will use during recording.
3. Adjust the input gain on your audio interface until the levels fall within the target range for your selected preset (shown in dotted lines on the meter).
4. Press **M** again to exit Monitor Mode and return to normal recording mode.

### Recording Controls

| Key          | Action                                                          |
|--------------|-----------------------------------------------------------------|
| **SPACE**    | Start/Stop recording                                            |
| **P**        | Play the current take                                           |
| **⌘/Ctrl+D** | Delete the current take (it moves to trash folder subdirectory) |
| **↑/↓**      | Navigate between utterances in the list                         |
| **←/→**      | Navigate between different takes of the current utterance       |
| **⌘/Ctrl+U** | Change the utterance ordering method                            |
| **F1**       | Show all keyboard shortcuts                                     |

**Important:** When multiple takes exist for an utterance, the system always uses the most recent take for export. You can navigate through all takes using the left/right arrow keys to review them, but only the last recorded take will be included in the final dataset.

### Recording Process

When recording each utterance, follow these steps:

1. Speaker takes a deep breath before recording starts
2. Press **SPACE** to start recording
3. Wait 1-2 seconds (speaker holds breath)
4. Speaker reads the utterance
5. Speaker holds breath at the end
6. Wait 1-2 seconds before pressing **SPACE** to stop
7. Review the spectrogram or press **P** to play

**Breathing**: No breath sounds at the beginning or end of recordings. Speaker should inhale before recording starts and hold breath when finished.

**Why silence matters**: The pauses at the beginning and end of each recording serve two purposes:

- They provide clean boundaries for audio processing algorithms
- When VAD (Voice Activity Detection) is enabled during export, these silence regions help the system accurately detect speech boundaries and help trim recordings to consistent silence lengths

**VAD Processing**: During dataset export, if VAD analysis is enabled, the system will:

- Detect the exact start and end of speech in each recording
- Generate precise timing information for downstream TTS training

### Display Options

| Key           | Action                        |
|---------------|-------------------------------|
| **M**         | Toggle meter display          |
| **I**         | Toggle info display at bottom |
| **F10**       | Fullscreen main window        |
| **Shift+F10** | Fullscreen speaker window     |

### Spectrogram Navigation

The spectrograms provide visual feedback of your recordings. You can interact with them using:

- **Mouse wheel**: Scroll up to zoom in for more detail, scroll down to zoom out.
- **Right-click + drag**: When zoomed in, hold the right mouse button and drag to pan left or right through the spectrogram.

## 4. Multi-Screen Setup

Revoxx supports dual-screen setups, allowing you to have separate displays for the engineer and the speaker.

### Enabling Speaker Display

1. Navigate to **Settings → 2nd Window → Enable 2nd Window** in the menu.
2. A second window will appear that can mirror the main recording interface or be configured for minimal display.
3. Drag this window to your second monitor, external display, or iPad.
4. Configure what appears on the second window through the settings:
   - **Full Interface**: Shows everything from the main window including spectrograms and meters.
   - **Minimal Mode**: Shows only the utterance text and status bar to maximize screen space for the text.
   - You can toggle individual elements like spectrograms, meters, and info displays.

The minimal mode is recommended for speakers as it removes technical distractions and makes the text as large as possible.

### Using iPad with Sidecar (macOS)

For a portable dual-screen setup on macOS:

1. Connect your iPad to your Mac using the Sidecar feature.
2. Once connected, simply drag the speaker window to the iPad display.
3. The speaker can now comfortably read from the iPad while you control the recording from your main screen.

## 5. Session Management

### Utterance Ordering

The order in which utterances appear during recording can be customized to suit your workflow:

- By default, utterances appear in the order they are listed in the script file.
- Press **⌘/Ctrl+U** to open the utterance ordering dialog where you can sort by:
  - **Label**: Alphabetical order by utterance ID
  - **Emotion level**: Groups utterances by emotion and intensity
  - **Text content**: Alphabetical order by the actual text
  - **Text length**: Shortest to longest utterances
  - **Number of takes**: Prioritize utterances with fewer recordings

For each sorting option, you can also choose the sort direction.

### Finding Utterances

When you need to locate specific utterances quickly:

- Use **Edit → Find Utterance** (⌘/Ctrl+F) to open the search dialog.
- You can search by typing any part of the utterance text.
- The search results can be sorted using the same criteria as the ordering options.
- Double-click any result to jump directly to that utterance.

### Session Progress

Revoxx automatically manages your recording progress:

- Your progress is saved automatically after each recording.
- Use **File → Recent Sessions** to see a list of your recent work and quickly resume where you left off.
- All session settings, including audio configuration and display preferences, are preserved between sessions.

## 6. Best Practices

### Recording Environment

Creating the right recording environment is essential for high-quality results:

1. **Acoustic Treatment**: Use a professionally treated room or vocal booth to minimize reflections and external noise.
2. **Microphone Position**: Maintain a consistent distance of 6-12 inches from the microphone. Use the monitoring mode to verify that levels remain stable as the speaker moves.
3. **Pop Filter**: Always use a pop filter positioned between the speaker and microphone to prevent plosive sounds from ruining takes.
4. **Headphones**: Provide closed-back headphones for the voice talent to prevent audio bleed into the microphone.

### Recording Workflow

Follow these guidelines for efficient and consistent recording sessions:

1. **Warm-up**: Always have the speaker perform vocal exercises before starting to ensure their voice is ready.
2. **Consistency**: Help the speaker maintain the same sitting position and energy level throughout the session.
3. **Breathing technique**: For longer utterances with natural pauses, breathing between phrases is acceptable and natural. However, ensure no breath sounds at the beginning or end of the recording.
4. **Breaks**: Schedule regular breaks every 30-45 minutes to prevent vocal fatigue and maintain quality.
5. **Multiple Takes**: For difficult or important utterances, record 2-3 takes to have options during dataset creation.
6. **Review**: Periodically stop to review recent recordings and ensure quality standards are being maintained.

## 7. Export and Dataset Creation

Once you have completed recording, you can export your sessions into datasets suitable for TTS training:

1. Select **File → Export Dataset** from the menu to open the export dialog.
2. Choose which recording sessions you want to include in the dataset. You can select multiple sessions from any speaker. Datasets are combined by speaker name automatically.
3. Configure the export options according to your needs:
   - T3 format is chosen automatically
   - **VAD analysis**: If VAD support is installed, you can generate voice activity timestamps for each audio file.
4. Click **Export** to create an organized dataset structure that's ready for TTS model training. The export process will handle file naming, metadata generation, and directory organization automatically.

## 8. Troubleshooting

### Audio Issues

If you encounter audio problems during recording:

- **No input signal**: Check that your audio interface is properly connected and that the correct drivers are installed. Verify that the input device is selected correctly and you have selected the correct channel.
- **Clipping or distortion**: Reduce the input gain on your audio interface. The peaks should never exceed -3 dB during normal speech.
- **Audio dropouts**: This may indicate buffer size issues. Close unnecessary applications and ensure your computer meets the performance requirements.

### Display Issues

For problems with the visual interface:

- **Text too small/large**: The text size adjusts automatically based on window size. Simply resize the window to change the text size.
- **Speaker window position lost**: If the second window appears off-screen, go to Settings → 2nd Window and disable/re-enable it.
- **Spectrogram not updating**: Press 'M' to toggle the meter display off and on again. This will refresh the visualization.

### Performance

To maintain optimal performance:

- **Slow response**: Close other CPU-intensive applications while recording. Audio processing requires significant resources. We have had best results with M-based Macs
- **Disk space**: Ensure you have adequate free space on your recording drive. Each hour of 48kHz/24-bit recording requires approximately 500MB.
- **Memory usage**: If the application becomes sluggish after extended use, save your session and restart Revoxx to clear the memory.

## Need Help?

- Press **F1** for keyboard shortcuts
- Report issues at: https://github.com/icelandic-lt/revoxx/issues
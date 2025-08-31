Talrómur 3 Script Files
=======================

This directory contains example script files in Festival Script Format that were 
used for recording the Talrómur 3 Icelandic emotional speech dataset.

File Format
-----------
The scripts follow the Festival script format with two variations:

1. Scripts with emotion levels (used for emotional recordings):
   ( <unique_id> "<emotion-level>: <utterance>" )
   
   Example:
   ( utt_001 "3: Ég er svo glöð í dag!" )

2. Scripts without emotion levels (used for non-emotional/neutral recordings):
   ( <unique_id> "<utterance>" )
   
   Example:
   ( utt_001 "Halló, hvað segirðu gott?" )

Files in this Directory
-----------------------
- t3_intensity_script.txt - Script with emotion intensity levels (0-5) used for 
                            emotional recordings in Talrómur 3
- t3_addendum.txt - Additional neutral/non-emotional utterances recorded as 
                    supplements to the main dataset

Emotion Levels
--------------
Talrómur 3 uses emotion levels 0-5 to indicate the intensity of the emotion:
- 0: Neutral (no emotion)
- 1-5: Increasing emotional intensity

These scripts serve as examples for creating your own recording scripts in the
correct format for use with Revoxx.
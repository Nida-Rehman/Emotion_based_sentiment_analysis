# Emotion-Based Sentiment Analysis

## Project Overview
This project integrates facial emotion recognition and sentiment analysis from text feedback to provide insights into emotional states. It combines machine learning techniques with a user-friendly web interface, allowing users to analyze emotions through facial expressions or text sentiment.

---

## Features
1. Text Sentiment Analysis  
   - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to classify text feedback into:
     - Positive
     - Negative
     - Neutral

2. Facial Emotion Recognition 
   - Uses the `FER` library to detect emotions like happiness, anger, and sadness from video frames.
   - Outputs the most frequently detected emotion over a 10-second duration.

3. Web-Based Interface
   - Simple and responsive interface for inputting text feedback or capturing facial expressions via webcam.

---

## How It Works
1. Text Analysis Workflow:
   - User submits text feedback via the web form.
   - VADER analyzes the sentiment and classifies it as Positive, Negative, or Neutral.
   - The result is displayed on the web page.

2. Facial Emotion Workflow:
   - Webcam captures video frames for 10 seconds.
   - FER library analyzes emotions for each frame.
   - The most frequently detected emotion is displayed on the results page.

---

## Prerequisites
1. Python 3.8 or higher
2. Flask for the web framework
3. Required Python libraries:
   - `flask`
   - `opencv-python`
   - `fer`
   - `vaderSentiment`



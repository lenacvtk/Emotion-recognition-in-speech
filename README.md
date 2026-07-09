# Emotion-recognition-in-speech

An end-to-end deep learning project focused on finding the optimal balance between classification accuracy and inference latency for real-time Speech Emotion Recognition (SER). 

This project systematically evaluates 18 different Convolutional Neural Network (CNN) architectures trained on two distinct audio feature representations to build a reliable, live-updating emotion detection system.

## Project Goals
* **Real-Time Inference:** Maintain a strict inference latency threshold under 100ms for seamless live processing.
* **Feature Comparison:** Benchmark **Log-Mel Spectrograms** (spatial-frequency energy) against **MFCC matrices** (compressed cepstral coefficients) to evaluate their impact on CNN pattern recognition.
* **Complexity Scaling:** Analyze how model complexity (scaling convolutional filters from $N$ to $512N$) affects accuracy vs. computational cost.
* **Live Implementation:** A rolling-window pipeline using `PyAudio` that captures mic input and displays predicted emotions instantly.

## Dataset & Features
* **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
* **Audio Pipeline:** 48kHz stereo downsampled to 16kHz mono, amplitude-normalized, and zero-padded for fixed-length CNN inputs.

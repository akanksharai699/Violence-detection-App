# Violence Detection System (Video-Based)

## Overview
A deep learning–based system for detecting violent activities in videos such as
street fights and assaults. The model analyzes temporal information from video
streams and predicts Violence / Non-Violence with confidence scores.

## Motivation
Initial training on sports-based datasets resulted in domain bias.
This was identified through incorrect predictions on real street-fight videos.
The issue was resolved by retraining on the RWF-2000 CCTV dataset.

## Architecture
- Video frame extraction using OpenCV
- CNN + LSTM for spatiotemporal feature learning
- GPU-accelerated training using PyTorch
- Sliding-window real-time inference on video streams

## Dataset
- Hockey Fight Dataset (baseline)
- RWF-2000 (real-world CCTV violence dataset)

## Results
- Significant improvement on real-world street-fight videos
- ~60% accuracy on unseen real-world data
- Reduced domain bias compared to sports-only training

## Key Challenges Solved
- Domain shift between sports and street violence
- Corrupted / empty video samples
- Model versioning conflicts
- GPU vs CPU training verification
- Stable real-time inference without webcam dependency

## Tech Stack
- Python, PyTorch
- OpenCV
- CUDA (GPU acceleration)
- CNN + LSTM

## Future Improvements
- Temporal voting for higher recall
- 3D CNN / SlowFast models
- CCTV IP stream integration

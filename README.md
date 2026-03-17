# Multimodal Deepfake Detection System

A comprehensive deep learning-based system designed to detect deepfakes across multiple media formats: **Images, Video, and Audio**. This project features a high-performance Flask backend and a modern React frontend.

## 🚀 Key Features

- **Multimodal Detection**: Supports detection for images (JPG/PNG), videos (MP4), and audio (WAV).
- **High Accuracy**: Achieving an overall accuracy of **94.4%** in testing across various media types.
- **Ensemble Model Strategy**: 
  - **Images**: Combines **Vision Transformer (ViT)** and **ConvNeXt** models for robust prediction.
  - **Videos**: Extracts and analyzes multiple frames using an ensemble approach.
  - **Audio**: Utilizes **Wav2Vec2** for precise synthetic voice detection.
- **Modern UI**: A premium, responsive interface built with React, Tailwind CSS, and Shadcn UI.
- **Optimized Storage**: Intelligent local storage management to handle high-resolution media previews without crashing.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, TensorFlow, PyTorch, Transformers (HuggingFace), OpenCV, Librosa.
- **Frontend**: React, Vite, Tailwind CSS, Shadcn UI, React Router.
- **Testing**: Automated testing suite with comprehensive scoring.

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- **FFmpeg**: Required for audio and video processing.
  - *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saiReddy2025/multimodal-deepfake-detection.git
   cd multimodal-deepfake-detection
   ```

2. **Backend Setup**:
   ```bash
   cd Backend/Models
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**:
   ```bash
   cd Frontend
   npm install
   ```

## 🚀 Running the Project

You can use the helper script in the root directory:
```bash
python start_project.py
```
This will concurrently start the backend (Port 5000) and the frontend (Vite dev server).

## 📊 Performance Metrics

| Media Type | Accuracy | Avg. Latency |
| :--- | :--- | :--- |
| Image | ~95% | 0.2s |
| Audio | ~92% | 0.6s |
| Video | ~96% | 8.3s |

## ⚠️ Important Note on Model Files

Due to GitHub's file size limits, the large pre-trained model files (`.pkl`, `.pt`) are excluded from this repository. To run the system, you must place the appropriate model weights in the `Backend/Models/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

# AI-Object-Classification
An AI-based Object Classification project using Teachable Machine, TensorFlow/Keras, OpenCV, and Python for real-time webcam object recognition.

## Overview

This project demonstrates a real-time Object Classification system using **Teachable Machine**, **TensorFlow/Keras**, **OpenCV**, and **Python**.

The application captures live video from the webcam, classifies the detected object, and displays the predicted class with its confidence score in real time.

---

## Features

- Real-time object classification using a webcam
- AI model trained with Google Teachable Machine
- Displays object name and confidence score
- Detects unknown objects using a confidence threshold
- Easy-to-use interface with OpenCV
- Press **Q** to exit the application

---

## Classes

The model was trained to recognize the following objects:

- Phone
- Cup
- Pen
- Nothing

---

## Technologies Used

- Python
- OpenCV
- TensorFlow
- Keras
- NumPy
- Pillow
- Google Teachable Machine

---

## Project Structure

```
AI-Object-Classification
│
├── main.py
├── keras_model.h5
├── labels.txt
├── requirements.txt
├── README.md
└── screenshots
    ├── screenshot1.png
    └── screenshot2.png
```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Output

The application opens the webcam and displays:

- Detected object
- Confidence score
- Unknown object detection
- Real-time prediction

Press **Q** to close the application.

---

## Screenshots

All output screenshots are stored in the **screenshots** folder.

## Author

**Ghaith**

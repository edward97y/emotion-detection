🎭 Real-Time Emotion Detection using YOLO and Mini-Xception

This project is a real-time facial emotion detection system that combines:  
- [YOLOv8](https://github.com/ultralytics/ultralytics) for face detection  
- A Mini-Xception model (trained on FER2013 and RAF-DB) for emotion classification  
- OpenCV for real-time webcam processing  

The app captures frames from your webcam, detects faces using YOLO, and then classifies the detected faces into emotions with a lightweight CNN model.

---

 🚀 Features
- Real-time face detection using YOLOv8  
- Emotion classification with Mini-Xception  
- Predicts one of the following emotions:  
  - Angry  
  - Fear  
  - Happy  
  - Neutral  
  - Sad  
- Displays prediction label and confidence score above each detected face  

---

🧠 Model Details
The emotion classifier is based on the Mini-Xception architecture:
- A lightweight version of the Xception model optimized for speed and real-time performance  
- Trained on FER2013 and RAF-DB datasets  
- Outputs probabilities for the 5 supported emotions  
- Designed to run efficiently alongside YOLO in a live video feed  

---

🛠 Requirements
Python 3.8+ is recommended.  

Installation
Install dependencies with:

 pip install -r requirements.txt

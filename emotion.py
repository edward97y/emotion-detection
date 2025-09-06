import cv2 as cv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



classifier = tf.keras.models.load_model('/home/ciphermind/programing/python/anaconda/jupyter/streamlit/emotion_project/emotion_modelv1.keras')


yolo=YOLO('/home/ciphermind/programing/python/anaconda/jupyter/streamlit/emotion_project/yolov8n-face.pt')

classes=['angry', 'fear', 'happy', 'neutral', 'sad']

def face_detect(frame):
    
    results=yolo.predict(frame,verbose=False)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
                conf=box.conf[0].cpu().numpy()
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                face=frame[y1:y2,x1:x2]
                if face.size >0:
                    img=cv.cvtColor(face,cv.COLOR_BGR2GRAY)
                    img=cv.resize(img,(48,48))
                    img=img.astype('float32')/255.0
                    img=np.expand_dims(img,axis=-1)
                    img=np.expand_dims(img,axis=0)
                    prediction=classifier.predict(img,verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    emotion_label=classes[predicted_class]
                    label=f"{emotion_label} ({confidence:.2f})"
                    font=cv.FONT_HERSHEY_COMPLEX
                    cv.putText(frame,label,(x1,y1-10),font,0.7,(0,255,0),2)
    return frame

   


cap=cv.VideoCapture(0)
while(cap.isOpened()):
    ret,frame=cap.read()
    if  not ret:
        break
    frame=face_detect(frame)
    cv.imshow('Emotion Detection',frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

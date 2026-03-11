import streamlit as st
import cv2

# Fix for headless OpenCV in Streamlit deployment
cv2.imshow = lambda *args: None
cv2.IMREAD_COLOR = 1
cv2.IMREAD_GRAYSCALE = 0
cv2.IMREAD_UNCHANGED = -1
cv2.setNumThreads = lambda *args: None
cv2.INTER_LINEAR = 1
cv2.INTER_NEAREST = 0
cv2.INTER_CUBIC = 2
cv2.INTER_AREA = 3
cv2.INTER_LANCZOS4 = 4
cv2.BORDER_CONSTANT = 0
cv2.BORDER_REPLICATE = 1
cv2.BORDER_REFLECT = 2
cv2.BORDER_WRAP = 3
cv2.BORDER_REFLECT_101 = 4

cv2.COLOR_BGR2RGB = 4
cv2.COLOR_RGB2BGR = 4
# cv2.cvtColor removed, use numpy slicing instead

from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

st.title("Human Detection for Suitcase Following")

image = st.camera_input("Capture an image")

if image is not None:
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]  # RGB to BGR
    results = model(img)
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # person class
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]  # BGR to RGB
    st.image(img)
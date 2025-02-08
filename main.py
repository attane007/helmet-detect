import cv2
import torch
from datetime import datetime
import os
import time

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create directory for storing images
os.makedirs("storage", exist_ok=True)
os.makedirs("storage/face_db", exist_ok=True)  # Directory for face images

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

last_save_time = 0

def save_face(image, face_location):
    """Save face to the database"""
    x, y, w, h = face_location
    face = image[y:y + h, x:x + w]
    filename = f"storage/face_db/face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, face)
    return filename

while True:
    start_time = time.time()  # Get the current time before processing a frame

    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Detection data

    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name']

        if name: 
            print(name)

        # Draw bounding boxes for detected objects
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with bounding boxes
    cv2.imshow("Helmet Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

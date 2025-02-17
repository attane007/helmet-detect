import cv2
import torch
from datetime import datetime
import os
import time

# Function to list available cameras
def find_available_cameras(max_test=10):
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Let user choose a camera
cameras = find_available_cameras()
if not cameras:
    print("No cameras found. Exiting...")
    exit()

print("Available cameras:")
for cam in cameras:
    print(f"{cam}: Camera {cam}")

selected_cam = int(input("Select a camera by number: "))
if selected_cam not in cameras:
    print("Invalid selection. Exiting...")
    exit()

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Create directory for storing images
os.makedirs("storage", exist_ok=True)
os.makedirs("storage/face_db", exist_ok=True)  # Directory for face images

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(selected_cam)

last_save_time = 0

def save_face(image, face_location):
    """Save face to the database"""
    x, y, w, h = face_location
    face = image[y:y + h, x:x + w]
    filename = f"storage/face_db/face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, face)
    return filename

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]

    helmet_detected = False
    person_detected = False

    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, name, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']

        if name == 'person':
            person_detected = True
            color = (0, 0, 255)  # Red for person
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if name == 'helmet':
            helmet_detected = True
            color = (0, 255, 0)  # Green for helmet
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Helmet', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check for person without helmet
    if person_detected and not helmet_detected:
        current_time = time.time()
        if current_time - last_save_time >= 3:  # บันทึกไม่เกินทุก 3 วินาที
            face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                face_path = save_face(frame, (fx, fy, fw, fh))
                print(f"Face saved: {face_path}")

            full_frame_filename = f"storage/no_helmet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(full_frame_filename, frame)
            print(f"Saved frame without helmet: {full_frame_filename}")
            last_save_time = current_time

    # Display the frame
    cv2.imshow("Helmet Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
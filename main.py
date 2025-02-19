import cv2
import torch
from datetime import datetime
import os
import time
from yolov5 import YOLOv5
import pathlib
import platform
import warnings

# Ignore specific warning
warnings.filterwarnings("ignore", category=FutureWarning)

# checking OS platform
if platform.system() == 'Windows':
    print("windows")
    pathlib.PosixPath = pathlib.WindowsPath

# Load YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = YOLOv5('helmet_traineds.pt', device='cpu')

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

# Create directory for storing images
os.makedirs("storage", exist_ok=True)

# Open the camera
cap = cv2.VideoCapture(selected_cam)

last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLO
    results = model.predict(frame)
    detections = results.pandas().xyxy[0]


    motorcyclists = []
    helmets = []

    # เก็บ bounding box ของ motorcyclist และ helmet
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, name, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
        
        if name == 'motorcyclist':
            motorcyclists.append((xmin, ymin, xmax, ymax))  # เก็บ bounding box ของ motorcyclist
            color = (0, 0, 255)  # สีแดง
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if name == 'helmet':
            helmets.append((xmin, ymin, xmax, ymax))  # เก็บ bounding box ของ helmet
            color = (0, 255, 0)  # สีเขียว
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Helmet', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ตรวจสอบว่าคนขับมอเตอร์ไซค์มีหมวกกันน็อคหรือไม่
    for (pxmin, pymin, pxmax, pymax) in motorcyclists:
        helmet_detected = False

        for (hxmin, hymin, hxmax, hymax) in helmets:
            # ตรวจสอบว่าหมวกอยู่ในขอบเขตของ motorcyclist
            if hxmin >= pxmin and hxmax <= pxmax and hymin >= pymin and hymax <= pymax:
                helmet_detected = True
                break  # ถ้ามีหมวกกันน็อคแล้ว ไม่ต้องเช็คต่อ

        if not helmet_detected:
            current_time = time.time()
            if current_time - last_save_time >= 2:  # ป้องกันการบันทึกซ้ำบ่อยเกินไป
                person_crop = frame[pymin:pymax, pxmin:pxmax]  # ครอปเฉพาะ motorcyclist
                save_path = f"storage/no_helmet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, person_crop)
                print(f"Saved person without helmet: {save_path}")
                last_save_time = current_time

    # Display the frame
    cv2.imshow("Helmet Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
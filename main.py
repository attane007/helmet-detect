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

def check_overlap(box1, box2):
    """Check if two boxes overlap (IoU method)."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the intersection coordinates
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    # Calculate intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

# Open the camera
cap = cv2.VideoCapture(selected_cam)

last_save_time = 0

helmet_boxes = []  # To store helmet bounding boxes
person_boxes = []  # To store person bounding boxes
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLO
    results = model.predict(frame)
    detections = results.pandas().xyxy[0]

    license_plate_detect = False


    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, name, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
        print(name)

        if name == 'motorcyclist':
            person_boxes.append((xmin, ymin, xmax, ymax))  # Save the bounding box for the person
            color = (0, 0, 255)  # Red for person
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Person', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if name == 'helmet':
            helmet_boxes.append((xmin, ymin, xmax, ymax))  # Save the bounding box for the helmet
            color = (0, 255, 0)  # Green for helmet
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'Helmet', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if name == 'license_plate':
            color = (255, 0, 0)  # Blue for license plate
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, 'license_plate', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Loop through each person and check if any helmet overlaps
    for person_box in person_boxes:
        helmet_detected = False
        for helmet_box in helmet_boxes:
            iou = check_overlap(person_box, helmet_box)
            if iou >= 0.1:  # If IoU is above 10%, consider that person has a helmet
                helmet_detected = True
                break

        if not helmet_detected:  # Save the image if person does not have a helmet
            current_time = time.time()
            if current_time - last_save_time >= 2:  # Save image every 1 second to avoid excessive saving
                # Crop the person's image (bounding box of person)
                xmin, ymin, xmax, ymax = person_box
                person_image = frame[ymin:ymax, xmin:xmax]  # Crop the person's region

                # Save the cropped person image
                full_frame_filename = f"storage/no_helmet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(full_frame_filename, person_image)
                print(f"Saved person without helmet: {full_frame_filename}")
                last_save_time = current_time
                helmet_boxes = []  # reset helmet bounding boxes
                person_boxes = []  # reset person bounding boxes

    # Display the frame
    cv2.imshow("Helmet Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
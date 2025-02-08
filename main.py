import cv2
import torch
from datetime import datetime
import os
import time
import paho.mqtt.client as mqtt
import warnings

# ตั้งค่า MQTT Broker และ ThingSpeak
MQTT_BROKER = "mqtt3.thingspeak.com"  # หรือ "broker.hivemq.com"
MQTT_PORT = 8883
WRITE_API_KEY = "ABCDEF1234567890" # ใช้สำหรับ ThingSpeak
CHANNEL_ID = "xxxx" # ใช้สำหรับ ThingSpeak
MQTT_USERNAME = "xxxxx"
MQTT_PASSWORD = "xxxxx"
MQTT_TOPIC = f"channels/{CHANNEL_ID}/publish/{WRITE_API_KEY}"

# ฟังก์ชันส่ง MQTT Message
def send_mqtt_message(message):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()  # ใช้สำหรับ HiveMQ Cloud ที่ต้องการ TLS
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    message = f"field1={message}"
    client.publish(MQTT_TOPIC, message)
    client.disconnect()

warnings.simplefilter("ignore", category=FutureWarning)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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

        if name and (time.time() - last_save_time > 1):  # ส่ง MQTT ทุกๆ 1 วินาที
            print(name)
            send_mqtt_message(1)
            last_save_time = time.time()  # อัปเดตเวลาส่งล่าสุด

        color = (0, 0, 255)
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

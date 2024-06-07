from ultralytics import YOLO
import cv2
import cvzone
import math
import time
 
cap = cv2.VideoCapture('natural-form.jpeg') 
cap.set(3, 1280)
cap.set(4, 720)
 
model = YOLO("artifacts-detect.pt")
classNames = ["artifact", "natural-formation"]
 
prev_frame_time = 0
 
while True:
    ret, img = cap.read()
    if not ret:
        break
    
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_frame_time)
    prev_frame_time = curr_time

    print(f"FPS: {fps}")
 
    cv2.imshow("Image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

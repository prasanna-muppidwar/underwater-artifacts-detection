import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import numpy as np
from skimage import exposure, color

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load YOLO model
model = YOLO("artifacts-detect.pt")
classNames = ["artifact", "natural-formation"]

# Variables for frame rate calculation
prev_frame_time = 0

# Parameters for visual odometry
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
color = np.random.randint(0, 255, (100, 3))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enhance the image for underwater visibility
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # YOLO object detection
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

    # Visual Odometry
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(img, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Calculate frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_frame_time)
    prev_frame_time = curr_time

    print(f"FPS: {fps}")

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

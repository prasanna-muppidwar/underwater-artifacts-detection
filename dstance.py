import cv2
import numpy as np

# Distance from camera to object (known object) measured in centimeters
Known_distance = 76.2

# Width of the known object in the real world in centimeters
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# Focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects using YOLO
def detect_objects(img, net, outputLayers):        
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

# Get bounding box coordinates
def get_box_dimensions(outputs, height, width):
    boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, class_ids, confidences

# Measure distance to object
def measure_distance(frame, net, output_layers, classes, known_width, focal_length):
    height, width, _ = frame.shape
    outputs = detect_objects(frame, net, output_layers)
    boxes, class_ids, confidences = get_box_dimensions(outputs, height, width)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label = str(classes[class_ids[i]])
        distance = Distance_finder(focal_length, known_width, w)
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, f"{label} {round(distance, 2)} cm", (x, y - 10), fonts, 0.6, RED, 2)
    return frame

# Load YOLO model
net, classes, output_layers = load_yolo()

# Reference image to find focal length
ref_image = cv2.imread("input.jpg")
ref_image_height, ref_image_width, _ = ref_image.shape
outputs = detect_objects(ref_image, net, output_layers)
boxes, class_ids, confidences = get_box_dimensions(outputs, ref_image_height, ref_image_width)
ref_image_object_width = boxes[0][2] if boxes else 0  # Width of the first detected object

# Calculate the focal length
focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_object_width)
print(f"Focal Length Found: {focal_length_found}")

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = measure_distance(frame, net, output_layers, classes, Known_width, focal_length_found)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

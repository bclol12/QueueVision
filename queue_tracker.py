import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)

# Get frame size first
ret, test_frame = cap.read()
h, w = test_frame.shape[:2]

stalls = [
    "Food Hut",
    "Sedap",
    "Sweet Hut",
]

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 165, 255),
]

# Output path: saves counts.json next to this script
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counts.json")

def make_zones(w, h, n):
    zw = w // n
    zones = {}
    for i, name in enumerate(stalls):
        x1 = i * zw
        x2 = x1 + zw if i < n - 1 else w
        zones[name] = (x1, 0, x2, h)
    return zones

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def in_zone(cx, cy, zone):
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def is_closed(frame, zone):
    """Detect if stall is closed by checking if the zone is grey/dark"""
    zx1, zy1, zx2, zy2 = zone
    roi = frame[zy1:zy2, zx1:zx2]

    # Convert to HSV and check saturation — grey has very low saturation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_brightness = np.mean(hsv[:, :, 2])

    # Low saturation + mid brightness = grey barrier
    return avg_saturation < 30 and 40 < avg_brightness < 200

# Build zones once
zones = make_zones(w, h, len(stalls))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False, conf=0.6, iou=0.5)
    annotated = results[0].plot()

    # Count people per zone
    counts = {name: 0 for name in stalls}

    for box in results[0].boxes.xyxy.tolist():
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        # Skip if box is too small (noise) or too big
        if box_w < 30 or box_h < 60:
            continue
        if box_w > w * 0.4 or box_h > h * 0.9:
            continue

        cx, cy = get_center(box)
        for name, zone in zones.items():
            if in_zone(cx, cy, zone):
                counts[name] += 1

    # Draw zones, counts, and closed detection
    for i, (name, zone) in enumerate(zones.items()):
        zx1, zy1, zx2, zy2 = zone
        color = colors[i]
        closed = is_closed(frame, zone)

        # Draw zone border
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), color, 2)

        # Draw dark background bar at bottom of each zone for readability
        bar_y1 = zy2 - 70
        bar_y2 = zy2
        cv2.rectangle(annotated, (zx1, bar_y1), (zx2, bar_y2), (0, 0, 0), -1)

        if closed:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2 - 70), (80, 80, 80), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, name[:8], (zx1 + 4, bar_y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, "CLOSED", (zx1 + 4, bar_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, name[:8], (zx1 + 4, bar_y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(annotated, f"{counts[name]}pax", (zx1 + 4, bar_y1 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.namedWindow("QueueVision", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("QueueVision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("QueueVision", annotated)

    # Write counts to JSON (atomic write)
    output = {}
    for name, zone in zones.items():
        if is_closed(frame, zone):
            output[name] = "CLOSED"
        else:
            output[name] = counts.get(name, 0)

    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f)
    os.replace(tmp_path, OUTPUT_PATH)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

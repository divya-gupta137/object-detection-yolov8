
import cv2
from ultralytics import YOLO
import random

# -------------------------
# Load YOLO models
# -------------------------
person_model = YOLO("yolov8n.pt")    # Person detection
obstacle_model = YOLO("yolov8n.pt")  # Obstacle detection

# -------------------------
# BLE scanner 
# -------------------------
class BLEScanner:
    def __init__(self, target_device_name="UserPhone"):
        self.target_name = target_device_name

    def get_mock_signal(self):
        """
        Simulate BLE RSSI signal for testing.
        Returns: {'rssi': value, 'direction': 'FORWARD/LEFT/RIGHT'}
        """
        rssi = random.randint(-80, -40)
        if rssi > -50:
            direction = "FORWARD"
        elif rssi > -65:
            direction = "LEFT"
        else:
            direction = "RIGHT"
        return {"rssi": rssi, "direction": direction}

# -------------------------
# Person detection function
# -------------------------
def detect_person(frame):
    frame_resized = cv2.resize(frame, (640, 480))
    results = person_model.predict(source=frame_resized, device="cpu", conf=0.3)
    person_boxes = []

    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # Class 0 = 'person'
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            person_boxes.append({"box": (x1, y1, x2, y2), "center": (cx, cy)})
    return frame_resized, person_boxes

# -------------------------
# Obstacle detection function
# -------------------------
def detect_obstacles(frame):
    frame_resized = cv2.resize(frame, (640, 480))
    results = obstacle_model.predict(source=frame_resized, device="cpu", conf=0.3)
    obstacle_positions = []

    for box in results[0].boxes:
        if int(box.cls[0]) != 0:  # Ignore person
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            obstacle_positions.append((cx, cy))
    return frame_resized, obstacle_positions

# -------------------------
# Suggest movement based on vision
# -------------------------
def suggest_movement(person_center, obstacles, frame_width=640):
    cx, cy = person_center
    frame_center = frame_width // 2

    # Basic forward/left/right suggestion
    if cx < frame_center - 50:
        dir = "LEFT"
    elif cx > frame_center + 50:
        dir = "RIGHT"
    else:
        dir = "FORWARD"

    # Check for obstacles in horizontal path
    blocked = any(dir == "FORWARD" and abs(ox - cx) < 50 for (ox, oy) in obstacles)
    if blocked:
        dir = "AVOID_OBSTACLE"
    return dir

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Image path (replace with live camera feed later)
    img_path = r"suitcase_project\road_img2.webp"
    frame = cv2.imread(img_path)
    if frame is None:
        print("❌ Could not read image. Check path!")
        exit()

    # 1️⃣ Vision-based detection
    frame, person_boxes = detect_person(frame)
    frame, obstacles = detect_obstacles(frame)
    print("Obstacle positions:", obstacles)

    # 2️⃣ BLE signal (mock)
    scanner = BLEScanner()
    ble_signal = scanner.get_mock_signal()
    print("BLE Signal:", ble_signal)

    # 3️⃣ Match BLE signal to closest person to center (assume target)
    if person_boxes:
        # Compute distance of each person center to frame center
        frame_center_x = frame.shape[1] // 2
        closest_person = min(person_boxes, key=lambda p: abs(p["center"][0] - frame_center_x))
        # Draw rectangle only around this person
        x1, y1, x2, y2 = closest_person["box"]
        cx, cy = closest_person["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Suggest movement for this person based on BLE + obstacles
        vision_dir = suggest_movement(closest_person["center"], obstacles)
        # Combine with BLE: give priority to BLE
        final_direction = ble_signal["direction"]
    else:
        final_direction = None

    print("Final movement direction (combined BLE + vision):", final_direction)

    # Save output
    output_path = r"suitcase_project\output_combined_ble_single_person.jpg"
    cv2.imwrite(output_path, frame)
    print(f"✅ Output saved as '{output_path}'")

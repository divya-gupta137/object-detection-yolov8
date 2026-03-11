
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_obstacles(frame):

    frame_resized = cv2.resize(frame, (640, 480))

    results = model.predict(source=frame_resized, device="cpu", conf=0.3)

    suggestions = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_resized, f"ID:{cls_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            suggestions.append("STOP")

    if not suggestions:
        suggestions.append("CLEAR")

    return frame_resized, suggestions

if __name__ == "__main__":
    img_path = r"suitcase_project\road_img1.jpg"

    frame = cv2.imread(img_path)
    if frame is None:
        print("Could not read image. Check path!")
        exit()

    frame, suggestions = detect_obstacles(frame)
    print("Obstacle suggestions:", suggestions)

    cv2.imwrite("output_obstacles.jpg", frame)
    print("Output saved as 'output_obstacles.jpg'")

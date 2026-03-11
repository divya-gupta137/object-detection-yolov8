import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def track_person(frame):
    frame_resized = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame_resized, device="cpu", conf=0.3)
    directions = []
    for box in results[0].boxes:
        if int(box.cls[0]) == 0: 
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame_resized, (cx, cy), 5, (0, 255, 255), -1)

            frame_center = frame_resized.shape[1] // 2
            if cx < frame_center - 50:
                directions.append("LEFT")
            elif cx > frame_center + 50:
                directions.append("RIGHT")
            else:
                directions.append("FORWARD")

    return frame_resized, directions

if __name__ == "__main__":
    img_path = r"C:\Users\Divya\Desktop\OpenCV\suitcase_project\person_img1.jpg"

    frame = cv2.imread(img_path)
    if frame is None:
        print("Could not read image. Check path!")
        exit()

    frame, directions = track_person(frame)
    print("Movement suggestions:", directions)

    output_path = r"C:\Users\Divya\Desktop\OpenCV\suitcase_project\output_detected.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Output saved as '{output_path}'")

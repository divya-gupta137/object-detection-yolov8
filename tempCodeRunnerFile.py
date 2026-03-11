
import cv2
from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Nano version for real-time webcam detection

def track_person(frame):
    """
    Detects humans in a frame and suggests movement directions.

    Args:
        frame: A single frame/image from video feed (BGR format)

    Returns:
        frame: Frame with rectangles and center points drawn
        directions: List of movement suggestions ("LEFT", "RIGHT", "FORWARD")
    """
    # Run YOLO detection
    results = model(frame)

    # Extract bounding boxes
    boxes = results[0].boxes
    directions = []

    for box in boxes:
        if box.cls == 0:  # Class 0 = 'person' in COCO dataset
            x1, y1, x2, y2 = map(int, box.xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw rectangle and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # Decide movement direction
            frame_center = frame.shape[1] // 2
            if cx < frame_center - 50:
                directions.append("LEFT")
            elif cx > frame_center + 50:
                directions.append("RIGHT")
            else:
                directions.append("FORWARD")

    return frame, directions

if __name__ == "__main__":
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, directions = track_person(frame)
        print("Movement suggestions:", directions)

        cv2.imwrite("YOLO Person Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

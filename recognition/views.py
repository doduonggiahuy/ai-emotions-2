from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("recognition/yolov8n-face.pt")  # Đường dẫn đến model YOLOv8

def video_stream():
    # Mở camera (0 là camera mặc định)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 Face Detection
        results = model.predict(frame, conf=0.5)

        # Vẽ bounding box lên khuôn mặt
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Encode frame thành JPEG
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    cap.release()

def index(request):
    return render(request, "recognition/index.html")

def video_feed(request):
    return StreamingHttpResponse(
        video_stream(), content_type="multipart/x-mixed-replace; boundary=frame"
    )

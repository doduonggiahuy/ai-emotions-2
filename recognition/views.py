from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
face_model = YOLO("recognition/yolov8n-face.pt")
emotion_model = YOLO("recognition/best_CLS2.pt")

def video_stream():
    # Mở camera (0 là camera mặc định)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # Nếu không mở được camera, kết thúc generator
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán khuôn mặt
        face_results = face_model.predict(frame, conf=0.5)
        face_result = face_results[0]

        if not face_result.boxes:
            continue

        for box in face_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                continue

            resized_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            emotion_result = emotion_model.predict(resized_face_rgb, conf=0.5, verbose=False)
            probs = emotion_result[0].probs

            if probs is not None:
                cls_id = probs.top1
                emotion_label = emotion_result[0].names[cls_id]
            else:
                emotion_label = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame thành JPEG
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    cap.release()

def index(request):
    # Hiển thị trang chính với kiểm tra lỗi camera
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render(request, "recognition/index.html", {"error": True})
        cap.release()
    except Exception as e:
        return render(request, "recognition/index.html", {"error": True})

    return render(request, "recognition/index.html", {"error": False})

def video_feed(request):
    stream = video_stream()
    if stream is None:
        return render(request, "recognition/index.html", {"error": True})
    return StreamingHttpResponse(   
        stream, content_type="multipart/x-mixed-replace; boundary=frame"
    )

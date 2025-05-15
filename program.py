import cv2
import tensorflow as tf
from gtts import gTTS
import easyocr
import torch
import numpy as np
import os
# Load pre-trained CNN model
model = tf.keras.models.load_model('model.h5')
# Initialize EasyOCR for text recognition
ocr = easyocr.Reader(['vi'])
# Load YOLOv5 model for traffic sign detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Initialize webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    results = yolo_model(frame)  # Phát hiện đối tượng bằng YOLOv5
    bbox = results.pandas().xyxy[0]  # xmin, ymin, xmax, ymax
    for index, row in bbox.iterrows():
        if row['name'] == 'traffic sign':
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cropped_image = frame[y1:y2, x1:x2]  # Cắt vùng chứa biển báo
            img_resized = cv2.resize(cropped_image, (32, 32))  # Resize về 32x32
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            img_norm = img_gray / 255.0  # Chuẩn hóa pixel về [0, 1]
            img_input = np.expand_dims(img_norm, axis=-1)  # Thêm kênh màu (grayscale)
            prediction = model.predict(img_input)  # Dự đoán bằng mô hình CNN
            label = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
            result = ocr.read_text(cropped_image)
            text = result[0][1] if result else ""  # Lấy văn bản từ kết quả OCR
            tts = gTTS(text=f"Biển báo {text}", lang='vi')  # Chuyển văn bản thành giọng nói tiếng Việt
            tts.save("output.mp3")  # Lưu file mp3
            os.system("start output.mp3")  # Mở file để phát âm thanh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung xanh quanh biển báo
            cv2.putText(frame, f"Label: {label}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Ghi nhãn lên khung
            # Hiển thị khung hình đã xử lý
            cv2.imshow('Traffic Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

B1: Cài đặt Visual Studio Code & Python

BƯỚC 2: Tạo môi trường ảo (Virtual Environment)
    Trong thư mục dự án, mở terminal (trong VS Code hoặc CMD/PowerShell), chạy:
python -m venv venv
.\venv\Scripts\activate                           (kích hoạt môi trường)

pip install opencv-python
pip install tensorflow
pip install easyocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # nếu dùng GPU
pip install gTTS
    (Cài YOLOv5)
git clone https://github.com/Duybeo2003/TrafficSigns.git
cd TrafficSigns
pip install -r requirements.txt

BƯỚC 5: Cấu hình Python interpreter trong VS Code
Nhấn (Ctrl + Shift + P) → tìm **Python: Select Interpreter**
Chọn đường dẫn đến venv (ví dụ: .venv\Scripts\python.exe)
----------------------------------------------------------------------------------------------------------------------------------
Code huấn luyện CNN: TrafficSigh_Main.py
CNN được huấn luyện lưu thành model.h5
Code Chạy Cam nhận diện biển báo(đang lỗi): program.py

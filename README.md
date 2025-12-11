# 🧠 Hand-drawn Digit Recognition AI

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Implementation-013243?logo=numpy&logoColor=white)

Ứng dụng web nhận diện chữ số viết tay (0-9) sử dụng thuật toán **Softmax Regression**. Điểm đặc biệt của dự án là thuật toán lõi được xây dựng hoàn toàn thủ công bằng **NumPy** để tối ưu hóa tính toán ma trận, không sử dụng các hàm có sẵn của framework Deep Learning.

## Tính năng nổi bật

* **Vẽ tương tác:** Hỗ trợ vẽ mượt mà trên Canvas (Desktop & Mobile).
* **Giao diện thích ứng (Adaptive UI):** Tự động đổi Theme (Pastel/Dark/Teal) theo Model được chọn.
* **Đa mô hình (Multi-Model):**
    1.  **Pixel Model:** Dựa trên độ đậm nhạt pixel gốc.
    2.  **Sobel Model:** Sử dụng thuật toán phát hiện cạnh (Edge Detection).
    3.  **Block Avg Model:** Nén ảnh trung bình khối để tăng tốc độ.
* **Trực quan hóa:** Hiển thị biểu đồ xác suất dự đoán thời gian thực.

## Demo

Xem video demo chi tiết tại: [YouTube Link](https://www.youtube.com/watch?v=XAMZ_AspcHE)

## Cài đặt & Chạy thử

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/USERNAME/REPO-NAME.git](https://github.com/USERNAME/REPO-NAME.git)
    cd REPO-NAME
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chạy Backend:**
    ```bash
    python backend/app.py
    ```

4.  **Mở Frontend:**
    Mở file `frontend/index.html` trên trình duyệt.

## Cấu trúc thư mục

```text
├── application/
│   ├── backend/
│   │   └── app.py                 # Flask Server & API Entry Point
│   └── frontend/
│       ├── index.html             # Giao diện người dùng
│       ├── script.js              # Logic vẽ Canvas & Call API
│       └── style.css              # Định dạng giao diện & Themes
├── models/
│   ├── model_function1.npz        # Trọng số Pixel Model
│   ├── model_function2.npz        # Trọng số Sobel Model
│   └── model_function3.npz        # Trọng số Block Avg Model
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Phân tích & Xử lý dữ liệu
│   └── 02_modeling.ipynb          # Huấn luyện mô hình
├── src/
│   ├── preprocessing.py           # Module xử lý ảnh đầu vào
│   └── utils.py                   # Các hàm tiện ích chung
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── README.md                      # Tài liệu dự án
```

## Thành viên nhóm
23127084 - Dương Thành Lộc
23127104 - Nguyễn Bình Minh Phương
23127221 - Nguyễn Tiến Luật
23127250 - Trần Hồng Phương
23127281 - Đặng Nghi Văn


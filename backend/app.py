import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- 1. LOAD 3 BỘ TRỌNG SỐ ---
models = {}

def load_weights(name, filepath):
    try:
        data = np.load(filepath)
        models[name] = {'W': data['W'], 'b': data['b']}
        print(f"✅ Đã tải {name}: W shape {models[name]['W'].shape}")
    except:
        print(f"⚠️ Không tìm thấy {filepath}")

print("⏳ Đang khởi động hệ thống...")
load_weights('pixel', 'weights_pixel.npz') # 784 features
load_weights('sobel', 'weights_sobel.npz') # 1568 features
load_weights('block', 'weights_block.npz') # 196 features

# --- 2. CÁC HÀM TOÁN HỌC CHUNG ---
def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- 3. CÁC HÀM XỬ LÝ ẢNH (Preprocessing) ---

# Xử lý chung: Mở ảnh -> Xám -> Resize -> Mảng -> Đảo màu -> Chuẩn hóa
def preprocess_common(image_file):
    img = Image.open(image_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255.0 - img_array # Đảo màu (Trắng trên nền đen)
    img_norm = img_array / 255.0  # Về 0-1
    return img_norm

# [Logic 1] Model Pixel
def process_pixel(img_norm):
    return img_norm.reshape(1, -1) # (1, 784)

# [Logic 2] Model Sobel
def process_sobel(img_norm):
    img_float = img_norm.astype(np.float32)
    sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = edges / (edges.max() + 1e-8)
    
    img_stacked = np.stack([img_norm, edges], axis=-1)
    return img_stacked.reshape(1, -1) # (1, 1568)

# [Logic 3] Model Block Averaging (Mới thêm)
def process_block_avg(img_norm, block_size=2):
    # Input: (28, 28)
    H, W = img_norm.shape
    new_h, new_w = H // block_size, W // block_size
    
    # Cắt cho chẵn (nếu cần)
    valid_h, valid_w = new_h * block_size, new_w * block_size
    img_cropped = img_norm[:valid_h, :valid_w]
    
    # Reshape thành (14, 2, 14, 2)
    # Sau đó tính mean ở các chiều block (axis 1 và 3)
    reshaped = img_cropped.reshape(new_h, block_size, new_w, block_size)
    img_blocked = reshaped.mean(axis=(1, 3)) # Kết quả ra (14, 14)
    
    return img_blocked.reshape(1, -1) # (1, 196)

# --- 4. API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    model_type = request.form.get('model_type', 'pixel')
    
    if model_type not in models:
        return jsonify({'error': f"Model '{model_type}' chưa được tải (kiểm tra file .npz)"}), 400

    file = request.files['file']
    
    try:
        # Bước 1: Xử lý chung
        img_norm = preprocess_common(file)
        
        # Bước 2: Xử lý đặc trưng riêng
        if model_type == 'sobel':
            vector = process_sobel(img_norm)
        elif model_type == 'block':
            vector = process_block_avg(img_norm)
        else:
            vector = process_pixel(img_norm)
            
        # Bước 3: Tính toán
        W = models[model_type]['W']
        b = models[model_type]['b']
        
        # Kiểm tra khớp shape (Tránh lỗi nhân ma trận)
        if vector.shape[1] != W.shape[0]:
             return jsonify({'error': f"Lệch Shape: Ảnh {vector.shape}, Model {W.shape}"}), 500

        logits = np.dot(vector, W) + b
        probs = softmax(logits)[0]
        prediction = np.argmax(probs)
        
        return jsonify({
            'digit': int(prediction),
            'probabilities': probs.tolist(),
            'model_used': model_type,
            'feature_count': vector.shape[1]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

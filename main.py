from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
import base64
from PIL import Image

# Import các hàm logic từ file logic.py
from logic import load_model, process_image_logic, MODEL_PATH

# --- Tải Model một lần duy nhất khi server khởi động ---
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file model. {e}")
    print("Vui lòng đảm bảo file 'best.pt' nằm ở: ", MODEL_PATH)
    model = None # Sẽ báo lỗi khi API được gọi
# ----------------------------------------------------

app = FastAPI(title="Logo & Biển Số API")

# --- Cấu hình CORS ---
# Cho phép giao diện web (chạy ở domain khác) có thể gọi API này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả method (POST, GET...)
    allow_headers=["*"], # Cho phép tất cả header
)

# --- Định nghĩa Dữ liệu trả về (Response Model) ---
class ProcessResponse(BaseModel):
    filename: str
    logo_name: str
    confidence_text: str
    processed_image_base64: str # Ảnh đã xử lý, được encode Base64

# --- Endpoint gốc (chỉ để kiểm tra server) ---
@app.get("/")
def read_root():
    return {"message": "Chào mừng bạn đến với API xử lý ảnh!"}

# --- Endpoint xử lý ảnh (trái tim của API) ---
@app.post("/process-image/", response_model=ProcessResponse)
async def api_process_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Lỗi server: Model chưa được tải.")
        
    # 1. Đọc dữ liệu ảnh từ file upload
    image_bytes = await file.read()
    
    try:
        # 2. Gọi hàm logic để xử lý
        pil_clean, best_logo_name, best_conf_text = process_image_logic(model, image_bytes)
        
        # 3. Chuyển ảnh PIL (kết quả) sang Base64 để gửi qua JSON
        buffered = io.BytesIO()
        pil_clean.save(buffered, format="JPEG") # Lưu ảnh vào bộ nhớ đệm
        img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 4. Trả về kết quả
        return ProcessResponse(
            filename=file.filename,
            logo_name=best_logo_name,
            confidence_text=best_conf_text,
            processed_image_base64=img_base64_str
        )

    except ValueError as e:
        # Lỗi nếu cv2.imdecode thất bại
        raise HTTPException(status_code=400, detail=f"File ảnh không hợp lệ: {e}")
    except Exception as e:
        # Các lỗi chung khác
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý server: {e}")

# --- Dùng để chạy server trực tiếp ---
if __name__ == "__main__":
    print(f"Khởi chạy server tại http://127.0.0.1:8000")
    print("Truy cập http://127.0.0.1:8000/docs để xem tài liệu API")
    uvicorn.run(app, host="127.0.0.1", port=8000)

import os
import json
import time
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv() 
# --- CẤU HÌNH HỆ THỐNG ---
API_KEY = os.getenv("GEMINI_API_KEY2")  # Nhập API Key của bạn
INPUT_DIR = "datasets/data_gen/source" # Thư mục chứa 100 file .txt thô
OUTPUT_FILE = "datasets/data_gen/silver_dataset.json"

# --- PROMPT KỸ THUẬT (SYSTEM INSTRUCTION) ---
# Prompt này áp dụng kỹ thuật Constraint Formatting và Role-prompting
SYSTEM_INSTRUCTION = """
Bạn là một chuyên gia AI phân tích dữ liệu lâm sàng (Clinical NLP Expert). 
Nhiệm vụ của bạn là đọc các đoạn hồ sơ bệnh án tiếng Việt và trích xuất tất cả các thực thể y khoa (Medical Entities) theo đúng 4 nhãn dưới đây.

[ĐỊNH NGHĨA NHÃN & QUY TẮC BẮT BUỘC]:
1. "THUỐC": Tên thuốc, hoạt chất, biệt dược, vắc-xin, vitamin. 
   - TUYỆT ĐỐI KHÔNG trích xuất: Dụng cụ y tế (ống thông, kim tiêm, dao mổ), thiết bị, phương pháp điều trị cơ học (hô hấp nhân tạo, phẫu thuật, xạ trị).
2. "TÊN_XÉT_NGHIỆM": Tên các kỹ thuật cận lâm sàng, chẩn đoán hình ảnh, xét nghiệm sinh hóa, huyết học, thăm dò chức năng (vd: siêu âm, CT scan, MRI, điện tâm đồ, nội soi).
3. "CHẨN_ĐOÁN": Tên bệnh lý, hội chứng, tổn thương, trạng thái bệnh (vd: suy tim, viêm ruột thừa, nhồi máu cơ tim).
4. "TRIỆU_CHỨNG": Các dấu hiệu lâm sàng, cảm nhận của bệnh nhân, biểu hiện bất thường (vd: đau đầu, buồn nôn, khó thở, sốt cao).

[YÊU CẦU ĐẦU RA]:
- Trả về ĐÚNG định dạng JSON với một list chứa các object có 2 key: "text" (chuỗi trích xuất nguyên bản từ văn bản) và "type" (nhãn tương ứng).
- Ví dụ Output chuẩn:
{
  "entities": [
    {"text": "viêm ruột thừa", "type": "CHẨN_ĐOÁN"},
    {"text": "đau bụng", "type": "TRIỆU_CHỨNG"},
    {"text": "siêu âm ổ bụng", "type": "TÊN_XÉT_NGHIỆM"},
    {"text": "Paracetamol", "type": "THUỐC"}
  ]
}
"""

def extract_entities_gemini(client, text_content, max_retries=5):
    """Gọi Gemini API để trích xuất thực thể với cơ chế chống nghẽn (Exponential Backoff)"""
    prompt = f"Trích xuất thực thể từ văn bản sau:\n\n{text_content}"
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0, # Đặt bằng 0 để đảm bảo tính Deterministic (không ảo giác)
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text).get("entities", [])
            
        except Exception as e:
            wait_time = (2 ** attempt) * 2
            print(f"  [!] Lỗi API (Thử lại {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return []

def align_entities_to_text(raw_text, extracted_entities):
    """
    Thuật toán map entity text trở lại văn bản gốc để lấy chính xác tọa độ start, end.
    Ưu tiên map các chuỗi dài trước để tránh lỗi overlap (chuỗi con).
    """
    if not extracted_entities:
        return []

    # Lọc trùng và sắp xếp độ dài giảm dần để match chuỗi dài trước
    unique_entities = { (e['text'].strip(), e['type']) for e in extracted_entities if e.get('text') }
    sorted_entities = sorted(list(unique_entities), key=lambda x: len(x[0]), reverse=True)
    
    final_mapped_entities = []
    # Sử dụng mảng boolean để track các ký tự đã được map, tránh 1 từ bị gán 2 nhãn
    mapped_chars = [False] * len(raw_text)

    for ent_text, ent_type in sorted_entities:
        # Dùng regex escape để an toàn với các ký tự đặc biệt (+, -, (, )) trong bệnh án
        pattern = re.compile(re.escape(ent_text), re.IGNORECASE)
        for match in pattern.finditer(raw_text):
            start_idx = match.start()
            end_idx = match.end()
            
            # Kiểm tra xem khoảng này đã bị entity dài hơn map chưa
            if not any(mapped_chars[start_idx:end_idx]):
                final_mapped_entities.append({
                    "text": raw_text[start_idx:end_idx],
                    "type": ent_type,
                    "start": start_idx,
                    "end": end_idx
                })
                # Đánh dấu khoảng này đã được map
                for i in range(start_idx, end_idx):
                    mapped_chars[i] = True

    # Trả về mảng đã sắp xếp theo thứ tự xuất hiện trong văn bản
    return sorted(final_mapped_entities, key=lambda x: x['start'])

def process_test_files():
    client = genai.Client(api_key=API_KEY)
    all_documents = []
    
    if not os.path.exists(INPUT_DIR):
        print(f"Thư mục {INPUT_DIR} không tồn tại!")
        return

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"Bắt đầu xử lý {len(txt_files)} files...")

    for idx, filename in enumerate(txt_files):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"[{idx + 1}/{len(txt_files)}] Đang xử lý: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
            
        if not raw_text:
            continue
            
        # 1. Gọi LLM trích xuất (chỉ lấy text và type)
        extracted_entities = extract_entities_gemini(client, raw_text)
        
        # 2. Python căn gióng tọa độ (start, end)
        mapped_entities = align_entities_to_text(raw_text, extracted_entities)
        
        all_documents.append({
            "file_name": filename,
            "text": raw_text,
            "entities": mapped_entities
        })
        
        time.sleep(1) # Nghỉ ngắn tránh rate limit

    # Lưu kết quả Silver Dataset
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Hoàn tất! Silver Dataset đã lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_test_files()
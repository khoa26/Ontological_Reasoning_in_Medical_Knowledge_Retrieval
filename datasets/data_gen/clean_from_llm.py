import json
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---
API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_FILE = "datasets/data_gen/f_synthetic_dataset.json" # File đã qua bước làm sạch bằng code trước đó
OUTPUT_FILE = "datasets/data_gen/final_semantic_dataset_2.json"

BATCH_SIZE = 5      # Số bệnh án gửi trong mỗi request
MAX_RETRIES = 5     # Số lần thử lại tối đa

# --- ĐỊNH NGHĨA STRUCTURED OUTPUT BẰNG PYDANTIC ---
class CleanedEntity(BaseModel):
    text: str = Field(description="Đoạn text của thực thể được giữ lại")
    type: str = Field(description="Nhãn của thực thể (vd: TRIỆU_CHỨNG, CHẨN_ĐOÁN...)")

class CleanedRecord(BaseModel):
    text: str = Field(description="Văn bản gốc của bệnh án (giữ nguyên, không sửa)")
    entities: list[CleanedEntity] = Field(description="Danh sách các thực thể SAU KHI đã lọc sạch trùng lặp ngữ nghĩa")

# --- PROMPT HƯỚNG DẪN LLM ---
SYSTEM_INSTRUCTION = """
Bạn là một chuyên gia hiệu đính dữ liệu y khoa (Medical Data Reviewer).
Tôi sẽ cung cấp cho bạn một danh sách các bệnh án dưới định dạng JSON. Mỗi bệnh án gồm văn bản gốc ("text") và danh sách các thực thể y khoa ("entities").

Nhiệm vụ của bạn là LỌC SẠCH danh sách "entities" theo các quy tắc sau:
1. Gom nhóm ngữ nghĩa (Semantic Deduplication): Nếu có nhiều thực thể mang CÙNG MỘT Ý NGHĨA lâm sàng (dù cách viết khác nhau), chỉ giữ lại MỘT thực thể duy nhất đại diện tốt nhất (thường là cụm đầy đủ nhất).
   - Ví dụ: "THIẾU MEN G6PD", "thiếu hụt men G6PD", "bệnh thiếu men G6PD" -> Chỉ giữ lại 1 cái.
   - Ví dụ: "đau ngực", "đau thắt ngực trái", "đau ngực dữ dội" -> Giữ lại "đau thắt ngực trái dữ dội" (nếu có) hoặc 1 cụm mô tả đầy đủ nhất.
2. Giữ nguyên: TUYỆT ĐỐI KHÔNG SỬA nội dung phần "text" gốc của bệnh án.
3. Chỉ lọc bên trong nội bộ của TỪNG bệnh án, không so sánh chéo giữa các bệnh án với nhau.
"""

def clean_batch_with_llm(client, batch_records, batch_idx, max_retries=MAX_RETRIES):
    """
    Gửi 1 batch dữ liệu lên Gemini để lọc trùng lặp ngữ nghĩa.
    Có tích hợp Exponential Backoff.
    """
    input_text = json.dumps(batch_records, ensure_ascii=False, indent=2)
    prompt = f"Hãy dọn dẹp các thực thể bị trùng lặp ngữ nghĩa cho {len(batch_records)} bệnh án dưới đây:\n\n{input_text}"
    
    for attempt in range(max_retries):
        try:
            print(f"  [Batch {batch_idx}] Đang gọi API... (Lần {attempt + 1}/{max_retries})")
            
            # Sử dụng mô hình 2.5-flash
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0, # Temperature = 0 để tránh LLM sáng tạo thêm rác
                    response_mime_type="application/json",
                    response_schema=list[CleanedRecord], # Ép cấu trúc đầu ra
                ),
            )
            
            # Parse kết quả
            cleaned_batch = json.loads(response.text)
            return cleaned_batch
            
        except Exception as e:
            # Tính thời gian chờ (Exponential Backoff: 2s, 4s, 8s, 16s, 32s)
            wait_time = 2 ** attempt
            print(f"  [!] Lỗi ở Batch {batch_idx}: {e}")
            if attempt < max_retries - 1:
                print(f"      -> Chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ THẤT BẠI hoàn toàn ở Batch {batch_idx} sau {max_retries} lần thử.")
                # Trả về dữ liệu gốc của batch này nếu thất bại hoàn toàn để không bị mất data
                return batch_records 

def main():
    if not API_KEY:
        print("❌ LỖI: Chưa cấu hình GEMINI_API_KEY")
        return

    client = genai.Client(api_key=API_KEY)

    # Đọc dữ liệu đầu vào
    if not os.path.exists(INPUT_FILE):
        print(f"❌ LỖI: Không tìm thấy {INPUT_FILE}")
        return
        
    print(f"🔄 Đang tải dữ liệu từ {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_records = len(dataset)
    print(f"📦 Tổng số bệnh án cần xử lý: {total_records}")

    final_dataset = []
    
    # Chia danh sách thành các Batch
    for i in range(0, total_records, BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        batch_idx = (i // BATCH_SIZE) + 1
        
        # Gọi LLM làm sạch batch
        cleaned_batch = clean_batch_with_llm(client, batch, batch_idx)
        final_dataset.extend(cleaned_batch)
        
        # Tạm nghỉ ngắn để giữ an toàn RPM (Requests Per Minute)
        time.sleep(2)

    # Thống kê
    old_entity_count = sum(len(r.get("entities", [])) for r in dataset)
    new_entity_count = sum(len(r.get("entities", [])) for r in final_dataset)
    removed = old_entity_count - new_entity_count

    # Ghi file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print("🎉 HOÀN TẤT QUÁ TRÌNH LỌC NGỮ NGHĨA (LLM SEMANTIC CLEANER)")
    print(f"📊 Số bệnh án sau xử lý: {len(final_dataset)}")
    print(f"🔍 Số thực thể ban đầu:  {old_entity_count}")
    print(f"✨ Số thực thể giữ lại:  {new_entity_count}")
    print(f"🗑️ Lọc thành công:       {removed} thực thể (trùng lặp ngữ nghĩa)")
    print(f"📂 File lưu tại:         {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
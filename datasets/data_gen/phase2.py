import os
import json
import time
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

# Tải biến môi trường
load_dotenv() 

# --- CẤU HÌNH HỆ THỐNG ---
API_KEY = os.getenv("GEMINI_API_KEY3") 
INPUT_FILE = "datasets/data_gen/base_dataset.json"
OUTPUT_FILE = "datasets/data_gen/synthetic_dataset_2.json"

# Cấu hình số lượng
TOTAL_NEW_SAMPLES_NEEDED = 750  # Tổng số mẫu bạn muốn sinh thêm
SAMPLES_PER_REQUEST = 5         # Số mẫu sinh ra trong mỗi lần gọi API
NUM_SEED_EXAMPLES = 2           # Số mẫu gốc dùng làm mồi (few-shot)

# --- ĐỊNH NGHĨA CẤU TRÚC ĐẦU RA (STRUCTURED OUTPUT) ---
# Việc dùng Pydantic sẽ ép Gemini trả về đúng cấu trúc JSON này, không có Markdown hay rác văn bản.
class MedicalEntity(BaseModel):
    text: str
    type: str

class SyntheticRecord(BaseModel):
    file_name: str
    text: str
    entities: list[MedicalEntity]

# --- PROMPT KỸ THUẬT (SYSTEM INSTRUCTION) ---
SYSTEM_INSTRUCTION = """
Bạn là một chuyên gia AI tạo lập dữ liệu y khoa lâm sàng (Clinical Data Synthesizer).
Nhiệm vụ của bạn là tạo ra các hồ sơ bệnh án giả định mới bằng tiếng Việt, kèm theo việc trích xuất các thực thể y khoa tương ứng.

[YÊU CẦU DỮ LIỆU ĐẦU RA]:
1. Văn phong: Phải mô phỏng chính xác văn phong lâm sàng thực tế (ngắn gọn, có tính chuyên môn, ghi chú của bác sĩ).
2. Nội dung: ĐA DẠNG HÓA các loại bệnh lý (Tim mạch, Hô hấp, Tiêu hóa, Thần kinh, Cơ xương khớp...), thuốc điều trị, xét nghiệm và triệu chứng. Tự sáng tạo nội dung mới mẻ, không sao chép y hệt mẫu gốc.
3. Gán nhãn thực thể (NER): Tuân thủ nghiêm ngặt 5 nhãn:
   - "THUỐC": Tên thuốc, hoạt chất, biệt dược, vắc-xin, vitamin(ví dụ: "Paracetamol", "Ibuprofen","Chlorpheniramine 0.4 MG/ML", "Capsaicin 0.38 MG/ML" ). (TUYỆT ĐỐI KHÔNG trích xuất: Dụng cụ y tế (ống thông, kim tiêm, dao mổ), thiết bị, phương pháp điều trị cơ học (hô hấp nhân tạo, phẫu thuật, xạ trị).)
   - "TÊN_XÉT_NGHIỆM": Tên các kỹ thuật cận lâm sàng, chẩn đoán hình ảnh, xét nghiệm sinh hóa, huyết học, thăm dò chức năng (vd: "siêu âm", "CT scan", "MRI", "điện tâm đồ", "nội soi" , "TWBC", "NEUT% (Tỷ lệ % bạch cầu trung tính)", "LYPH% (Tỷ lệ bạch cầu lympho)").
   - "KẾT_QUẢ_XÉT_NGHIỆM": Kết quả xét nghiệm bệnh nhân thực hiện, bao gồm giá trị và đơn vị của xét nghiệm (vd: "14,43", "76,4", "12,8").
   - "CHẨN_ĐOÁN": Tên bệnh lý, hội chứng, tổn thương, trạng thái bệnh (vd: "suy tim", "viêm ruột thừa", "nhồi máu cơ tim", "bệnh trào ngược dạ dày - thực quản" ).
   - "TRIỆU_CHỨNG": Các dấu hiệu lâm sàng, cảm nhận của bệnh nhân, biểu hiện bất thường (vd: "đau đầu", "buồn nôn", "khó thở", "sốt cao", "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi").
"""

def generate_synthetic_batch(client, seed_samples, num_to_generate, current_iteration, max_retries=3):
    """Sử dụng Few-shot prompting để sinh dữ liệu mới với cấu trúc ép buộc"""
    
    seed_text = json.dumps(seed_samples, ensure_ascii=False, indent=2)
    
    prompt = f"""
    Dưới đây là {len(seed_samples)} mẫu bệnh án chuẩn (Seed Data) kèm theo các thực thể đã được trích xuất:
    {seed_text}

    Dựa vào cấu trúc và văn phong trên, hãy SÁNG TẠO VÀ SINH RA CHÍNH XÁC {num_to_generate} MẪU BỆNH ÁN HOÀN TOÀN MỚI.
    Lưu ý:
    - Giá trị "file_name" hãy đặt theo định dạng "synthetic_batch{current_iteration}_<số_ngẫu_nhiên>.txt".
    - Trích xuất thực thể CHÍNH XÁC chuỗi ký tự có xuất hiện trong phần "text".
    """
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.8, # Temperature cao để đa dạng hóa ca bệnh
                    response_mime_type="application/json",
                    # Ép kiểu dữ liệu trả về theo đúng Schema đã định nghĩa: mảng các SyntheticRecord
                    response_schema=list[SyntheticRecord], 
                ),
            )
            
            # Kết quả trả về lúc này chắc chắn 100% là chuỗi JSON hợp lệ, ta chỉ cần parse
            result_json = json.loads(response.text)
            return result_json
                
        except Exception as e:
            wait_time = (attempt + 1) * 5
            print(f"  [!] Lỗi API (Thử lại {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return []

def main():
    if not API_KEY:
        print("❌ LỖI: Chưa tìm thấy GEMINI_API_KEY trong hệ thống.")
        return

    client = genai.Client(api_key=API_KEY)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ LỖI: Không tìm thấy file {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        silver_data = json.load(f)
        
    print(f"✅ Đã tải {len(silver_data)} mẫu Silver Data làm hạt giống.")
    
    synthetic_dataset = []
    iterations = TOTAL_NEW_SAMPLES_NEEDED // SAMPLES_PER_REQUEST
    
    print(f"🚀 Bắt đầu sinh dữ liệu: {iterations} vòng lặp (Mỗi vòng {SAMPLES_PER_REQUEST} mẫu)...")

    for i in range(iterations):
        print(f"\n[Vòng {i + 1}/{iterations}] Đang sinh cụm {SAMPLES_PER_REQUEST} mẫu mới...")
        
        # Chọn ngẫu nhiên mẫu mồi để mỗi vòng mô hình có cảm hứng sinh ca bệnh khác nhau
        seed_samples = random.sample(silver_data, NUM_SEED_EXAMPLES)
        
        new_samples = generate_synthetic_batch(client, seed_samples, SAMPLES_PER_REQUEST, current_iteration=(i+1))
        
        if new_samples:
            synthetic_dataset.extend(new_samples)
            print(f"  -> Nhận thành công {len(new_samples)} mẫu. Tổng kho dữ liệu mới: {len(synthetic_dataset)}/{TOTAL_NEW_SAMPLES_NEEDED}")
        else:
            print("  -> Vòng lặp này thất bại, bỏ qua.")
            
        # Nghỉ ngắn giữa các request để đảm bảo an toàn cho RPM (Requests Per Minute)
        time.sleep(3) 

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Lưu toàn bộ dữ liệu mới
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(synthetic_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 HOÀN TẤT! Đã sinh {len(synthetic_dataset)} mẫu dữ liệu mới. Lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
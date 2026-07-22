import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CẤU HÌNH ---
load_dotenv()  # Tự động load file .env

API_KEY = os.getenv("GEMINI_API_KEY")  # Lấy key từ .env

if not API_KEY:
    raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong file .env. Vui lòng kiểm tra lại.")

INPUT_FILE = "datasets/ViMed/vimedner_processed_dataset.json"
OUTPUT_FILE = "datasets/ViMed/vimed_cleaned_dataset.json"
PATCH_DIR = "datasets/ViMed/patches"
BATCH_SIZE = 30

# Prompt tối ưu hóa...
SYSTEM_INSTRUCTION = """
Bạn là một chuyên gia NLP y khoa tiếng Việt. Nhiệm vụ của bạn là rà soát và chuẩn hóa lại danh sách các thực thể (entities) được trích xuất trong văn bản y khoa.
Quy tắc xử lý bắt buộc:
1. Nhãn 'THUỐC': Chỉ chứa tên dược phẩm, hoạt chất, biệt dược cụ thể (ví dụ: amlodipine, aspirin, chlorpheniramine). Lọc bỏ ngay các thiết bị, dụng cụ y tế, thủ thuật hoặc biện pháp điều trị không phải thuốc.
2. Nhãn 'TÊN_XÉT_NGHIỆM': Chỉ chứa tên các kỹ thuật cận lâm sàng, chẩn đoán hình ảnh, xét nghiệm.
3. Nhãn 'CHẨN_ĐOÁN' và 'TRIỆU_CHỨNG': Giữ nguyên nếu đã đúng.
4. Trả về đúng định dạng JSON gốc (danh sách các object) với entities đã được làm sạch.
"""

def split_json_file(input_path, output_dir, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_path):
        print(f"⚠️ Không tìm thấy file gốc: {input_path}")
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    patch_files = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        patch_filename = os.path.join(output_dir, f"patch_{i // batch_size}.json")
        with open(patch_filename, 'w', encoding='utf-8') as pf:
            json.dump(batch_data, pf, ensure_ascii=False, indent=2)
        patch_files.append(patch_filename)

    print(f"📦 Đã chia {len(data)} mẫu thành {len(patch_files)} patch.")
    return patch_files


def clean_patch_with_gemini(client, patch_file_path, max_retries=5):
    with open(patch_file_path, 'r', encoding='utf-8') as f:
        patch_content = f.read()

    prompt = f"""
    Hãy xử lý danh sách dữ liệu JSON y khoa sau đây theo đúng hệ thống quy tắc đã định nghĩa. 
    Chỉ trả về cấu trúc JSON hợp lệ, không kèm bất kỳ lời giải thích nào khác.
    
    Dữ liệu đầu vào:
    {patch_content}
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
            
        except Exception as e:
            wait_time = (2 ** attempt) * 2
            print(f"⚠️ Lỗi lần {attempt + 1}/{max_retries} ({patch_file_path}): {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Chờ {wait_time} giây rồi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"❌ Thất bại sau {max_retries} lần thử.")
                return None


def main():
    client = genai.Client(api_key=API_KEY)

    patch_files = split_json_file(INPUT_FILE, PATCH_DIR, BATCH_SIZE)
    if not patch_files:
        return

    cleaned_all_data = []

    for idx, patch_file in enumerate(patch_files):
        print(f"🔄 Đang xử lý patch {idx + 1}/{len(patch_files)}: {patch_file}")
        
        result = clean_patch_with_gemini(client, patch_file)
        
        if result:
            if isinstance(result, dict):
                values = list(result.values())
                batch_result = values[0] if values and isinstance(values[0], list) else result
            else:
                batch_result = result
                
            if isinstance(batch_result, list):
                cleaned_all_data.extend(batch_result)
            else:
                print(f"⚠️ Kết quả không phải list, dùng dữ liệu gốc.")
                with open(patch_file, 'r', encoding='utf-8') as pf:
                    cleaned_all_data.extend(json.load(pf))
        else:
            print(f"⚠️ Dùng dữ liệu gốc cho patch lỗi.")
            with open(patch_file, 'r', encoding='utf-8') as pf:
                cleaned_all_data.extend(json.load(pf))
        
        time.sleep(1)  # Tránh rate limit

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_all_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Hoàn tất! Tổng mẫu sau khi làm sạch: {len(cleaned_all_data)}")
    print(f"📁 Kết quả: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
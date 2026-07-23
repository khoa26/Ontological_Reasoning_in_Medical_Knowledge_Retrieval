import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN FILE ---
INPUT_FILE = "datasets/data_gen/synthetic_dataset.json" 
OUTPUT_FILE = "datasets/data_gen/f_synthetic_dataset.json"

def clean_entities(entities):
    """
    Hàm xử lý:
    1. Trùng lặp tuyệt đối (Case-insensitive): Xóa các entity giống hệt nhau về text và type sau khi chuyển chữ thường. Giữ lại bản thể gốc đầu tiên.
    2. Bao hàm: Xóa entity có text nằm trong text của entity khác VÀ cùng type (dựa trên chữ thường).
    """
    if not entities:
        return []

    # --- BƯỚC 1: LỌC TRÙNG LẶP TUYỆT ĐỐI (KHÔNG PHÂN BIỆT HOA THƯỜNG) ---
    unique_entities = []
    seen = set()
    
    for ent in entities:
        orig_text = ent.get('text', '').strip()
        orig_type = ent.get('type', '').strip()
        
        if not orig_text:
            continue
            
        # Chuyển về chữ thường để tạo định danh (identifier) so sánh
        lower_text = orig_text.lower()
        lower_type = orig_type.lower()
        identifier = (lower_text, lower_type)
        
        # Nếu chưa từng gặp cặp (text, type) này, tiến hành lưu trữ
        if identifier not in seen:
            seen.add(identifier)
            # Lưu lại text/type gốc để giữ đúng format ban đầu, 
            # kèm thêm trường lower để tiện xử lý so sánh ở bước 2
            unique_entities.append({
                "orig_text": orig_text,
                "orig_type": orig_type,
                "lower_text": lower_text,
                "lower_type": lower_type
            })

    # --- BƯỚC 2: LỌC THỰC THỂ BỊ BAO HÀM (CÙNG TYPE) ---
    # Sắp xếp theo độ dài chuỗi giảm dần (Chuỗi dài ưu tiên giữ lại trước)
    unique_entities.sort(key=lambda x: len(x['orig_text']), reverse=True)

    final_entities = []
    for current_ent in unique_entities:
        is_nested = False
        
        # So sánh với các thực thể (dài hơn) đã được chọn vào danh sách giữ lại
        for kept_ent in final_entities:
            # So sánh hoàn toàn dựa trên các trường chữ thường đã được tạo ở Bước 1
            if (current_ent['lower_text'] in kept_ent['lower_text']) and (current_ent['lower_type'] == kept_ent['lower_type']):
                is_nested = True
                break 
                
        # Nếu không bị bao hàm, đưa vào danh sách kết quả cuối cùng
        if not is_nested:
            final_entities.append(current_ent)

    # --- BƯỚC 3: DỌN DẸP & TRẢ VỀ FORMAT CHUẨN ---
    # Xóa bỏ các trường 'lower' phụ trợ, chỉ trả về đúng key "text" và "type" kèm giá trị gốc
    result = []
    for ent in final_entities:
        result.append({
            "text": ent["orig_text"],
            "type": ent["orig_type"]
        })

    return result

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ LỖI: Không tìm thấy file {INPUT_FILE}")
        return

    print(f"🔄 Đang đọc dữ liệu từ {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_original_entities = 0
    total_cleaned_entities = 0

    print("🧹 Đang tiến hành làm sạch với cơ chế Case-Insensitive...")
    for record in dataset:
        original_entities = record.get("entities", [])
        total_original_entities += len(original_entities)
        
        # Gọi hàm làm sạch
        cleaned_entities = clean_entities(original_entities)
        
        record["entities"] = cleaned_entities
        total_cleaned_entities += len(cleaned_entities)

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Ghi file với định dạng chuẩn
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Báo cáo kết quả
    removed_count = total_original_entities - total_cleaned_entities
    print("-" * 40)
    print(f"✅ ĐÃ LÀM SẠCH THÀNH CÔNG!")
    print(f"📊 Tổng số bệnh án xử lý: {len(dataset)}")
    print(f"🔍 Tổng thực thể ban đầu : {total_original_entities}")
    print(f"🗑️ Đã xóa (trùng/chứa) : {removed_count}")
    print(f"✨ Số thực thể giữ lại   : {total_cleaned_entities}")
    print(f"📂 File kết quả lưu tại  : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
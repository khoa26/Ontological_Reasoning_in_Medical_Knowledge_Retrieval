import json
import random

def process_and_merge_datasets(file_paths, output_train_path, output_val_path, split_ratio=0.9):
    # 1. Định nghĩa Template chuẩn của NuExtract3
    # Bắt buộc sử dụng mảng chứa "verbatim-string" để ép mô hình trích xuất nguyên bản
    template_dict = {
        "TRIỆU_CHỨNG": ["verbatim-string"],
        "TÊN_XÉT_NGHIỆM": ["verbatim-string"],
        "KẾT_QUẢ_XÉT_NGHIỆM": ["verbatim-string"],
        "CHẨN_ĐOÁN": ["verbatim-string"],
        "THUỐC": ["verbatim-string"]
    }
    
    # Dump template ra chuỗi string định dạng JSON
    template_str = json.dumps(template_dict, ensure_ascii=False, indent=2)
    
    merged_data = []

    # 2. Đọc và gộp các file dữ liệu gốc
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
            
    print(f"Tổng số bệnh án thu thập được: {len(merged_data)}")

    # Trộn ngẫu nhiên dữ liệu để tránh thiên lệch khi train
    random.seed(42)
    random.shuffle(merged_data)

    formatted_dataset = []

    # 3. Xử lý từng bệnh án sang chuẩn Prompt của NuExtract
    for item in merged_data:
        text_content = item.get("text", "").strip()
        entities = item.get("entities", [])

        # Khởi tạo Output rỗng chứa mảng
        output_dict = {
            "TRIỆU_CHỨNG": [],
            "TÊN_XÉT_NGHIỆM": [],
            "KẾT_QUẢ_XÉT_NGHIỆM": [],
            "CHẨN_ĐOÁN": [],
            "THUỐC": []
        }

        # Bóc tách và gom nhóm các entities từ dữ liệu gốc
        for ent in entities:
            ent_type = ent.get("type")
            ent_text = ent.get("text")
            
            # Chỉ thêm vào nếu type hợp lệ và chưa tồn tại trong mảng (tránh trùng lặp)
            if ent_type in output_dict and ent_text not in output_dict[ent_type]:
                output_dict[ent_type].append(ent_text)

        # Chuyển Output dict thành chuỗi JSON
        output_str = json.dumps(output_dict, ensure_ascii=False, indent=2)

        # 4. Ghép chuỗi Prompt hoàn chỉnh
        full_prompt = f"<|input|>\n### Template:\n{template_str}\n### Text:\n{text_content}\n<|predict|>\n{output_str}<|end_of_text|>"
        
        # Đưa vào danh sách kết quả (chỉ chứa 1 key "text" duy nhất cho Unsloth)
        formatted_dataset.append({"text": full_prompt})

    # 5. Chia tập Train / Validation
    split_index = int(len(formatted_dataset) * split_ratio)
    train_data = formatted_dataset[:split_index]
    val_data = formatted_dataset[split_index:]

    # Hàm lưu file JSONL
    def save_jsonl(data_list, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for record in data_list:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 6. Xuất file
    save_jsonl(train_data, output_train_path)
    save_jsonl(val_data, output_val_path)

    print(f"Đã tạo file Train: {output_train_path} ({len(train_data)} mẫu)")
    print(f"Đã tạo file Val:   {output_val_path} ({len(val_data)} mẫu)")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
# Thay thế tên file gốc bằng file thực tế của bạn đang lưu trên Colab
input_files = ["datasets/data_gen/final_silver_dataset.json", "datasets/ViMed/vimed_cleaned_dataset.json"] 

process_and_merge_datasets(
    file_paths=input_files,
    output_train_path="fine_tuning/dataset/nuextract3_train.jsonl",
    output_val_path="fine_tuning/dataset/nuextract3_val.jsonl",
    split_ratio=0.9
)
import json
import os

def merge_and_process_acr_data():
    # Danh sách các file đầu vào
    input_files = ["datasets/acrDrAid/data_train.json", "datasets/acrDrAid/data_dev.json", "datasets/acrDrAid/data_test.json"]
    merged_data = []
    
    print("--- Đang tiến hành đọc và gộp các file dữ liệu ---")
    
    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Đã đọc {len(data)} mẫu từ file: {file_path}")
                
                # Bổ sung trích xuất trực tiếp từ viết tắt (acronym) để tiện theo dõi
                for item in data:
                    text = item.get("text", "")
                    start = item.get("start_char_idx", -1)
                    length = item.get("length_acronym", -1)
                    
                    if start >= 0 and length > 0 and start + length <= len(text):
                        acronym = text[start : start + length]
                        item["acronym"] = acronym
                    else:
                        item["acronym"] = ""
                        
                merged_data.extend(data)
        else:
            print(f"Cảnh báo: Không tìm thấy file {file_path}, bỏ qua.")
            
    # Lưu toàn bộ dữ liệu gộp ra 1 file duy nhất
    output_file = "datasets/acrDrAid/data_merged.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nGộp thành công! Tổng số lượng mẫu dữ liệu sau khi gộp: {len(merged_data)}")
    print(f"Đã lưu kết quả vào file: {output_file}")

if __name__ == "__main__":
    merge_and_process_acr_data()
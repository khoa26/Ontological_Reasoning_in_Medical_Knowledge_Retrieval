import json
import random

# Tên file đầu vào
input_file1 = 'datasets/ViMed/vimed_cleaned_dataset.json'
input_file2 = 'datasets/data_gen/final_silver_dataset.json'

# Tên file đầu ra cho Train và Validation
train_output_file = 'fine_tuning/dataset/nuextract_train.jsonl'
val_output_file = 'fine_tuning/dataset/nuextract_val.jsonl'

# Định nghĩa các nhãn cần trích xuất (để tạo Template)
template_keys = [
    "TRIỆU_CHỨNG", 
    "TÊN_XÉT_NGHIỆM", 
    "KẾT_QUẢ_XÉT_NGHIỆM", 
    "CHẨN_ĐOÁN", 
    "THUỐC"
]

def write_jsonl(data, output_path):
    """Hàm phụ trách việc format và ghi dữ liệu ra file .jsonl"""
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in data:
            raw_text = item.get("text", "").strip()
            entities = item.get("entities", [])
            
            # 1. Khởi tạo dictionary kết quả
            predict_dict = {key: [] for key in template_keys}
            
            # 2. Phân loại các thực thể vào đúng key trong dictionary
            for ent in entities:
                ent_text = ent.get("text")
                ent_type = ent.get("type")
                
                # Bỏ qua nếu nhãn không nằm trong template hoặc thực thể đã tồn tại (lọc trùng)
                if ent_type in predict_dict and ent_text not in predict_dict[ent_type]:
                    predict_dict[ent_type].append(ent_text)
            
            # 3. Tạo chuỗi JSON cho Template (rỗng) và Predict (có dữ liệu)
            template_str = json.dumps({key: [] for key in template_keys}, ensure_ascii=False, indent=2)
            predict_str = json.dumps(predict_dict, ensure_ascii=False, indent=2)
            
            # 4. Lắp ghép thành cấu trúc Prompt chuẩn của NuExtract
            formatted_text = f"<|input|>\n### Template:\n{template_str}\n### Text:\n{raw_text}\n\n<|predict|>\n{predict_str}<|end_of_text|>"
            
            # 5. Đóng gói vào key "text" và ghi từng dòng ra file JSONL
            jsonl_record = {"text": formatted_text}
            out_f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')


def convert_and_split_data():
    # 1. Đọc dữ liệu từ cả 2 file
    with open(input_file1, 'r', encoding='utf-8') as f1:
        vimed_data = json.load(f1)
        
    with open(input_file2, 'r', encoding='utf-8') as f2:
        test_data = json.load(f2)

    # 2. Gộp chung và xáo trộn ngẫu nhiên
    combined_data = vimed_data + test_data
    random.seed(42) # Cố định seed để dễ dàng tái tạo lại bộ dữ liệu khi cần
    random.shuffle(combined_data)

    # 3. Tính toán vị trí cắt (90% cho Train, 10% cho Val)
    total_samples = len(combined_data)
    split_index = int(total_samples * 0.9)

    train_data = combined_data[:split_index]
    val_data = combined_data[split_index:]

    # 4. Ghi ra 2 file riêng biệt
    write_jsonl(train_data, train_output_file)
    write_jsonl(val_data, val_output_file)

    # In báo cáo kết quả
    print(f"✅ Đã trộn và chia thành công {total_samples} mẫu.")
    print(f"   -> Tập Train (90%): {len(train_data)} mẫu lưu tại {train_output_file}")
    print(f"   -> Tập Val (10%): {len(val_data)} mẫu lưu tại {val_output_file}")

if __name__ == "__main__":
    convert_and_split_data()
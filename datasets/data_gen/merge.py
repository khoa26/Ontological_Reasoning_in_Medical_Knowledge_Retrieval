import json
import os

input_files =[
    "datasets/data_gen/base_dataset.json",
    "datasets/data_gen/synthetic_dataset.json",
    "datasets/data_gen/synthetic_dataset_2.json"
]
output_file = "datasets/data_gen/final_silver_dataset.json"
def merge_datasets():
    merged_data = []
    
    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if "file_name" in item:
                        del item["file_name"]
                    merged_data.append(item)
        else:
            print(f"Cảnh báo: Không tìm thấy file {file_path}, bỏ qua.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã lưu kết quả vào file: {output_file}")

if __name__ == "__main__":
    merge_datasets()
import json

def process_dataset(input_file, output_file):
    # Đọc dữ liệu từ file JSON gốc
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Duyệt qua từng tài liệu trong dataset
    for document in dataset:
        # Kiểm tra xem tài liệu có trường 'entities' không
        if 'entities' in document:
            for entity in document['entities']:
                # Loại bỏ trường 'start' và 'end'
                # Dùng pop() với giá trị mặc định là None để không báo lỗi nếu trường không tồn tại
                entity.pop('start', None)
                entity.pop('end', None)

    # Lưu dữ liệu đã xử lý sang file JSON mới
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để giữ nguyên tiếng Việt, indent=2 để format file dễ nhìn
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Đã xử lý xong! File mới được lưu tại: {output_file}")

# --- Thay đổi tên file cho phù hợp với máy của bạn ---
input_filename = 'datasets/data_gen/silver_dataset.json'   # Tên file gốc của bạn
output_filename = 'datasets/data_gen/base_dataset.json'  # Tên file mới muốn lưu

process_dataset(input_filename, output_filename)
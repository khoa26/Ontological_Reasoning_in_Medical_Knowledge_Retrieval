import json
import random
import os

def load_json_data(file_path):
    """Đọc dữ liệu từ file JSON thô của acrDrAid."""
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_chatml_format(raw_data):
    """
    Chuyển đổi dữ liệu thô sang format ChatML (messages) dùng cho Qwen 2.5 Instruct.
    """
    # Các mẫu câu hỏi đa dạng để mô hình không bị học vẹt (overfit)
    prompt_templates = [
        "Dựa vào ngữ cảnh của câu y khoa sau, hãy giải nghĩa từ viết tắt '{acronym}':\n\n\"{text}\"",
        "Trong câu: \"{text}\"\nTừ '{acronym}' là viết tắt của cụm từ gì?",
        "Hãy cho biết dạng viết đầy đủ của từ '{acronym}' xuất hiện trong văn bản dưới đây:\n\n{text}",
        "Xác định nghĩa của từ viết tắt '{acronym}' dựa trên ngữ cảnh sau:\n{text}",
        "Văn bản: {text}\nCâu hỏi: Từ '{acronym}' có nghĩa là gì?"
    ]

    # Các mẫu câu trả lời đa dạng
    response_templates = [
        "Trong ngữ cảnh này, '{acronym}' là viết tắt của \"{expansion}\".",
        "Dựa vào ngữ cảnh, từ '{acronym}' có nghĩa là \"{expansion}\".",
        "Từ viết tắt '{acronym}' được viết đầy đủ là \"{expansion}\".",
        "\"{expansion}\" là dạng viết đầy đủ của từ '{acronym}'.",
        "Đó là \"{expansion}\"."
    ]

    formatted_data = []

    for item in raw_data:
        text = item.get("text", "")
        start = item.get("start_char_idx", -1)
        length = item.get("length_acronym", -1)
        expansion = item.get("expansion", "")
        
        # Kiểm tra tính hợp lệ của index
        if start < 0 or length <= 0 or start + length > len(text):
            continue
            
        # Cắt từ viết tắt từ văn bản gốc dựa vào index
        acronym = text[start : start + length]
        
        # Kiểm tra nếu cắt ra bị trống hoặc toàn khoảng trắng
        if not acronym.strip():
            continue
            
        # Chọn ngẫu nhiên template để tạo sự đa dạng ngữ liệu
        user_prompt = random.choice(prompt_templates).format(acronym=acronym, text=text)
        assistant_response = random.choice(response_templates).format(acronym=acronym, expansion=expansion)
        
        # Tạo object theo format ChatML chuẩn của Qwen/Llama
        chat_object = {
            "messages": [
                {"role": "system", "content": "Bạn là một trợ lý y khoa AI tiếng Việt. Nhiệm vụ của bạn là giải nghĩa chính xác các từ viết tắt trong hồ sơ bệnh án dựa vào ngữ cảnh."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        
        formatted_data.append(chat_object)
        
    return formatted_data

def save_jsonl(data, output_file):
    """Lưu danh sách dict thành file định dạng JSONL."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Đã lưu thành công {len(data)} mẫu vào: {output_file}")

def process_all_splits():
    """Xử lý đồng thời cả 3 tập Train, Dev, Test."""
    files_mapping = {
        "datasets/acrDrAid/data_merged.json": "datasets/acrDrAid/qwen_acronym_train.jsonl",
    }

    for input_file, output_file in files_mapping.items():
        print(f"\n--- Đang xử lý file: {input_file} ---")
        raw_data = load_json_data(input_file)
        if raw_data:
            formatted = convert_to_chatml_format(raw_data)
            save_jsonl(formatted, output_file)
            
            # In thử 1 mẫu để kiểm tra kết quả
            if formatted:
                print("Ví dụ mẫu đầu ra:")
                print(json.dumps(formatted[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    process_all_splits()

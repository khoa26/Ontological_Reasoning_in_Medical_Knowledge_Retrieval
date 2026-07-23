import os
import json

LABEL_MAPPING = {
    "ten_benh": "CHẨN_ĐOÁN",
    "trieu_chung_benh": "TRIỆU_CHỨNG",
    "bien_phap_chan_doan": "TÊN_XÉT_NGHIỆM",
    "bien_phap_dieu_tri": "THUỐC"
}

def filter_overlapping_entities(entities):
    """
    Hàm lọc các entities trùng lặp hoặc bao hàm lẫn nhau (chuỗi con).
    """
    if not entities:
        return []
        
    # Bước 1: Loại bỏ trùng lặp hoàn toàn (bỏ qua phân biệt hoa thường)
    unique_entities = []
    seen = set()
    for ent in entities:
        identifier = (ent['text'].lower(), ent['type'])
        if identifier not in seen:
            seen.add(identifier)
            unique_entities.append(ent)
            
    # Bước 2: Sắp xếp các entity theo độ dài text giảm dần (để ưu tiên giữ lại chuỗi dài nhất)
    unique_entities.sort(key=lambda x: len(x['text']), reverse=True)
    
    # Bước 3: Lọc bỏ các entity là chuỗi con của entity khác có cùng nhãn
    final_entities = []
    for ent in unique_entities:
        is_substring = False
        for kept_ent in final_entities:
            # Nếu text của ent hiện tại nằm trong một ent đã giữ lại VÀ cùng type
            if ent['text'].lower() in kept_ent['text'].lower() and ent['type'] == kept_ent['type']:
                is_substring = True
                break
        
        if not is_substring:
            final_entities.append(ent)
            
    return final_entities

def parse_vimedner_file(file_path):
    documents = []
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return documents

    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    doc = process_sentence(tokens, tags)
                    if doc:
                        documents.append(doc)
                    tokens = []
                    tags = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
        
        if tokens:
            doc = process_sentence(tokens, tags)
            if doc:
                documents.append(doc)
                
    return documents

def process_sentence(tokens, tags):
    full_text = " ".join(tokens)
    entities = []
    
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            raw_label = tag[2:] 
            mapped_type = LABEL_MAPPING.get(raw_label)
            
            entity_tokens = [tokens[i]]
            i += 1
            
            while i < len(tags) and tags[i] == f"I-{raw_label}":
                entity_tokens.append(tokens[i])
                i += 1
                
            if mapped_type:
                entity_text = " ".join(entity_tokens)
                entities.append({
                    "text": entity_text,
                    "type": mapped_type
                })
        else:
            i += 1
            
    # Gọi hàm lọc entity trước khi trả về kết quả
    filtered_entities = filter_overlapping_entities(entities)
    
    return {
        "text": full_text,
        "entities": filtered_entities
    }

if __name__ == "__main__":
    all_documents = []
    
    data_files = [
        'datasets/ViMed/train.txt',
        'datasets/ViMed/dev.txt',
        'datasets/ViMed/test.txt'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📥 Đang xử lý file: {file_path}")
            docs = parse_vimedner_file(file_path)
            all_documents.extend(docs)
        else:
            print(f"⚠️ Không tìm thấy file tại đường dẫn: {file_path}")
            
    output_filename = 'datasets/ViMed/vimedner_processed_dataset.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn tất! Tổng số mẫu bệnh án đã xử lý: {len(all_documents)}")
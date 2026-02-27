import json
from pathlib import Path

def integrate_descriptions(split_name='sample_data'):
    """将图像描述整合回原始 docs.json"""

    base_dir = Path('/home/ll/Desktop/codes/semeval2026-task12-dataset')

    # 读取原始数据
    with open(base_dir / split_name / 'docs_updated.json', 'r', encoding='utf-8') as f:
        docs_data = json.load(f)

    # 读取所有图像描述
    descriptions = {}
    images_dir = base_dir / f'valik_prepared/{split_name}/images'

    for txt_file in images_dir.glob('*.txt'):
        # 从文件名提取 UUID
        filename = txt_file.stem  # topic1_abc123
        uuid = filename.split('_', 1)[1] if '_' in filename else filename

        with open(txt_file, 'r', encoding='utf-8') as f:
            descriptions[uuid] = f.read()

    # 整合描述到数据中
    for topic in docs_data:
        for doc in topic['docs']:
            uuid = doc['uuid']
            if uuid in descriptions:
                doc['image_description'] = descriptions[uuid]
            else:
                doc['image_description'] = None

    # 保存增强后的数据
    output_file = base_dir / split_name / 'docs_with_descriptions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(docs_data, f, indent=4, ensure_ascii=False)

    print(f"✓ 已保存增强数据到: {output_file}")
    stats = sum(1 for t in docs_data for d in t['docs'] if d.get('image_description'))
    print(f"  成功添加 {stats} 个图像描述")

if __name__ == "__main__":
    for split in ['sample_data', 'train_data', 'dev_data']:
        integrate_descriptions(split)
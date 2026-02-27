from pathlib import Path

def merge_texts(split_name='train_data'):
    """合并原始文本和图像描述"""
    base_dir = Path(f'valik_prepared/{split_name}')
    texts_dir = base_dir / 'texts'
    images_dir = base_dir / 'images'
    output_dir = base_dir / 'merged_texts'
    output_dir.mkdir(exist_ok=True)

    for text_file in texts_dir.glob('*.txt'):
        uuid = text_file.stem  # topic1_abc123

        # 读取原始文本
        with open(text_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # 查找对应的图像描述
        img_desc_file = images_dir / f"{uuid}.txt"
        image_description = ""
        if img_desc_file.exists():
            with open(img_desc_file, 'r', encoding='utf-8') as f:
                image_description = f.read()

        # 合并
        merged_content = f"""{original_text}

--- 图像描述 ---
{image_description}
"""

        # 保存合并后的文本
        output_file = output_dir / f"{uuid}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(merged_content)

    print(f"✓ 合并完成: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Process all three dataset splits
    for split in ['train_data', 'dev_data', 'sample_data', 'test_data']:
        print(f"\nProcessing {split}...")
        merge_texts(split)
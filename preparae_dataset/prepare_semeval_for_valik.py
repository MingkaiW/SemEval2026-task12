import json
from pathlib import Path
import shutil

def prepare_semeval_data(split_name='sample_data'):
    """
    准备 SemEval 数据以供 VaLiK 处理

    Args:
        split_name: 'sample_data', 'train_data', 或 'dev_data'
    """
    print(f"准备 {split_name} 数据...")

    base_dir = Path('/home/ll/Desktop/codes/semeval2026-task12-dataset')
    split_dir = base_dir / split_name

    # 读取数据（使用 docs_updated.json，包含 local_image_path）
    with open(split_dir / 'docs_updated.json', 'r', encoding='utf-8') as f:
        docs_data = json.load(f)

    with open(split_dir / 'questions.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    # 创建输出目录
    output_dir = base_dir / f'valik_prepared/{split_name}'
    images_dir = output_dir / 'images'
    texts_dir = output_dir / 'texts'
    images_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    # 创建 UUID 到问题的映射
    uuid_to_question = {q['id']: q for q in questions}

    processed_count = 0
    skipped_count = 0

    # 处理每个主题
    for topic in docs_data:
        topic_id = topic['topic_id']
        topic_text = topic['topic']

        for doc in topic['docs']:
            uuid = doc['id']

            # 检查 local_image_path 是否存在且非空
            local_image_path = doc.get('local_image_path')
            if not local_image_path:
                skipped_count += 1
                continue

            # 复制图像文件
            src_image_path = base_dir / local_image_path
            if src_image_path.exists() and src_image_path.is_file():
                # 使用 topic_uuid 作为文件名以保持唯一性
                dst_image_path = images_dir / f"topic{topic_id}_{uuid}{src_image_path.suffix}"
                shutil.copy2(src_image_path, dst_image_path)

                # 创建对应的文本文件（原始文本）
                text_content = f"""主题: {topic_text}

标题: {doc.get('title', '')}
来源: {doc.get('source', '')}
链接: {doc.get('link', '')}

摘要:
{doc.get('snippet', '')}

正文:
{doc.get('content', '')}
"""

                # 如果有对应的问题，添加问题信息
                if uuid in uuid_to_question:
                    question_data = uuid_to_question[uuid]
                    text_content += f"""

相关问题:
目标事件: {question_data.get('target_event', '')}
选项A: {question_data.get('option_a', '')}
选项B: {question_data.get('option_b', '')}
选项C: {question_data.get('option_c', '')}
选项D: {question_data.get('option_d', '')}
"""

                # 保存文本文件
                text_path = texts_dir / f"topic{topic_id}_{uuid}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                processed_count += 1
            else:
                print(f"  ! 跳过 {uuid}: 图像文件不存在或不是文件")
                skipped_count += 1

    print("✓ 完成！")
    print(f"  成功处理: {processed_count} 个文档")
    print(f"  跳过: {skipped_count} 个文档")
    print(f"  图像目录: {images_dir}")
    print(f"  文本目录: {texts_dir}")

    return output_dir

if __name__ == "__main__":
    # 处理所有数据集
    for split in ['sample_data', 'train_data', 'dev_data', 'test_data']:
        try:
            prepare_semeval_data(split)
            print()
        except Exception as e:
            print(f"✗ 处理 {split} 时出错: {e}")
            import traceback
            traceback.print_exc()
            print()
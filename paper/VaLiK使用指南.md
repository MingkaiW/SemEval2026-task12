# VaLiK 框架使用指南 - SemEval 2026 Task 12

## 📋 目录
1. [项目概述](#项目概述)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [运行流程](#运行流程)
5. [常见问题](#常见问题)

---

## 🎯 项目概述

### VaLiK 是什么？
**VaLiK** = Vision-align-to-Language integrated Knowledge Graph（视觉对齐语言的集成知识图谱）

这是一个发表在 ICCV 2025 的研究框架，通过三阶段流程为大语言模型提供多模态推理能力：

1. **基于专家集成的视觉到语言建模**：使用多个视觉语言模型（VLM）为图像生成文本描述
2. **跨模态相似度验证**：基于图像-文本相似度修剪描述（可选）
3. **多模态知识图谱构建**：使用 LightRAG 构建知识图谱以增强推理

### 你的数据情况
```
当前 SemEval 数据结构：
├── train_data/
│   ├── docs.json              # 文档语料库（包含嵌入的图像）
│   └── questions.jsonl        # 问题数据
├── dev_data/
│   ├── docs.json
│   └── questions.jsonl
├── sample_data/
│   ├── docs.json
│   └── questions.jsonl
└── downloaded_images/         # 已下载的图像
    ├── train_data/
    │   ├── topic_1/
    │   │   ├── <uuid>.jpg
    │   │   └── ...
    │   └── topic_2/
    ├── dev_data/
    └── sample_data/

统计信息：
- sample_data: 10个主题, 164个文档, 163张图像
- train_data: 36个主题, 775个文档, 762张图像
- dev_data: 36个主题, 775个文档, 762张图像
- 总计: 1,714个文档, 1,687张图像
```

---

## 🔧 环境准备

### 步骤 1: 创建 Conda 环境

```bash
# 进入 VaLiK 目录
cd /home/ll/Desktop/codes/semeval2026-task12-dataset/VaLiK

# 方法1: 使用 requirements.txt
conda create -n valik python=3.10
conda activate valik
pip install -r requirements.txt

# 方法2: 使用 environment.yml（推荐）
conda env create -f environment.yml
conda activate valik
```

### 步骤 2: 检查 GPU 和 CUDA

```bash
# 检查 CUDA 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# 检查显存大小
nvidia-smi
```

**根据显存选择模型：**
- **< 16GB**: 使用 LLaVA-7B 或 BLIP2
- **16-24GB**: 使用 LLaVA-13B 或 Qwen2-VL-7B
- **40GB+**: 使用 Qwen2-VL-72B（需要量化）
- **80GB (A100)**: 可以运行论文中的完整配置

### 步骤 3: 安装 Ollama（推荐用于 LLaVA）

```bash
# 下载并安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 服务
ollama serve &

# 拉取所需模型
ollama pull llava:7b          # 用于图像描述
ollama pull qwen2.5:7b        # 用于知识图谱构建（或使用 32b 获得更好质量）

# 测试 Ollama
ollama list
```

### 步骤 4: 验证安装

```bash
# 测试导入关键库
python -c "from transformers import AutoModel; print('✓ transformers')"
python -c "from lightrag import LightRAG; print('✓ lightrag')"
python -c "import torch; print('✓ torch')"
python -c "from PIL import Image; print('✓ PIL')"
```

---

## 📊 数据准备

### 步骤 5: 创建数据适配脚本

由于 VaLiK 期望的数据格式与 SemEval 不完全匹配，需要创建适配脚本：

创建文件：`/home/ll/Desktop/codes/semeval2026-task12-dataset/prepare_semeval_for_valik.py`

```python
import json
import os
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

    # 读取数据
    with open(split_dir / 'docs.json', 'r', encoding='utf-8') as f:
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
    uuid_to_question = {q['uuid']: q for q in questions}

    processed_count = 0

    # 处理每个主题
    for topic in docs_data:
        topic_id = topic['topic_id']
        topic_text = topic['topic']

        for doc in topic['docs']:
            uuid = doc['uuid']

            # 复制图像文件
            src_image_path = base_dir / doc.get('local_image_path', '')
            if src_image_path.exists():
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
问题: {question_data.get('question', '')}
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

    print(f"✓ 完成！处理了 {processed_count} 个文档")
    print(f"  图像目录: {images_dir}")
    print(f"  文本目录: {texts_dir}")

    return output_dir

if __name__ == "__main__":
    # 处理所有数据集
    for split in ['sample_data', 'train_data', 'dev_data']:
        prepare_semeval_data(split)
```

### 步骤 6: 运行数据准备脚本

```bash
cd /home/ll/Desktop/codes/semeval2026-task12-dataset
python prepare_semeval_for_valik.py
```

这将创建以下结构：
```
valik_prepared/
├── sample_data/
│   ├── images/          # 所有图像的平面结构
│   │   ├── topic1_<uuid>.jpg
│   │   └── topic2_<uuid>.png
│   └── texts/           # 对应的原始文本
│       ├── topic1_<uuid>.txt
│       └── topic2_<uuid>.txt
├── train_data/
└── dev_data/
```

---

## 🚀 运行流程

### 阶段 1: 图像到文本转换（必需）

#### 选项 A: 使用 LLaVA（推荐，易于设置）

```bash
cd /home/ll/Desktop/codes/semeval2026-task12-dataset/VaLiK

# 确保 Ollama 正在运行
ollama serve &

# 先在 sample_data 上测试
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  llava \
  --llava_version 7b

# 如果测试成功，处理完整数据集
python src/Image_to_Text.py \
  --input ../valik_prepared/train_data/images \
  llava \
  --llava_version 7b

python src/Image_to_Text.py \
  --input ../valik_prepared/dev_data/images \
  llava \
  --llava_version 7b
```

**输出**: 每个图像文件旁边会生成一个同名的 `.txt` 文件，包含图像描述。
例如：`topic1_abc123.jpg` → `topic1_abc123.txt`

#### 选项 B: 使用 Qwen3-VL（新模型，平衡性能）

```bash
# 首先确保安装了 Qwen3-VL 模型
ollama pull qwen3-vl:8b

# 在 sample_data 上测试
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  qwen3 \
  --qwen3_version 8b

# 处理完整数据集
python src/Image_to_Text.py \
  --input ../valik_prepared/train_data/images \
  qwen3 \
  --qwen3_version 8b

python src/Image_to_Text.py \
  --input ../valik_prepared/dev_data/images \
  qwen3 \
  --qwen3_version 8b
```

**可用版本**：
- `qwen3-vl:8b` - 8B 参数，~8GB 显存，速度快，质量好（推荐）
- `qwen3-vl:14b` - 14B 参数，~14GB 显存，质量更好
- `qwen3-vl:72b` - 72B 参数，~40GB 显存，最佳质量

**优势**：
- 通过 Ollama 运行，设置简单（类似 LLaVA）
- 比 LLaVA 更新的模型架构
- 平衡了质量和速度
- 支持多种模型大小选择

#### 选项 C: 使用 Qwen2-VL（最佳质量，需要更多显存）

```bash
# 使用量化以节省显存
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  qwen2-vl \
  --qwen2vl_version 7b \
  --use_quantization

# 如果有足够显存（40GB+），可以使用 72B 版本
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  qwen2-vl \
  --qwen2vl_version 72b \
  --use_quantization
```

#### 选项 D: 使用 BLIP2（快速，质量较低）

```bash
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  blip2 \
  --blip2_version flan-t5
```

#### 选项 E: 使用集成方法（论文推荐，最佳质量）

```bash
# 运行多个模型并合并结果
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  llava --llava_version 7b

python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  qwen3 --qwen3_version 8b

python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  blip2 --blip2_version flan-t5

# 然后手动合并描述（需要自己写脚本）
```

### 阶段 2: 跨模态相似度验证（可选，推荐）

这一步会过滤掉与图像相似度低的描述。

```bash
cd /home/ll/Desktop/codes/semeval2026-task12-dataset/VaLiK

# 对单个图像-文本对进行验证
python src/Prune/similarity_verification.py \
  --image_path ../valik_prepared/sample_data/images/topic1_abc123.jpg \
  --text_path ../valik_prepared/sample_data/images/topic1_abc123.txt \
  --threshold 0.20 \
  --mode sentence
```

**批量处理脚本**（需要自己创建）：

创建 `batch_verify.py`：
```python
import os
import subprocess
from pathlib import Path

images_dir = Path('../valik_prepared/sample_data/images')

for img_file in images_dir.glob('*.jpg'):
    txt_file = img_file.with_suffix('.txt')
    if txt_file.exists():
        cmd = [
            'python', 'src/Prune/similarity_verification.py',
            '--image_path', str(img_file),
            '--text_path', str(txt_file),
            '--threshold', '0.20',
            '--mode', 'sentence'
        ]
        subprocess.run(cmd)
```

### 阶段 3: 知识图谱构建（可选）

```bash
cd /home/ll/Desktop/codes/semeval2026-task12-dataset/VaLiK/src/LightRAG

# 先合并原始文本和图像描述
cd /home/ll/Desktop/codes/semeval2026-task12-dataset
```

创建合并脚本 `merge_texts.py`：
```python
from pathlib import Path

def merge_texts(split_name='sample_data'):
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
    merge_texts('sample_data')
```

运行合并：
```bash
python merge_texts.py
```

然后使用 LightRAG 构建知识图谱：

```bash
cd VaLiK/src/LightRAG

# 修改 lightrag_ollama_demo.py 中的路径和参数
# 然后运行
python lightrag_ollama_demo.py
```

---

## 📈 结果整合

### 步骤 7: 将描述合并回原始数据集

创建 `integrate_descriptions.py`：
```python
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
```

运行整合：
```bash
python integrate_descriptions.py
```

---

## ❓ 常见问题

### Q1: Ollama 连接失败
```bash
# 确保 Ollama 服务正在运行
ps aux | grep ollama

# 如果没有运行，启动它
ollama serve &

# 等待几秒钟让服务启动
sleep 5

# 测试连接
ollama list
```

### Q2: CUDA 内存不足
**解决方案**：
1. 使用更小的模型（LLaVA-7B 而不是 13B）
2. 启用量化（`--use_quantization`）
3. 减小批处理大小
4. 使用 DeepSpeed 或 bitsandbytes 进行优化

### Q3: 图像描述质量不佳
**解决方案**：
1. 使用集成方法（多个 VLM）
2. 尝试不同的模型（Qwen2-VL 通常质量最好）
3. 调整提示词（修改 VaLiK 代码中的 prompt）
4. 使用相似度验证过滤低质量描述

### Q4: 处理速度太慢
**加速方法**：
1. 使用 GPU（必须）
2. 增大批处理大小（如果显存允许）
3. 使用更快的模型（BLIP2）
4. 并行处理多个 GPU

### Q5: 如何选择模型？

| 需求 | 推荐模型 | 显存需求 | 处理速度 | 质量 |
|------|---------|---------|---------|------|
| **快速测试** | BLIP2 | ~8GB | 快 | 中等 |
| **易于设置** | LLaVA-7B | ~12GB | 中等 | 好 |
| **平衡推荐** | **Qwen3-VL-8B** | **~8GB** | **中等** | **很好** |
| **最佳质量** | Qwen2-VL-72B | ~40GB | 慢 | 最佳 |
| **生产环境** | 集成方法 | 变化 | 中等 | 最佳 |

**Qwen3-VL 优势**：
- ✅ 通过 Ollama 运行，设置简单
- ✅ 更新的视觉语言模型架构，性能优于 LLaVA
- ✅ 显存需求适中（8GB 即可运行 8B 版本）
- ✅ 质量和速度的最佳平衡

---

## 🎯 推荐工作流程

### 第一次运行（测试）
```bash
# 1. 准备环境
conda activate valik
cd /home/ll/Desktop/codes/semeval2026-task12-dataset

# 2. 准备数据（仅 sample_data）
python prepare_semeval_for_valik.py

# 3. 确保模型已安装
ollama pull qwen3-vl:8b

# 4. 运行 VaLiK（仅图像描述，使用 Qwen3-VL）
cd VaLiK
ollama serve &
python src/Image_to_Text.py \
  --input ../valik_prepared/sample_data/images \
  qwen3 --qwen3_version 8b

# 5. 检查结果
ls -lh ../valik_prepared/sample_data/images/*.txt | head

# 6. 整合回数据集
cd ..
python integrate_descriptions.py
```

### 完整运行（生产）
在测试成功后，扩展到完整数据集：
```bash
# 处理 train_data
python src/Image_to_Text.py \
  --input ../valik_prepared/train_data/images \
  qwen3 --qwen3_version 8b

# 处理 dev_data
python src/Image_to_Text.py \
  --input ../valik_prepared/dev_data/images \
  qwen3 --qwen3_version 8b

# 整合所有结果
python integrate_descriptions.py
```

---

## 📚 参考资源

- **VaLiK 论文**: ICCV 2025 - "Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning"
- **Ollama 文档**: https://ollama.com/
- **LightRAG 文档**: VaLiK/src/LightRAG/lightrag/README.md
- **模型下载**:
  - LLaVA: `ollama pull llava:7b`
  - Qwen2-VL: Hugging Face 或 Ollama
  - BLIP2: 自动从 Hugging Face 下载

---

## 💡 最佳实践

1. **始终先在 sample_data 上测试**
2. **保存中间结果**（以防失败需要重新运行）
3. **监控 GPU 使用率**（`nvidia-smi` 或 `watch -n 1 nvidia-smi`）
4. **记录实验配置**（模型版本、参数、处理时间等）
5. **备份原始数据**（在运行前）

---

## 📞 需要帮助？

如果遇到问题：
1. 检查错误日志
2. 验证环境配置（`conda list`）
3. 确认 GPU 和 CUDA 可用
4. 查看 VaLiK README: `VaLiK/README.md`
5. 检查 Ollama 服务状态

祝实验顺利！🚀
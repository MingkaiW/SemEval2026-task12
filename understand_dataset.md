# datainspection
/home/ll/Desktop/codes/semeval2026-task12-dataset/dataset_analysis.ipynb

find that the docs info contains image modality info, download and index them:

# 2. Download and Index Image Modality Info

/home/ll/Desktop/codes/semeval2026-task12-dataset/process_images.py

# 3. use valik to analysis as baseline

# 4. convert dataset to the valik format

/home/ll/Desktop/codes/semeval2026-task12-dataset/preparae_dataset/prepare_semeval_for_valik.py

使用 docs_updated.json - 包含 local_image_path 字段
添加空值检查 - 在处理前验证 local_image_path 存在且非空
添加文件类型检查 - 使用 .is_file() 确保不会复制目录
添加跳过计数 - 统计跳过的文档数量
添加异常处理 - 防止一个数据集失败影响其他数据集
移除了 question 字段 - 修正为只显示 target_event 和选项
清理了未使用的导入 - 移除了 os 模块


# ML baselines based on siglip+qwen3:0.6b-embedding and ML classifier
/home/ll/Desktop/codes/semeval2026-task12-dataset/baselines/ml_mlp_baseline.py


INFO:__main__:Saved text embeddings to cache/original_embeddings/text/c8c591548cdcd854d750c3a0f390fcb8.pt
INFO:__main__:Saved image embeddings to cache/original_embeddings/image/c8c591548cdcd854d750c3a0f390fcb8.pt
INFO:__main__:train - Feature shape: (1819, 1792), Labels shape: (1819, 4)
INFO:__main__:
================================================================================
INFO:__main__:STEP 2: Preparing Training/Validation Split
INFO:__main__:================================================================================
INFO:__main__:Split training set: 1456 train, 363 val
INFO:__main__:Training samples: 1456
INFO:__main__:Validation samples: 363
INFO:__main__:
================================================================================
INFO:__main__:STEP 3: Training and Evaluating Classifiers
INFO:__main__:================================================================================
INFO:__main__:Normalizing features...
INFO:__main__:
============================================================
INFO:__main__:Training Logistic Regression...
INFO:__main__:============================================================
INFO:__main__:Logistic Regression training completed.
INFO:__main__:
Training Metrics for Logistic Regression:
INFO:__main__:  hamming_loss: 0.0759
INFO:__main__:  accuracy: 0.7404
INFO:__main__:  f1_micro: 0.8997
INFO:__main__:  f1_macro: 0.8980
INFO:__main__:  f1_samples: 0.8418
INFO:__main__:  precision_micro: 0.9314
INFO:__main__:  recall_micro: 0.8701
INFO:__main__:
Validation Metrics for Logistic Regression:
INFO:__main__:  hamming_loss: 0.3382
INFO:__main__:  accuracy: 0.2121
INFO:__main__:  f1_micro: 0.5549
INFO:__main__:  f1_macro: 0.5534
INFO:__main__:  f1_samples: 0.4454
INFO:__main__:  precision_micro: 0.5784
INFO:__main__:  recall_micro: 0.5331
INFO:__main__:
============================================================
INFO:__main__:Training Random Forest...
INFO:__main__:============================================================
INFO:__main__:Random Forest training completed.
INFO:__main__:
Training Metrics for Random Forest:
INFO:__main__:  hamming_loss: 0.0000
INFO:__main__:  accuracy: 1.0000
INFO:__main__:  f1_micro: 1.0000
INFO:__main__:  f1_macro: 1.0000
INFO:__main__:  f1_samples: 1.0000
INFO:__main__:  precision_micro: 1.0000
INFO:__main__:  recall_micro: 1.0000
INFO:__main__:
Validation Metrics for Random Forest:
INFO:__main__:  hamming_loss: 0.3822
INFO:__main__:  accuracy: 0.1350
INFO:__main__:  f1_micro: 0.4553
INFO:__main__:  f1_macro: 0.4521
INFO:__main__:  f1_samples: 0.3310
INFO:__main__:  precision_micro: 0.5213
INFO:__main__:  recall_micro: 0.4042
INFO:__main__:
============================================================
INFO:__main__:Training Gradient Boosting...
INFO:__main__:============================================================
INFO:__main__:Gradient Boosting training completed.
INFO:__main__:
Training Metrics for Gradient Boosting:
INFO:__main__:  hamming_loss: 0.1106
INFO:__main__:  accuracy: 0.5941
INFO:__main__:  f1_micro: 0.8379
INFO:__main__:  f1_macro: 0.8354
INFO:__main__:  f1_samples: 0.7012
INFO:__main__:  precision_micro: 0.9823
INFO:__main__:  recall_micro: 0.7306
INFO:__main__:
Validation Metrics for Gradient Boosting:
INFO:__main__:  hamming_loss: 0.3747
INFO:__main__:  accuracy: 0.1074
INFO:__main__:  f1_micro: 0.4213
INFO:__main__:  f1_macro: 0.4159
INFO:__main__:  f1_samples: 0.2888
INFO:__main__:  precision_micro: 0.5410
INFO:__main__:  recall_micro: 0.3449
INFO:__main__:
================================================================================
INFO:__main__:CLASSIFIER COMPARISON
INFO:__main__:================================================================================
INFO:__main__:
Classifier                F1-Micro     F1-Macro     Accuracy     Hamming Loss
INFO:__main__:--------------------------------------------------------------------------------
INFO:__main__:Logistic Regression       0.5549       0.5534       0.2121       0.3382      
INFO:__main__:Random Forest             0.4553       0.4521       0.1350       0.3822      
INFO:__main__:Gradient Boosting         0.4213       0.4159       0.1074       0.3747      
INFO:__main__:================================================================================
INFO:__main__:
Best classifier based on f1_micro: Logistic Regression (score: 0.5549)
INFO:__main__:
================================================================================
INFO:__main__:Training and Evaluation Complete!
INFO:__main__:================================================================================

# 6
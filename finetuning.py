import torch
import random
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
import numpy as np
import os
import gc

# 设置环境变量以优化GPU使用
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer并行警告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 更好的CUDA错误报告


# 1. 数据加载与预处理
def load_data(syno_path, anto_path):
    synonyms = []
    with open(syno_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word1, word2 = parts[0], ' '.join(parts[1:])
                synonyms.append((word1, word2, 1))  # 标签1=同义词

    antonyms = []
    with open(anto_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word1, word2 = parts[0], parts[1]
                antonyms.append((word1, word2, 0))  # 标签0=反义词

    return synonyms + antonyms


# 2. 格式化输入
def format_example(word1, word2, label=None):
    text = (
        "判断两个词是否为同义词，只输出是或否\n"
        f"词1：{word1}\n词2：{word2}\n"
    )
    if label is not None:
        answer = "是" if label == 1 else "否"
        text += answer
    return text


# 3. 自定义数据集类 - 优化内存使用
class SynonymDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encodings = []

        # 预计算所有样本
        for idx in range(len(data)):
            word1, word2, label = data[idx]
            text = format_example(word1, word2, label)
            encoding = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=64,  # 减少序列长度以节省显存
                return_tensors="pt"
            )

            # 创建标签：只计算答案位置的损失
            input_ids = encoding["input_ids"].squeeze(0)
            labels = input_ids.clone()

            # 找到答案开始位置
            prompt_text = text.split("答案：")[0] if "答案：" in text else text
            answer_start_idx = len(tokenizer.encode(prompt_text, add_special_tokens=True))

            # 将非答案位置设为-100（损失忽略）
            labels[:answer_start_idx] = -100

            self.encodings.append({
                "input_ids": input_ids,
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": labels
            })

        # 释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encodings[idx]


# 4. 修复自定义Trainer类 - 添加**kwargs参数
class SynonymTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# 主函数
def main():
    # 检查GPU可用性
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个GPU设备")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
    else:
        print("警告：未发现GPU，将使用CPU训练（速度会很慢）")
        device = torch.device("cpu")

    # 加载数据
    dataset = load_data("syno_light.txt", "anto_light.txt")
    random.shuffle(dataset)

    # 划分数据集 (80% 训练, 20% 验证)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    # 加载模型和tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 配置量化以节省显存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用4位量化
        bnb_4bit_quant_type="nf4",  # 使用NF4量化
        bnb_4bit_use_double_quant=True,  # 使用双重量化进一步压缩
        bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16
    )

    # 加载模型到GPU - 使用量化配置
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"  # 自动分配设备
    )

    # 设置填充token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # 创建数据集
    train_dataset = SynonymDataset(train_data, tokenizer)
    val_dataset = SynonymDataset(val_data, tokenizer)

    # 训练参数 - 优化显存使用
    training_args = TrainingArguments(
        output_dir="./syno_detector",
        per_device_train_batch_size=2,  # 进一步减小批次大小
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,  # 适当增加学习率
        weight_decay=0.01,
        fp16=True if torch.cuda.is_available() else False,
        gradient_accumulation_steps=16,  # 增加梯度累积步数
        gradient_checkpointing=True,  # 启用梯度检查点
        optim="paged_adamw_8bit",  # 使用分页优化器减少内存峰值
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,  # 移除未使用的列以节省内存
        group_by_length=True,  # 按长度分组样本以减少填充
    )

    # 创建Trainer
    trainer = SynonymTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 开始训练
    print("开始训练模型...")
    try:
        # 训练前释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer.train()
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()
        # 尝试使用更简单的配置重新训练
        print("尝试使用禁用FP16的配置...")
        training_args.fp16 = False
        trainer = SynonymTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()

    # 保存模型
    model.save_pretrained("./syno_detector_final")
    tokenizer.save_pretrained("./syno_detector_final")
    print("模型保存完成")

    # 测试推理
    test_pairs = [
        ("乘隙", "顺便", 1),
        ("前", "后", 0),
        ("教育", "训诲", 1),
        ("冷", "热", 0)
    ]

    model.eval()
    for word1, word2, true_label in test_pairs:
        prompt = format_example(word1, word2)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        # 解析预测结果
        answer_token = outputs.sequences[0, -1].item()
        prediction = tokenizer.decode([answer_token])

        # 获取置信度
        scores = outputs.scores[0][0]
        token_yes = tokenizer.encode("是")[0]
        token_no = tokenizer.encode("否")[0]

        probs = torch.softmax(scores, dim=-1)
        prob_yes = probs[token_yes].item()
        prob_no = probs[token_no].item()

        # 根据预测结果选择置信度
        if prediction == "是":
            confidence = prob_yes
            pred_label = 1
        elif prediction == "否":
            confidence = prob_no
            pred_label = 0
        else:
            confidence = max(prob_yes, prob_no)
            pred_label = 1 if prob_yes > prob_no else 0

        print(f"\n词对: '{word1}' - '{word2}'")
        print(f"真实标签: {'同义词' if true_label == 1 else '反义词'}")
        print(f"预测结果: {'同义词' if pred_label == 1 else '反义词'} (置信度: {confidence:.2%})")
        print(f"预测正确: {pred_label == true_label}")


if __name__ == "__main__":
    main()
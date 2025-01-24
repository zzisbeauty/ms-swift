"""
想要使用 ds 脚本运行预训练，此脚本改写来自中文 llama 训练脚本
"""

import os
import shutil
import deepspeed
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM


output_model = os.path.join(os.getcwd(), "output_model")
os.makedirs(output_model, exist_ok=True)


# shutil.copy("pretrain.sh", output_model)
# for config_file in os.listdir():
#     if config_file.startswith("ds_config_zero") and config_file.endswith(".json"):
#         shutil.copy(config_file, output_model)


# os.environ["CUDA_HOME"] = "/usr/local/cuda/"
os.environ["NCCL_P2P_DISABLE"] = "1"

deepspeed_config_path = "./ds_config_zero3.json"


model_config_path = "/home/vipuser/Desktop/DeepSpeed/models_or_modelconfig/FlagAlpha/Atom-7B/config.json"
tokenizer_path = "/home/vipuser/Desktop/DeepSpeed/models_or_modelconfig/FlagAlpha/Atom-7B"
train_files = ["../../data/wiki_zh/train_lm_task_0.csv", "../../data/wiki_zh/train_lm_task_1.csv"]
validation_files = ["../../data/wiki_zh/dev_lm_task.csv"]


# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_config_path)


# DeepSpeed 初始化
def deepspeed_train():
    # DeepSpeed 训练参数
    training_args = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        logging_dir=os.path.join(output_model, "logs"),
        logging_strategy="steps",
        logging_steps=5,
        save_steps=100,
        eval_steps=5000000,
        save_total_limit=2000,
        seed=42,
        evaluation_strategy="steps",
        use_fast_tokenizer=False,
        max_eval_samples=500,
        warmup_steps=5000,
        block_size=1024,
        overwrite_output_dir=True,
        report_to="tensorboard",
        run_name=output_model,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        ignore_data_skip=True,
        ddp_timeout=18000000,
    )

    # 训练数据加载器
    from datasets import load_dataset

    train_dataset = load_dataset('csv', data_files=train_files)['train']
    eval_dataset = load_dataset('csv', data_files=validation_files)['train']

    # DeepSpeed 初始化
    model, optimizer, _, _ = deepspeed.initialize(
        args=training_args,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config_params={
            'deepspeed': deepspeed_config_path
        }
    )

    # Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=None,  # 使用默认的 data_collator
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    # import deepspeed
    deepspeed_train()

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

import json
import numpy as np
import pandas as pd
import os, re, json
os.environ["WANDB_DISABLED"] = "true"

from tqdm import tqdm
tqdm.pandas()



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.gradient_checkpointing_enable()
tokenizer.model_max_length=1024
system_prompt = """Bạn là Option Judge. Đánh nhãn cho từng phương án dựa trên:
- Ngữ cảnh/câu hỏi sau (tiếng Việt).
- Bản tóm tắt (distilled evidence) dưới đây.
- KHÔNG quy đổi đơn vị thời lượng sang số học; chỉ so sánh theo bậc đơn vị (giây < phút < giờ < ngày < tuần < tháng < năm) và theo ngôn ngữ tự nhiên.

Ngữ cảnh: {context}
Câu hỏi: {question}

Distilled evidence:
\"\"\"{distilled}\"\"\"

Các phương án: {opts_join}

Hãy trả về MỘT JSON ARRAY, mỗi phần tử tương ứng một phương án, theo schema:
{{
  "option": string,
  "label": "yes" | "no",
}}

Chỉ xuất JSON array, không giải thích thêm."""

def format_data(example):
    conversation = tokenizer.apply_chat_template(
        [
         {"role": "user", "content": system_prompt.format(distilled="", context = example['context'], question=example['question'], opts_join=json.dumps(example['options'], ensure_ascii=False))}, {"role": "assistant", "content": json.dumps([{"option": x, "label": y} for x, y in zip(example['options'], example['labels'])], ensure_ascii=False, indent=2)}],
        tokenize=False
    )
    tokenized = tokenizer(
        conversation,
        padding=True,
        truncation=True,
        return_tensors=None
    )
    return tokenized

data = []
with open("data/raw/duration_training_dataset.txt", 'r' ) as f:
    for x in f.readlines():
        x = x.strip()
        x = json.loads(x)
        data.append(x)
df_train = pd.DataFrame(data)

train_dataset = Dataset.from_pandas(df_train)
train_dataset = train_dataset.map(format_data)

training_args = SFTConfig(
    output_dir="./saved_dir",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    # eval_strategy="epoch",
    fp16=True,
    bf16=False,
    # report_to="none",
    completion_only_loss=False,
    # optim="adamw_8bit",
    optim="paged_adamw_8bit",
    max_seq_length=1024,
    gradient_checkpointing=True,  # Use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={'use_reentrant':False},  # Arguments for gradient checkpointing
    report_to="none"
)
response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    train_dataset=train_dataset.select_columns(['input_ids', 'attention_mask']),
    # eval_dataset=val_dataset,
    args=training_args,
    # formatting_func=lambda x: x["text"],
    data_collator=collator,

)

trainer.train()
trainer.save_model()

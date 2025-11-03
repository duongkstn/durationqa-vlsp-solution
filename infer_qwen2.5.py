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
model_name = "saved_dir"

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

def inference(example):
    conversation = tokenizer.apply_chat_template(
        [
         {"role": "user", "content": system_prompt.format(distilled="", context = example['context'], question=example['question'], opts_join=json.dumps(example['options'], ensure_ascii=False))}],
        tokenize=False, add_generation_prompt=True
    )
    tokenized = tokenizer(
        [conversation],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    output = tokenizer.decode(model.generate(**tokenized, max_new_tokens=512)[0][len(tokenized.input_ids[0]):], skip_special_tokens=True)
    return output

data = []
with open("data/raw/private_test.txt", 'r' ) as f:
    for x in f.readlines():
        x = x.strip()
        x = json.loads(x)
        data.append(x)
conversations = []
for example in data:
    conversation = tokenizer.apply_chat_template(
        [
         {"role": "user", "content": system_prompt.format(distilled="", context = example['context'], question=example['question'], opts_join=json.dumps(example['options'], ensure_ascii=False))}],        tokenize=False, add_generation_prompt=True
    )
    conversations.append(conversation)

answers = []
batch_size = 64
for j in tqdm(range(0, len(data), batch_size)):
    batch_conversations = conversations[j : j + batch_size]
    tokenized = tokenizer(
        batch_conversations,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    print('done tokenized')
    output = model.generate(**tokenized, max_new_tokens=512)
    print('done model.generate')

    for i, x in enumerate(data[j : j + batch_size]):
        answer = tokenizer.decode(output[i][len(tokenized.input_ids[i]):], skip_special_tokens= True)
        answers.append(answer)

df_test = pd.DataFrame(data)
df_test["answer"] = answers #df_test.progress_apply(lambda x: inference(x), axis=1)
df_test.to_csv("submit_private/inference_qwen2.5.csv", index=False)
print(df_test)

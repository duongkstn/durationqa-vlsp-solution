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
#model_name = "saved_dir"
model_name = "saved_dir_qwen3_prompt2"

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.model_max_length=512
system_prompt = """BẠN LÀ CHUYÊN GIA ĐÁNH GIÁ THỜI GIAN cho câu hỏi tiếng Việt.
NHIỆM VỤ: Đánh giá từng phương án thời gian có HỢP LÝ trong thực tế hay không (yes | no) cho hoạt động được mô tả.

VÍ DỤ THAM KHẢO:
---
{examples}

---
Ngữ cảnh: {context}
Câu hỏi: {question}

Các phương án: {opts_join}

Hãy trả về MỘT ARRAY, mỗi phần tử tương ứng một phương án theo thứ tự, theo schema:
["yes", "no", ....]

Chỉ xuất ARRAY, không giải thích thêm."""


def inference(example):
    conversation = tokenizer.apply_chat_template(
        [
         {"role": "user", "content": system_prompt.format(context = example['context'], question=example['question'], opts_join=json.dumps(example['options'], ensure_ascii=False))}],
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
data_retrieval = json.load(open("results_retrieval_private.json"))
conversations = []
for example, retrieval in zip(data, data_retrieval):
    examples = "\n---\n".join([f"""Ngữ cảnh: {x['context']}
Câu hỏi: {x['question']}
Các phương án: {json.dumps(x['options'], ensure_ascii=False)}
Answer: 
{json.dumps(x['labels'], ensure_ascii=False)}""" for x in retrieval[:2]])
    conversation = tokenizer.apply_chat_template(
        [
         {"role": "user", "content": system_prompt.format(examples = examples, context = example['context'], question=example['question'], opts_join=json.dumps(example['options'], ensure_ascii=False))}],        tokenize=False, add_generation_prompt=True
    )
    print(conversation)
    print('************************************')
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
df_test.to_csv("submit_private/inference_qwen3_retrieval_prompt2.csv", index=False)
print(df_test)

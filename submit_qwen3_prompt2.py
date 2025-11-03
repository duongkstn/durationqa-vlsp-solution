import json
import pandas as pd
import os, re


data = []
with open("data/raw/private_test.txt", 'r' ) as f:
    for x in f.readlines():
        x = x.strip()
        x = json.loads(x)
        data.append(x)

df = pd.read_csv("submit_private/inference_qwen3_retrieval_prompt2.csv")

answers = []
for i, answer in enumerate(df['answer'].values.tolist()):
    searched = re.search(r"\[((\"{1,2}(yes|no)\"{1,2}),\s*)*(\"{1,2}(yes|no)\"{1,2})\]", answer)
    if searched != None:
        
        kq = searched.group()
        kq = re.findall(r"yes|no", kq)
        kq = [x.lower().strip() if x.lower().strip() in ["yes", "no"] else "no" for x in kq]
        if len(kq) < len(data[i]['options']):
            kq = kq + ["no"] * (len(data[i]['options']) - len(kq))
        kq = kq[:len(data[i]['options'])]
    else:
        kq = ["no"] * len(data[i]['options'])
    data[i]["labels"] = kq

with open("submit_private/results.txt", 'a') as f:
    for x in data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

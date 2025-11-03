import json, os, re

files = [
    "submit_private/results_0.77_infer_qwen2.5.txt",
    "submit_private/results_0.78_infer_qwen3_prompt2.txt",
    "submit_private/results_0.80_infer_qwen3_retrieval_prompt2.txt",
    "submit_private/results_0.7911_infer_qwen3_prompt2_aug.txt",
    "submit_private/results_0.7927_infer_qwen3_retrieval_prompt2_aug.txt",
]

labels = []
data = []
for i, file in enumerate(files):
    label = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            label.append(line["labels"])
            if i == 0:
                del line["labels"]
                data.append(line)
    labels.append(label)

def major_vote(l):

    count_yes = l.count("yes")
    count_no = l.count("no")
    if count_yes >= count_no:
        return "yes"
    return "no"

for i, datum in enumerate(data):
    L = [
        labels[j][i]
        for j in range(len(files))
        ]  # len: so file
    T = [[L[j][k] for j in range(len(L))] for k in range(len(L[0]))]
    data[i]["labels"] = [major_vote(x) for x in T]

with open("submit_private/results.txt", 'w') as f:
    for x in data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")



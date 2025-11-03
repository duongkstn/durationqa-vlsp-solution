# ⏰ VLSP 2025 Challenge on TemporalQA — Subtask 2: DurationQA - First Rank Solution
**Team:** Engineers — *Đào Nguyên Dương, Nguyễn Xuân Thành ([https://github.com/xt2201](https://github.com/xt2201))*  
 
**Competition:** [https://vlsp.org.vn/vlsp2025/eval/tempqa](https://vlsp.org.vn/vlsp2025/eval/tempqa)
 
**Paper:** **Retrieval-Guided Fine-tuning for Vietnamese Event Duration Question Answering** (Duong Nguyen Dao, Thanh Xuan Nguyen)
 
**Presentation:** [https://www.youtube.com/watch?v=-klcvd7Ak8Q](https://www.youtube.com/watch?v=-klcvd7Ak8Q)

---
 
## 🚀 I. Approach  
 
Please refer to our paper for more details.
 
Our approach combines retrieval-based fine-tuning of large language models (Qwen2.5 and Qwen3) to enhance temporal reasoning and duration understanding in question answering tasks.
 
---
 
## 🛠 II. Training  
 
### Training Steps  
 
Training is conducted using these scripts, each corresponding to a specific configuration:
 
- **`train_qwen2.5.py`** — Fine-tune **Qwen2.5**, output LoRA weights saved in `saved_dir/`.  
- **`train_qwen3_prompt2.py`** — Fine-tune **Qwen3** using an List output format prompt, weights saved in `saved_dir_qwen3_prompt2/`.  
- **`train_qwen3_prompt2_aug.py`** — Fine-tune **Qwen3** with **cross-lingual data augmentation** from *UDST-DurationQA*, weights saved in `saved_dir_qwen3_prompt2_aug/`.
 
Each model is trained independently.
 
---
 
## ⚙️ III. Inference & Submission  
 
### Step 1: Retrieval  
 
Run retrieval scripts to compute similarity scores:
 
- **`retrieval.py`** → for **public test**
- **`retrieval_private.py`** → for **private test**
 
These scripts produce:
- `similarities.npy`, `similarities_private.npy` — Similarity matrices between test and training samples.
- `results_retrieval.json`, `results_retrieval_private.json` — Retrieved few-shot examples.
 
---
 
### Step 2: Inference  
 
Perform inference using **five configurations**:
 
| Script | Description |
|--------|--------------|
| `infer_qwen2.5.py` | Inference with the fine-tuned **Qwen2.5** model. |
| `infer_qwen3_prompt2.py` | Inference with the fine-tuned **Qwen3** model. |
| `infer_qwen3_retrieval_prompt2.py` | Inference with fine-tuned **Qwen3** and retrieved few-shot samples. |
| `infer_qwen3_prompt2_aug.py` | Inference with fine-tuned **Qwen3** using *UDST-DurationQA* augmentation. |
| `infer_qwen3_retrieval_prompt2_aug.py` | Inference with fine-tuned **Qwen3** using augmentation + retrieval-based few-shot samples. |
 
Output files are saved under `submit/inference_*.csv`.
 
---
 
### Step 3: Submission  
 
- Run **`submit_*.py`** scripts to clean inference results and produce final submission files (`results_*.txt`). The team only stored results from the public test runs and did not save results from the private test.
- To submit on AIHub, zip the final result:  
  ```bash
  zip -r results.zip results.txt
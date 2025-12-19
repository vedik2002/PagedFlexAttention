# HPML Project: Breaking the Granite-4 Ceiling — Memory-Efficient Inference with Paged FlexAttention

## Team Information
- **Team Name**: Team 13
- **Members**:
  - Vedik Agarwal (va2565)
  - Aditya Handur-Kulkarni (ash2285)
  - Siddhesh Fuladi (sf3320)
  - Jason Tan (ljt2143)

Department of Computer Science, Columbia University

---

## 1. Problem Statement
Large Language Models (LLMs) increasingly require long-context and high-concurrency inference, which places significant pressure on GPU memory due to key–value (KV) cache growth and fragmentation in attention mechanisms. While Paged Attention has shown promise for transformer-only architectures, its interaction with hybrid architectures remains underexplored.

This project studies the integration of **Paged Attention with FlexAttention** into **IBM Granite-4**, a hybrid architecture combining transformer self-attention and Mamba-style layers. The goal is to evaluate whether this integration preserves accuracy while improving memory efficiency, and to analyze its impact on inference latency, throughput, and GPU memory utilization. The work is strictly limited to **inference-time optimization** and does not involve retraining.

---

## 2. Model Description
- **Architecture**: IBM Granite-4.0-H-1B  
  - 24-layer decoder-only hybrid model  
  - Transformer self-attention + Mamba-style layers  
  - 16 attention heads  
  - Rotary positional embeddings  

- **Framework**:
  - PyTorch 2.9
  - Hugging Face Transformers
  - FlexAttention

- **Attention Variants**:
  - **Granite Baseline**: PyTorch SDPA / FlashAttention
  - **Granite with Paged Attention**: Paged Attention implemented using FlexAttention

- **Custom Modifications**:
  - A safe and modular patching strategy replaces Granite-4 self-attention layers with a paged attention wrapper **only during inference**
  - No changes to model weights or training dynamics

---

## 3. Final Results Summary

| Metric | MMLU (Baseline) | MMLU (Paged) | TruthfulQA (Baseline) | TruthfulQA (Paged) |
|------|------|------|------|------|
| Final Accuracy | 0.6786 | 0.6786 | 0.2778 | 0.3333 |
| Inference Latency (ms) | 859 | 1780 | 936 | 1305 |
| Throughput (QPS) | 1.16 | 0.56 | 1.07 | 0.77 |
| Input Token Throughput (tok/s) | 477 | 230 | 344 | 247 |
| Peak GPU Memory (GB) | 9.74 | 9.80 | 7.41 | 7.47 |
| Device | NVIDIA A100 80GB | NVIDIA A100 80GB | NVIDIA A100 80GB | NVIDIA A100 80GB |


---

## Batching Efficiency (Baseline vs Paged)

| Batch Size | Baseline Latency (ms) | Baseline QPS | Baseline Peak GPU (GB) | Paged Latency (ms) | Paged QPS | Paged Peak GPU (GB) |
| ---------: | --------------------: | -----------: | ---------------------: | -----------------: | --------: | ------------------: |
|          1 |                218.24 |         4.58 |                   5.08 |             289.84 |      3.45 |                5.15 |
|          2 |                262.78 |         3.81 |                   7.43 |             304.07 |      3.29 |                7.49 |
|          4 |                256.08 |         3.90 |                  12.12 |             276.02 |      3.62 |               12.19 |
|          8 |                253.37 |         3.95 |                  21.50 |             263.78 |      3.79 |               21.57 |
|         16 |                251.74 |         3.97 |                  40.28 |             257.56 |      3.88 |               40.34 |
|         32 |                251.91 |         3.97 |                  77.82 |             255.54 |      3.91 |               77.87 |

---

## Context Length Scalability (Baseline vs Paged)

| Context Length (tokens) | Baseline Avg Latency (ms) | Baseline ms/token | Baseline Peak GPU (GB) | Paged Avg Latency (ms) | Paged ms/token | Paged Peak GPU (GB) |
| ----------------------: | ------------------------: | ----------------: | ---------------------: | ---------------------: | -------------: | ------------------: |
|                     128 |                    618.55 |            4.7581 |                   5.08 |                1172.16 |         9.0166 |                5.15 |
|                     256 |                    819.35 |            3.1513 |                   7.41 |                1377.78 |         5.2991 |                7.47 |
|                     512 |                   1023.34 |            1.9680 |                   9.73 |                1578.74 |         3.0360 |                9.79 |
|                    1024 |                   1430.65 |            1.3890 |                  14.39 |                2002.98 |         1.9446 |               14.45 |
|                    2048 |                   2266.81 |            1.1058 |                  23.70 |                2101.57 |         1.0252 |               23.75 |
|                    4096 |                   3940.64 |            0.9611 |                  42.33 |                3819.25 |         0.9315 |               42.36 |

---


## 4. Reproducibility Instructions

### A. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
````

---

### B. WandB Dashboard

View evaluation metrics here:

**WandB Dashboard Link**: *https://api.wandb.ai/links/ash2285-cu/ivfnf8a1*

---

### C. Specify for Training or For Inference or if Both

This project focuses on **inference only**. Training is **not required** to reproduce the reported results.


---

### D. Evaluation

To evaluate the pretrained model:


```bash
python eval.py --mode <default | paged>
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce the minimum reported results:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

# Step 2: Evaluate inference 
python eval.py --mode <default | paged>

```

---

## 5. Repository Structure

```text
.
├── code/        # Core implementation of Granite Baseline and Paged Attention
├── notebooks/   # Profiling and analysis notebooks
├── figures/     # Figures used in the paper
├── results/     # Benchmark outputs and logs
└── readme.md
```

---

## 6. Notes

* All scripts are located in `code/`, `paged_attention.py`, `paged_granite.py`, `eval.py`, `plot.py`, and `tests.py`
* Figures are saved in `figures/`
* This repository focuses exclusively on **inference-time optimization**
* Small-batch latency overhead is expected due to page indirection

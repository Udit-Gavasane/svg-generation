# SVG Generation from Text Prompts — Deep Learning Midterm

NYU Tandon — Deep Learning (CS-GY 9223 / ECE-GY 7123) | Spring 2026

## Overview

This project fine-tunes a small language model to generate valid SVG (Scalable Vector Graphics) code from natural language text prompts. Given a description like "a red circle on white background", the model outputs valid SVG markup that visually matches the description.

This was developed as part of the [Kaggle competition](https://www.kaggle.com/competitions/dl-spring-2026-svg-generation-from-text-prompts-extended-deadline) for the NYU Deep Learning course.


---

## Repository Structure
```
svg-generation/
├── README.md
├── requirements.txt
├── training/
│   └── Deep_Learning_Midterm_4.ipynb   # Full training pipeline
├── inference/
│   └── generate.py                      # Local inference script
│   └── generate.py output.txt           # inference console output
└── results/
    └── submission_3b.csv                # Final submission file
```

---

## Model

- **Base model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) via Unsloth
- **LoRA config**: r=128, lora_alpha=128, target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Model Weights

Pre-trained LoRA adapters are available on Google Drive:
- **4-epoch adapter (primary submission)**: [svg_3b_adapter](https://drive.google.com/drive/folders/YOUR_LINK_HERE)

---

## Data

Download the competition data from Kaggle:
```bash
kaggle competitions download -c dl-spring-2026-svg-generation -p ./data
cd data && unzip dl-spring-2026-svg-generation.zip
```

Requires a Kaggle account and API key (`kaggle.json` placed at `~/.kaggle/kaggle.json`).

The dataset contains:
- `train.csv` — 50,000 (prompt, SVG) pairs
- `test.csv` — 1,000 prompts for inference
- `sample_submission.csv` — submission format

---

## Training

Training was done on Google Colab with H100 80GB GPU using Unsloth for fast LoRA fine-tuning.

### Training Run 1 — Primary (4 epochs, 47k samples)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Coder-3B-Instruct-bnb-4bit |
| LoRA rank | 128 |
| LoRA alpha | 128 |
| Epochs | 4 |
| Batch size | 16 |
| Gradient accumulation | 2 (effective batch = 32) |
| Learning rate | 1e-4 |
| Warmup steps | 200 |
| Optimizer | paged_adamw_8bit |
| LR scheduler | cosine |
| Max seq length | 4096 |
| Training samples | 46,699 (filtered from 50k) |
| GPU | H100 80GB |
| Training time | ~2 hours |

Data filtering: SVGs between 200-6000 characters that start with `<svg`.

### Training Run 2 — Continued Training (4 more epochs, 23k samples)

Built on top of Training Run 1 adapter with cleaner data and lower learning rate.

| Parameter | Value |
|-----------|-------|
| Starting weights | 4-epoch adapter |
| Epochs | 4 additional |
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Learning rate | 5e-5 (reduced to prevent overwriting) |
| Training samples | 23,342 (filtered to 200-2000 chars) |
| GPU | G4 (T4) |
| Training time | ~1.5 hours |

To run training:
1. Open `training/Deep_Learning_Midterm_4.ipynb` in Google Colab
2. Set runtime to GPU (H100 recommended)
3. Run all cells in order

---

## Inference

Inference was run locally on Windows with RTX 4080 Laptop GPU (12GB VRAM).

### Setup
```bash
pip install -r requirements.txt
```

### Download model weights

Download the adapter from Google Drive (link above) and place at:
```
D:\path\to\svg_3b_adapter\
```

Update `ADAPTER_PATH` in `generate.py` to match your local path.

### Run inference
```bash
python inference/generate.py
```

This generates `submission_3b.csv` with 1000 SVGs — one per test prompt.

### Inference configuration

| Parameter | Value |
|-----------|-------|
| max_new_tokens | 800 |
| do_sample | False (greedy decoding) |
| repetition_penalty | None |
| Fallback | Black circle SVG |

### Post-processing

Each generated SVG goes through:
1. Extract `<svg>...</svg>` block using regex
2. Fix width/height to 256x256
3. Fix viewBox to `0 0 256 256`
4. Add xmlns if missing
5. Remove disallowed tags (feGaussianBlur, animate, script, etc.)
6. Truncate to 8000 chars max
7. Validate with `xml.etree.ElementTree`
8. Fallback to black circle if invalid

---

## Results


---

## Ablations

### 1. Model size: 1.5B vs 3B
- 1.5B model generated repetitive loops on complex prompts
- 3B model produced more coherent SVG structure
- 3B scored higher despite same training setup

### 2. Base model vs fine-tuned
- Base model (no training): — generates valid SVGs but visually wrong
- Fine-tuned (4 epochs): — generates SVGs that match prompts

### 3. Fallback strategy
- All-fallback submission (black circle)
- This established the baseline — any valid generation above this is improvement

### 4. Post-processing approach
- Strict ET parsing: ~70% valid SVGs
- Lenient regex approach: ~98% valid SVGs, better final score
- Conclusion: SVG renderer is lenient, strict validation was counterproductive


---

## AI Tooling Disclosure

Claude (Anthropic) was used for:
- Debugging training and inference code
- Post-processing pipeline development
- Identifying and fixing SVG validation issues

---

## Reproducibility

Fixed seeds used throughout:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

All training hyperparameters are documented above and in the notebook.

---

## Requirements
```
unsloth
transformers>=4.40.0
peft
trl
accelerate
bitsandbytes
pandas
torch>=2.0.0
datasets
```

Install with:
```bash
pip install -r requirements.txt
```

# Multilingual Dense Passage Retrieval with LLM-Enhanced Hard Negatives

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Introduction

This project extends Dense Passage Retrieval (DPR) by introducing a progressive, LLM-enhanced hard negative mining pipeline that achieves up to 383% improvement over the BM25 baseline on MS MARCO. Our approach integrates LLM-powered negative classification, generation, and retrieval-augmented ranking for superior semantic-aware negative selection in multilingual settings.

**Extends Research**: ["A Study of Dense Passage Retrieval with Hard Negatives"](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/rajapakse-2024-study.pdf) (Yates & de Rijke, 2024) by adding LLM classification, generative negatives, and RAG-based ranking.

## Key Highlights

- **383% performance gain** over BM25 retrieval with RAG enhancements
- **Progressive mining** via four hard negative mining strategies
- **Multilingual support**: English (MS MARCO), TyDi QA (9 languages), mMARCO (13 languages)
- **Local LLM integration** with Ollama for scalable filtering and generation
- **Production-ready** preprocessing with automated validation and error handling
- **Comprehensive evaluation** with MRR@10, Recall@K, nDCG@10

## Pipeline Overview

| Phase | Strategy | Key Innovation |
|-------|----------|----------------|
| **Phase 1** | BM25 Baseline | Traditional BM25 hard negative mining |
| **Phase 2** | Data Preparation | Quality filtering & validation |
| **Phase 3** | LLM-Enhanced | Ollama-based negative classification |
| **Phase 4** | RAG-Enhanced | FAISS-powered context-aware selection |
| **Phase 5** | Multilingual | Fine-tuning on 15+ languages |

## Novel Contributions Beyond Base Research

While traditional DPR training relies on **BM25-based negative sampling** that suffers from lexical overlap bias and generates semantically weak negatives, our work introduces **three progressive enhancement techniques**:

### 1. LLM-Powered Binary Classification
**Problem**: BM25 generates false negatives that mislead training  
**Solution**: Ollama-based intelligent filtering distinguishes true hard negatives (score 70-100) from false negatives  
**Impact**: Quality control layer absent in baseline approaches

### 2. Query-Conditioned Hard Negative Generation
**Problem**: BM25 relies only on lexical matching  
**Solution**: LLM generates semantically challenging negatives by understanding query intent and creating topically related but factually incorrect passages  
**Impact**: Moves beyond simple keyword matching to semantic difficulty

### 3. RAG-Enhanced Ranking Mechanism
**Problem**: Selected negatives may not be contextually appropriate  
**Solution**: FAISS-powered retrieval scores negatives using contextual information, ensuring semantic space alignment  
**Impact**: Novel contribution - adaptive selection based on retrieval context (not in base paper)

## Performance Results

### MS MARCO English Retrieval (300 samples)

| Model | MRR@10 | Recall@10 | nDCG@10 | Improvement |
|-------|--------|-----------|---------|-------------|
| **BM25 Baseline** | 0.0259 | 6.3% | 0.0346 | — |
| **LLM-Enhanced** | 0.0713 | 25.0% | 0.1129 | +189% |
| **RAG-Enhanced** ⭐ | **0.1237** | **30.7%** | **0.1662** | **+383%** |

### Multilingual Evaluation (TyDi QA)

#### In-Distribution Languages (Swahili, Bengali, Telugu)
| Model | MRR@10 | Recall@10 | nDCG@10 |
|-------|--------|-----------|---------|
| BM25 Baseline | 0.4430 | 74.3% | 0.5153 |
| LLM Enhanced | 0.4278 | 74.7% | 0.5045 |
| **RAG Enhanced** ⭐ | **0.4452** | **76.0%** | **0.5215** |

#### Out-of-Distribution Languages (Arabic, Japanese, Indonesian, Russian)
| Model | MRR@10 | Recall@10 | nDCG@10 |
|-------|--------|-----------|---------|
| BM25 Baseline | 0.3661 | 66.9% | 0.4376 |
| **LLM Enhanced** ⭐ | **0.4398** | **76.2%** | **0.5164** |
| RAG Enhanced | 0.4304 | 74.3% | 0.5052 |

#### Zero-Shot Languages (Chinese, French, Dutch, German)
| Model | MRR@10 | Recall@10 | nDCG@10 |
|-------|--------|-----------|---------|
| BM25 Baseline | 0.3315 | 64.5% | 0.4059 |
| **LLM Enhanced** ⭐ | **0.4698** | **77.1%** | **0.5472** |
| RAG Enhanced | 0.4552 | 75.2% | 0.5306 |

<img width="5364" height="1779" alt="phase5_comparison" src="https://github.com/user-attachments/assets/a1715e43-243c-473a-8a38-d60c8a631968" />



### Key Findings

- **RAG-Enhanced model achieves 383% improvement** over BM25 baseline on MS MARCO
- **LLM-Enhanced excels in out-of-distribution** and zero-shot multilingual scenarios
- **Progressive mining strategies** demonstrate consistent performance gains
- **Best model varies by language**: RAG for in-distribution, LLM for zero-shot

## Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060/4060 or better)
- **CUDA**: 11.8 or higher
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for datasets and models

### Software Setup

```bash
# Create conda environment
conda create -n mldpr python=3.10
conda activate mldpr

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install FAISS with GPU support
conda install -c pytorch -c nvidia faiss-gpu=1.7.4

# Install dependencies
pip install transformers==4.35.0 datasets==2.14.0 pandas tqdm
pip install simpletransformers==0.64.3 sentence-transformers==2.2.2
pip install rank-bm25 requests
```

### Ollama Setup (Required for Phase 3)

```bash
# 1. Install Ollama from https://ollama.ai/download

# 2. Pull the model (8B parameter Llama 3.1)
ollama pull llama3.1:8b

# 3. Start Ollama server (keep running in separate terminal)
ollama serve
```

## Quick Start

### 1. Download Data

```bash
bash download_data.sh
```

### 2. Full Training Pipeline

```bash
bash train.sh
```

Runs all phases sequentially with validation checks.

### 3. Evaluation

```bash
bash evaluate.sh
```

Evaluates all trained models on MS MARCO dev set.

## Project Structure

```
Multilingual-DPR-Retrieval-System/
├── data/                # Datasets and processed data
├── models/              # Trained models by phase
├── data_processing/     # Preprocessing & negative mining scripts
├── train_scripts/       # Phase-specific training scripts
├── evaluation_scripts/  # Model evaluation code
├── results/             # Metrics & evaluation results
├── train.sh             # Automation script
├── evaluate.sh          # Model evaluation script
└── README.md            # Project documentation
```

## Usage Guide

### Full Automated Pipeline

```bash
bash train.sh
```

**Expected Output:**
- Phase 1: BM25 baseline model (~2 hours)
- Phase 3: LLM-enhanced model (~4 hours with Ollama)
- Phase 4: RAG-enhanced model (~3 hours)
- Phase 5: Multilingual fine-tuned models (~2 hours)

### Individual Phase Training

#### Phase 1: BM25 Baseline

```bash
# Step 1: Download datasets
bash download_data.sh

# Step 2: Mine BM25 hard negatives
python data_processing/prepare_bm25_negatives.py

# Step 3: Train baseline DPR model
python train_scripts/train_phase1_bm25_baseline.py
```

#### Phase 3: LLM-Enhanced

```bash
# Prerequisites: Ollama running in separate terminal
ollama serve

# Step 1: Classify negatives and generate new hard negatives
python data_processing/llm_classification.py

# Step 2: Train with LLM-enhanced negatives
python train_scripts/train_phase3_llm_enhanced.py
```

#### Phase 4: RAG-Enhanced

```bash
# Step 1: Build FAISS index and score negatives
python data_processing/rag_selection.py

# Step 2: Train with RAG-selected negatives
python train_scripts/train_phase4_rag_enhanced.py
```

#### Phase 5: Multilingual Fine-tuning

```bash
# Fine-tune all models on TyDi QA (9 languages)
python train_scripts/train_phase5_multilingual_finetune.py
```

### Evaluation

```bash
bash evaluate.sh
```

Generates evaluation metrics in `results/` directory with MRR@10, Recall@K, nDCG@10.

## Technical Details

### Progressive Hard Negative Mining

#### Phase 1: BM25 Mining (Baseline)
- Traditional lexical matching (BM25Okapi)
- Top-100 retrieval with relevance filtering
- Removes exact positive matches

#### Phase 3: LLM Classification (Novel - Quality Filtering)
- Binary classification: HARD (70-100) vs EASY (0-30) using Ollama llama3.1:8b
- Generative augmentation: Creates 3 query-conditioned hard negatives per sample
- Semantic understanding: Analyzes query intent, not just keyword matching
- Checkpoint resilience: Saves progress every 100 samples

#### Phase 4: RAG Selection (Novel - Context-Aware Ranking)
- FAISS IndexFlatIP: Builds semantic index from hard negatives corpus
- Context retrieval: Top-3 similar passages per query for informed scoring
- LLM + RAG scoring: Evaluates negatives with retrieved contextual knowledge
- Adaptive selection: Top 50% based on semantic appropriateness

### Model Architecture

- **Base Model**: `bert-base-multilingual-cased` (110M parameters)
- **Encoder**: Dual-encoder architecture (query + passage encoders)
- **Training**: Contrastive learning with in-batch negatives + mined hard negatives
- **Optimization**: AdamW with linear warmup, FP16 mixed precision

### Evaluation Metrics

- **MRR@10**: Mean Reciprocal Rank at top-10
- **Recall@K**: Proportion of relevant passages in top-K (K=1,5,10)
- **nDCG@10**: Normalized Discounted Cumulative Gain at top-10

## Datasets

| Dataset | Language(s) | Size | Purpose |
|---------|------------|------|---------|
| [MS MARCO](https://microsoft.github.io/msmarco/) | English | 8.8M passages | Training & evaluation |
| [TyDi QA](https://github.com/google-research-datasets/tydiqa) | 9 languages | ~200K questions | Multilingual fine-tuning |
| [mMARCO](https://github.com/unicamp-dl/mMARCO) | 13 languages | 8.8M passages | Zero-shot evaluation |

**Supported Languages (15 total)**:
- **TyDi QA**: Arabic, Bengali, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, Thai
- **mMARCO**: Arabic, Chinese, Dutch, French, German, Hindi, Indonesian, Italian, Japanese, Portuguese, Russian, Spanish, Vietnamese

## Citation

```bibtex
@misc{prakash2025multilingual-dpr-llm,
  title={Multilingual Dense Passage Retrieval with LLM-Enhanced Hard Negatives},
  author={Divyanshu Prakash, Deepak Kumar, Aditya Prakash},
  year={2025},
  howpublished={\url{https://github.com/divyanshu-prakash-rx/Multilingual-DPR-Retrieval-System-with-LLM-Enhanced-Hard-Negatives}},
  note={Novel contributions: LLM classification layer, query-conditioned generation, 
        and RAG-enhanced ranking for hard negative mining. Extends traditional 
        BM25-based approaches with semantic-aware selection achieving 383\% improvement.}
}
```

### Base Research (Extended By This Work)

```bibtex
@article{yates2024study,
  title={A Study of Dense Passage Retrieval with Hard Negatives},
  author={Thilina C. Rajapakse, Andrew Yates, Maarten de Rijke},
  journal={University of Amsterdam},
  year={2024},
  url={https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/rajapakse-2024-study.pdf},
  note={Compared BM25, clustering, and dense index methods for monolingual 
        and multilingual retrieval}
}
```

## Acknowledgments

This project extends and builds upon:

- **["A Study of Dense Passage Retrieval with Hard Negatives"](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/rajapakse-2024-study.pdf)** - Foundational research comparing negative sampling techniques by Yates & de Rijke (2024)
- **[DPR](https://github.com/facebookresearch/DPR)** - Dense Passage Retrieval architecture (Facebook AI)
- **[SimpleDPR](https://github.com/ThilinaRajapakse/simpletransformers)** - Simple Transformers library
- **[Ollama](https://ollama.ai/)** - Local LLM inference for classification and generation
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient similarity search for RAG ranking (Meta AI)
- **MS MARCO, TyDi QA, mMARCO** - High-quality retrieval benchmarks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or collaboration opportunities:

- **GitHub**: [@divyanshu-prakash-rx](https://github.com/divyanshu-prakash-rx)
- **Repository**: [Multilingual-DPR-Retrieval-System-with-LLM-Enhanced-Hard-Negatives](https://github.com/divyanshu-prakash-rx/Multilingual-DPR-Retrieval-System-with-LLM-Enhanced-Hard-Negatives)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐

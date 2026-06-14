# SharpGPT

A GPT-style autoregressive language model implemented from scratch using PyTorch and PyTorch Lightning.

The project covers tokenizer training, large-scale pretraining, Transformer architecture implementation, custom optimization, dataset preparation, and training infrastructure.

SharpGPT is pretrained on **6 billion tokens of NVIDIA ClimbMix** and serves as a foundation for experimentation with modern language model architectures and efficient training techniques.

---

## Model Specifications

| Parameter | Value |
|------------|------------|
| Architecture | Decoder-only Transformer |
| Training Tokens | 6 Billion |
| Context Length | 1,024 |
| Layers | 20 |
| Hidden Dimension | 1,024 |
| Attention Heads | 16 |
| KV Heads | 4 |
| Head Dimension | 64 |
| MLP Dimension | 4,096 |
| Vocabulary Size | 32,786 |
| Value Embedding Rank | 16 |
| Backbone Parameters | ~224M |
| Token Embeddings | ~69M |
| Value Embeddings | ~330M |
| Total Parameters | ~623M |

---

## Features

### Architecture

- Decoder-only Transformer
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- Pre-Norm Transformer blocks
- Grouped Query Attention (GQA)
- Query RMSNorm
- Key RMSNorm
- ReLU² feed-forward networks
- DeepSeek-style residual scaling
- Alternate-layer value embeddings
- Flash Attention support when available

### Training Infrastructure

- Custom Muon–AdamW hybrid optimizer
- PyTorch Lightning integration
- Dataclass-based configuration system
- Modular training pipeline
- Reproducible experiment setup
- Custom learning rate scheduling

### Data Pipeline

- Large-scale corpus preprocessing
- Efficient token generation workflow
- Streaming-compatible dataset preparation
- Optimized PyTorch DataLoaders
- High-throughput batch generation

### Tokenization

- Custom tokenizer training pipeline
- 32K vocabulary tokenizer
- Reusable tokenizer artifacts

---

## Training Data

### Base Pretraining

- **6 Billion Tokens of NVIDIA ClimbMix**

---

## Repository Structure

```text
Project/
│
├── Model_dir/
│   ├── DataLoaders.py
│   ├── HyperParam_Classes.py
│   ├── Model_Classes.py
│   ├── Optimizer.py
│   ├── prepare_pretraining_data.py
│   ├── smoke_test.py
│   └── train.py
│
├── tokenizers_dir/
│   ├── DataPrep.py
│   ├── Tokenizer_train.py
│   └── tokenizer_32k.json
│
├── val_loss/
│   ├── training_log.json
│   ├── val_delta_loss.png
│   └── val_loss.png
│
├── LICENSE
└── README.md
```

---

## Architecture Overview

### Embedding Layer

Input tokens are mapped into dense vector representations through a learned embedding table.

The language modeling head shares weights with the token embeddings through weight tying.

---

### Attention Layer

Each Transformer block applies:

- Query, Key, and Value projections
- Query & Key RMSNorm
- Rotary positional encoding
- Causal self-attention
- Output projection

---

### Feed Forward Network

```text
Input
  │
  ▼
Linear Expansion
  │
  ▼
ReLU²
  │
  ▼
Linear Projection
  │
  ▼
Output
```

---

### Transformer Block

```text
Input
 │
 ├── RMSNorm
 ├── Attention
 ├── Residual
 │
 ├── RMSNorm
 ├── MLP
 ├── Residual
 │
 ▼
Output
```

Residual branches are scaled by:

```text
1 / √(2L)
```

where **L** is the number of Transformer layers.

---


## Available Tokenizers

| Tokenizer | Vocabulary Size |
|------------|------------|
| tokenizer_32k.json | 32,768 |

---

## Tech Stack

- Python
- PyTorch
- PyTorch Lightning
- Hugging Face Datasets
- Hugging Face Tokenizers
- NumPy

---

## Project Goal

SharpGPT is an end-to-end language model training project focused on building modern GPT-style architectures from the ground up.

The project explores:

- Tokenizer training
- Dataset preparation
- Transformer implementation
- Custom optimization techniques
- Large-scale language model pretraining
- Efficient training infrastructure

while maintaining a modular, reproducible, and extensible codebase for future experimentation.

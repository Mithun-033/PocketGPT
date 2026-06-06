# MineGPT

<p align="center">
  <img src="MineGPT2.png">
</p>

MineGPT is a GPT-style autoregressive language model built from scratch using PyTorch and PyTorch Lightning. The project covers the complete language model development pipeline, including tokenizer training, dataset preparation, model implementation, custom optimization, training infrastructure, and domain-specific continued pretraining.

The model is pretrained on **1 billion tokens from NVIDIA ClimbMix** and further trained on **Minecraft Wiki** and **Minecraft Question & Answer datasets** to specialize its knowledge in the Minecraft domain.

---

## Features

### Architecture

- Decoder-only GPT architecture
- Rotary Positional Embeddings (RoPE)
- RMSNorm normalization
- Pre-Norm Transformer blocks
- Multi-Head Self Attention
- PyTorch Scaled Dot Product Attention (SDPA)
- Flash Attention kernels when supported by hardware
- Query RMSNorm
- Key RMSNorm
- ReLU² feed-forward networks
- DeepSeek style scaled Residual paths [1 / √(2L)]
- Weight tying between token embeddings and output projection head

### Optimization

- Custom Muon–AdamW hybrid optimizer
- Hand-built learning rate scheduler
- Modular optimizer implementation
- Decoupled optimization pipeline

### Training Infrastructure

- PyTorch Lightning Trainer
- PyTorch Lightning DataModule
- Dataclass-driven hyperparameter management
- Modular training pipeline
- Reproducible experiment configuration

### Data Pipeline

- Large-scale pretraining data preparation pipeline
- Efficient tokenized dataset generation
- Optimized PyTorch DataLoaders
- Throughput-focused DataLoader configuration
- Streaming-friendly preprocessing workflow

### Tokenization

- Custom tokenizer training pipeline
- 16K vocabulary tokenizer
- 32K vocabulary tokenizer
- Reusable tokenizer artifacts stored as JSON

---

## Training Data

### Base Pretraining

MineGPT is pretrained on:

- **1 Billion Tokens of NVIDIA ClimbMix**

### Continued Pretraining

After base pretraining, the model undergoes additional Minecraft-focused training using:

- Minecraft Wiki
- Minecraft Question & Answer datasets

This continued training improves performance on Minecraft-related terminology, mechanics, entities, crafting systems, and gameplay knowledge.

---

## Repository Structure

```text
MineGPT/
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
│   ├── tokenizer_16k.json
│   └── tokenizer_32k.json
│
├── MineGPT2.png
├── LICENSE
└── README.md
```
---

## Component Overview

### DataLoaders.py

Contains:

- PyTorch Lightning DataModule
- Optimized PyTorch DataLoaders
- Dataset loading utilities
- Batching and training data pipeline

### HyperParam_Classes.py

Contains:

- Dataclass-based hyperparameter definitions
- Model configuration classes
- Training configuration classes

### Model_Classes.py

Contains the complete GPT architecture implementation including:

- Token embeddings
- RoPE positional encoding
- Multi-head self attention
- RMSNorm layers
- Transformer blocks
- ReLU² MLP layers
- Language modeling head

### Optimizer.py

Contains:

- Custom Muon–AdamW hybrid optimizer
- Custom learning rate scheduler
- Optimization utilities

### prepare_pretraining_data.py

Contains:

- Dataset preprocessing pipeline
- Token generation workflow
- Pretraining dataset preparation

### train.py

Contains:

- PyTorch Lightning training workflow
- Model initialization
- Trainer configuration
- Training loop orchestration

### tokenizers_dir/

Contains:

- Tokenizer training pipeline
- Tokenizer data preparation
- Trained tokenizer artifacts

---

## Architecture Overview

### Embedding Layer

The model begins with a token embedding layer that converts token IDs into dense vector representations.

The output language modeling head shares weights with the token embedding layer through weight tying.

---

### Attention Layer

Each transformer block contains:

- Linear QKV projection
- Query RMSNorm
- Key RMSNorm
- Rotary Positional Embeddings (RoPE)
- Causal self-attention
- Scaled Dot Product Attention
- Output projection layer

---

### Feed Forward Network

The MLP layer consists of:

```text
Input
  │
  ▼
Linear Up Projection
  │
  ▼
ReLU² Activation
  │
  ▼
Linear Down Projection
  │
  ▼
Output
```

---

### Transformer Block

Each transformer block follows a Pre-Norm design:

```text
Input
 │
 ├── RMSNorm
 ├── Multi-Head Attention
 ├── Residual Connection
 │
 ├── RMSNorm
 ├── ReLU² MLP
 ├── Residual Connection
 │
 ▼
Output
```

Residual branches are scaled by a factor of 1 / √(2L),

where L is the number of transformer layers.

---

## Training Pipeline

### 1. Train Tokenizer

```bash
python tokenizers_dir/Tokenizer_train.py
```

### 2. Prepare Pretraining Data

```bash
python prepare_pretraining_data.py
```

### 3. Configure Hyperparameters

All model and training hyperparameters are defined through dataclass-based configuration classes inside:

```text
HyperParam_Classes.py
```

### 4. Start Training

```bash
python train.py
```

Training is orchestrated through PyTorch Lightning using dedicated Lightning Modules and Lightning DataModules.

---

## Available Tokenizers

| Tokenizer | Vocabulary Size |
|------------|----------------|
| tokenizer_16k.json | 16,384 |
| tokenizer_32k.json | 32,768 |

These tokenizers can be used directly for pretraining, finetuning, and inference workflows.

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

MineGPT is an end-to-end language model training project designed to explore modern GPT training techniques while maintaining a fully modular codebase.

The project covers:

- Tokenizer training
- Dataset preparation
- GPT architecture implementation
- Custom optimization strategies
- Efficient training infrastructure
- Large-scale language model pretraining
- Domain-specific continued pretraining

with a focus on building a Minecraft-specialized language model from the ground up.

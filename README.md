# Torch-Velocity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/Torch-Velocity/blob/main/velocity_demo.ipynb)

**Adaptive Speculative Decoding for LLM Inference Optimization**

An implementation of speculative decoding with adaptive lookahead (γ), demonstrating 1.5-2.5x inference speedups on transformer models.

## The Problem

Large Language Models are **memory-bandwidth bound**, not compute-bound. When generating tokens autoregressively, we spend most of our time moving weights from VRAM to compute units—even for trivial continuations like "the" or "and".

## The Solution

**Speculative Decoding** uses a small, fast "draft" model to speculatively generate K tokens, then verifies them all in a single parallel forward pass through the large "target" model.

This implementation adds **adaptive γ** (lookahead length):
- High acceptance rate → increase γ (be aggressive)
- Low acceptance rate → decrease γ (be conservative)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Speculative Decoding Loop                │
├─────────────────────────────────────────────────────────────┤
│  1. DRAFT: Small model generates γ tokens (fast)            │
│  2. VERIFY: Large model scores all γ tokens in ONE pass     │
│  3. ACCEPT/REJECT: Rejection sampling per Leviathan et al.  │
│  4. ROLLBACK: Rewind KV cache to valid state                │
│  5. ADAPT: Adjust γ based on acceptance rate                │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

- **KVCacheManager**: Pre-allocated key-value cache with O(1) rollback
- **Rejection Sampling**: Mathematically guaranteed to match target distribution
- **Adaptive γ**: Dynamic lookahead based on real-time acceptance rates

## Usage

Open `velocity_demo.ipynb` in Google Colab (free T4 GPU) and run all cells.

```bash
# Local setup
pip install -r requirements.txt
jupyter notebook velocity_demo.ipynb
```

## Models

| Role   | Model         | Parameters |
|--------|---------------|------------|
| Draft  | distilgpt2    | 82M        |
| Target | gpt2-medium   | 355M       |

## References

1. Leviathan et al. (2023) - "Fast Inference from Transformers via Speculative Decoding"
2. Chen et al. (DeepMind, 2023) - "Accelerating Large Language Model Decoding with Speculative Sampling"
3. SpecDec++ (2024) - Adaptive candidate lengths

## Author

Matt McManus

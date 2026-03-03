# GPT-2 (124M) PyTorch Implementation

## Overview
This repository contains a strictly first-principles implementation of a 124-million parameter Generative Pre-trained Transformer (GPT-2). 

Rather than relying on high-level library abstractions like the Hugging Face `Trainer`, this project focuses on the foundational mathematics and software engineering required to build frontier models from scratch in PyTorch, utilizing the architectural blueprint from Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Architecture Highlights
* **Custom Self-Attention:** Ground-up implementation of Causal and Multi-Head Attention, manually deriving the matrix projections, causal masking, and scaling.
* **Optimized Data Pipeline:** Custom PyTorch `DataLoader` and `Dataset` classes integrating the `tiktoken` BPE tokenizer with sliding window stride logic for efficient batching.
* **First-Principles Design:** Explicit handling of absolute positional embeddings, layer normalization, and GELU activations to match the original OpenAI specification.

## Repository Structure
* `GPT2_Implementation.ipynb`: An interactive walkthrough of the architecture, data loading, and tensor shape verifications.
* *(Modular .py files in active development)*

## Current Status: Active Development
* [x] Forward Pass & Architecture Design
* [x] Custom Tokenization Pipeline
* [x] Cross-Entropy Loss Calculation
* [ ] Weight Tying Optimization
* [ ] Full-Scale Training Loop Execution

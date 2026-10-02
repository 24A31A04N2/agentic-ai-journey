# 🤖 Day 14/250 — AI/ML Landscape — Conceptual Foundations

![Day](https://img.shields.io/badge/Day-14/250-blue)
![Phase](https://img.shields.io/badge/Phase-1-purple)
![Language](https://img.shields.io/badge/Language-Python_3.11-yellow)

> *"To build agents that can reason and act, we must first deeply understand the statistical engine that gives them language and logic."*

## 📚 What I Learned Today

| Concept | Description |
|---------|-------------|
| **AI Taxonomy** | Understanding the clear evolution from Rule-Based AI -> ML -> DL -> GenAI -> Agentic AI. |
| **Neural Networks** | First principles of layers, weights, biases, and backpropagation using a simple XOR example. |
| **Next-Token Prediction** | How LLMs essentially operate as massive statistical engines predicting the next word based on probability. |
| **Self-Attention** | The core of the Transformer architecture, explaining how words weigh their relevance to every other word in a sequence. |
| **Encoder vs Decoder** | The structural difference between bidirectional models (BERT) and autoregressive models (GPT). |
| **Agentic Frontier** | Why the shift from static text generation to autonomous tool-use and reasoning loops is the next logical step. |

## 🛠️ What I Built

- A multi-section conceptual Python script (`day14_ai_ml_landscape.py`) utilizing pure Python and NumPy.
- A **2-layer Neural Network** trained from scratch to solve the XOR problem.
- A **Bigram Language Model** demonstrating rudimentary next-token prediction.
- A scaled dot-product **Self-Attention** mechanism simulator.
- A matrix-based visualization comparing **Encoder (BERT-style)** vs **Decoder (GPT-style)** masking.

## 💻 Code Highlights

### Self-Attention Core Mechanism
```python
# Scaled dot-product attention
scores = np.dot(Q, K.T) / np.sqrt(d_k)

# Softmax for attention weights
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Contextualized output
output = np.dot(attention_weights, V)
```

## 🚀 Run It

```bash
cd day-14
python day14_ai_ml_landscape.py
```

## 🧠 Why This Matters for Agents
Agentic AI isn't magic; it is an orchestration layer built on top of autoregressive decoders. Understanding attention, causal masking, and transition probabilities helps in debugging hallucinations, optimizing prompt context windows, and building reliable agent reasoning loops (like ReAct).

## 📖 Resources

| Resource | Description |
|----------|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The seminal paper introducing the Transformer architecture. |
| [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Fantastic foundational video on from-scratch language modeling. |

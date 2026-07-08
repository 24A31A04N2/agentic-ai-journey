# 📐 Day 15/150 — NumPy & Vector Mathematics

![Day](https://img.shields.io/badge/Day-15%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-2%3A%20AI%2FML%20Essentials-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)

> **Key Insight:** Under the hood, LLMs and Vector DBs don't read words — they process matrices of high-dimensional vectors. Understanding dot products, normalization, and cosine similarity is non-negotiable for RAG, attention mechanisms, and routing agents.

---

## 📌 What I Learned Today

| Concept | What It Does | Agent Application |
|---------|-------------|-------------------|
| **Broadcasting & Vectorization** | Perform element-wise operations on arrays without explicit `for` loops | Blazing-fast similarity calculation over millions of doc embeddings |
| **Dot Product** | Measures the projection of one vector onto another | Core operation of cosine similarity and projection layers |
| **Vector Normalization** | Rescales vector magnitude to 1 | Prepares embedding vectors so that dot product equals cosine similarity |
| **Cosine Similarity** | Calculates the cosine of the angle between two vectors ($\cos \theta$) | Core similarity metric for RAG document retrieval |
| **Matrix Multiplication** | Multiplies matrices to map features to different spaces | Drives the attention mechanism ($Q \times K^T$) in Transformers |
| **Random Projection** | Compresses high-dimensional vectors randomly | Dimensionality reduction for vector embeddings while preserving distance |

---

## 🔨 What I Built

### `day15_vector_math.py`
A comprehensive, zero-dependency NumPy walkthrough demonstrating the mathematical foundations of modern AI:

- **Section 1: NumPy Basics** — Creating arrays, inspecting shapes, dtypes, and leveraging broadcasting.
- **Section 2: Vector Operations** — Dot products, Euclidean distance, Manhattan distance, and unit normalization.
- **Section 3: Cosine Similarity** — Hand-cranked cosine similarity calculation and batch calculation.
- **Section 4: Attention Mechanism Math** — Visualizing Q, K, V matrix multiplications that power self-attention.
- **Section 5: Simulated Embeddings** — Translating mock text into numerical vectors and measuring similarities.
- **Section 6: Mini Semantic Search Engine** — A complete local indexing and retrieval class using cosine similarity.
- **Bonus Utilities** — Fast top-K selection with `np.argpartition`, temperature scaling for logits, and random projections.

---

## 📂 Code Highlights

### Manual Cosine Similarity Calculation
```python
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))
```

### Self-Attention Calculation
```python
def compute_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Formula: Softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    d_k = Q.shape[1]
    attention_scores = (Q @ K.T) / np.sqrt(d_k)
    attention_weights = softmax(attention_scores)
    context_vectors = attention_weights @ V
    return context_vectors
```

### Vector Store Retrieval
```python
# Batch cosine similarity against all document embeddings at once:
similarities = batch_cosine_similarity(query_embedding, self.embeddings)
# Sort indices in descending order
top_indices = np.argsort(similarities)[::-1][:top_k]
```

---

## ▶️ Run It

Make sure you have NumPy installed:
```bash
pip install numpy
```

Run the interactive demonstration:
```bash
python day15_vector_math.py
```

---

## 🧠 Why This Matters for Agents

1. **RAG (Retrieval-Augmented Generation)**: Standard databases search by keywords. Vector databases search by semantic meaning using cosine similarity.
2. **LLM Mechanics**: LLMs generate token probabilities (logits). The `temperature` parameter scales these logits using a softmax function to control output randomness.
3. **Agent Routers**: By comparing a user's prompt embedding to pre-defined task embeddings, agents can instantly select the best tool or path to take.

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| NumPy Documentation | https://numpy.org/doc/stable/ |
| Vector Embeddings Explained | https://vickiboykis.com/what_are_embeddings/ |
| Math of Self-Attention | https://transformer-circuits.pub/2021/framework/index.html |
| Cosine Similarity Theory | https://en.wikipedia.org/wiki/Cosine_similarity |
| Softmax Function | https://en.wikipedia.org/wiki/Softmax_function |

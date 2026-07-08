"""
🤖 Day 15/150 — NumPy & Vector Mathematics — The Math Behind Embeddings
========================================================================
Phase 2: AI/ML Essentials (Days 15–28)

Concepts:
  1. NumPy Fundamentals — Arrays, shapes, dtypes, broadcasting, vectorized ops
  2. Vector Operations — Dot product, magnitude, normalization, distance metrics
  3. Cosine Similarity — THE core metric for semantic search in RAG agents
  4. Matrix Operations — Multiplication, transpose, inverse — how attention works
  5. Embedding Simulation — Simulating word/sentence embeddings as vectors
  6. Practical Agent Application — Mini semantic search engine with cosine similarity

Why this matters for Agentic AI:
  - Every modern AI agent thinks in VECTORS. When a user sends a query, it's
    converted into a dense vector (embedding). The agent searches a vector
    database for the most similar documents — that's Retrieval-Augmented
    Generation (RAG).
  - Cosine similarity decides which documents are "relevant" to a query.
    Understanding it means understanding WHY your agent retrieves what it does.
  - The attention mechanism in every Transformer (GPT, Claude, Gemini) is
    literally Q·Kᵀ — a matrix multiplication followed by softmax. If you don't
    understand dot products and matrix ops, attention is a black box.
  - Embeddings ARE the language of AI. Text → Vector → Math → Answer.
    This day gives you the mathematical fluency to debug, optimize, and
    reason about every vector operation your agents will ever perform.

Section Map:
  Section 1: NumPy Fundamentals .................... lines ~55–220
  Section 2: Vector Operations ..................... lines ~220–420
  Section 3: Cosine Similarity ..................... lines ~420–570
  Section 4: Matrix Operations ..................... lines ~570–720
  Section 5: Embedding Simulation .................. lines ~720–870
  Section 6: Mini Semantic Search Engine ........... lines ~870–1000+
"""

import sys
import time

# Ensure UTF-8 output on Windows — critical for emoji and Unicode in demos
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# NumPy is our ONLY external dependency for this entire day.
# In production, libraries like sentence-transformers produce embeddings,
# but the math underneath is always NumPy (or a GPU-accelerated equivalent).
import numpy as np


# ============================================================================
# SECTION 1: NumPy Fundamentals — Arrays, Shapes, Dtypes, Broadcasting
# ============================================================================
# Why: NumPy arrays are the universal data structure for numerical AI/ML.
# Every embedding, every weight matrix, every gradient in deep learning
# is a NumPy array (or its GPU cousin, a PyTorch/TensorFlow tensor).
# You MUST be fluent in array creation, indexing, reshaping, and broadcasting
# before you can reason about embeddings and similarity.

def demo_array_creation():
    """
    Demonstrate the many ways to create NumPy arrays.

    Why it matters:
      - np.array(): Convert Python lists to optimized C arrays (100x faster).
      - np.zeros/ones(): Initialize weight matrices or placeholder embeddings.
      - np.random.randn(): Simulate embeddings — real embeddings look like
        normally distributed random vectors with structure.
      - np.linspace/arange(): Create evenly spaced sequences for plotting,
        temperature schedules, learning rate decays.
    """
    print("=" * 78)
    print("📐 SECTION 1: NumPy Fundamentals")
    print("=" * 78)

    # --- 1a. From Python lists --- 
    # This is how you'd convert a raw embedding API response to a NumPy array.
    # OpenAI's embedding API returns a list of floats; we wrap it immediately.
    raw_embedding = [0.023, -0.456, 0.789, 0.012, -0.345]
    embedding = np.array(raw_embedding, dtype=np.float32)
    print(f"\n🔹 From Python list → NumPy array:")
    print(f"   Raw list type : {type(raw_embedding)}")
    print(f"   NumPy type    : {type(embedding)}")
    print(f"   Shape         : {embedding.shape}  (1D vector, 5 dimensions)")
    print(f"   Dtype         : {embedding.dtype}   (32-bit float — standard for embeddings)")
    print(f"   Values        : {embedding}")

    # --- 1b. Common initializers ---
    # Zeros: used to initialize bias vectors or empty embedding slots
    zero_vec = np.zeros(768, dtype=np.float32)
    print(f"\n🔹 np.zeros(768): Placeholder embedding (768-dim like BERT)")
    print(f"   Shape: {zero_vec.shape}, Sum: {zero_vec.sum()}")

    # Ones: useful for mask vectors (1 = keep, 0 = ignore in attention)
    attention_mask = np.ones(10, dtype=np.int32)
    attention_mask[7:] = 0  # Last 3 tokens are padding → mask them out
    print(f"\n🔹 Attention mask (1=real token, 0=padding):")
    print(f"   {attention_mask}")

    # Identity matrix: used in linear algebra, residual connections
    identity = np.eye(4, dtype=np.float32)
    print(f"\n🔹 np.eye(4): Identity matrix (residual connections add I·x)")
    print(f"   Shape: {identity.shape}")
    print(f"   {identity}")

    # --- 1c. Random arrays — simulating embeddings ---
    # Real embeddings from models like text-embedding-3-small are ~1536-dim
    # vectors that LOOK random but encode semantic meaning.
    # np.random.randn gives us standard normal distribution (mean=0, std=1).
    rng = np.random.default_rng(seed=42)  # Reproducible randomness
    fake_embedding = rng.standard_normal(8).astype(np.float32)
    print(f"\n🔹 Simulated embedding (8-dim, normal distribution):")
    print(f"   {fake_embedding}")
    print(f"   Mean: {fake_embedding.mean():.4f}, Std: {fake_embedding.std():.4f}")

    return embedding


def demo_shapes_and_dtypes():
    """
    Shapes and dtypes — the two things you check FIRST when debugging.

    Why it matters:
      - Shape mismatches cause 90% of ML bugs. If your query embedding is
        (1, 768) but your document embeddings are (N, 1536), your dot product
        will fail or silently give wrong results.
      - Dtype matters for memory: float64 uses 2x the RAM of float32.
        Agents processing thousands of embeddings must be memory-efficient.
    """
    print(f"\n{'─' * 78}")
    print(f"📏 Shapes, Dtypes, and Reshaping")
    print(f"{'─' * 78}")

    # 1D vector — a single embedding
    vec_1d = np.array([1.0, 2.0, 3.0])
    print(f"\n🔹 1D vector: shape={vec_1d.shape}, ndim={vec_1d.ndim}")

    # 2D matrix — a BATCH of embeddings (common in vector databases)
    # Shape (3, 4) means: 3 documents, each with a 4-dimensional embedding
    batch = np.array([
        [0.1, 0.2, 0.3, 0.4],   # Document 1 embedding
        [0.5, 0.6, 0.7, 0.8],   # Document 2 embedding
        [0.9, 1.0, 1.1, 1.2],   # Document 3 embedding
    ])
    print(f"🔹 2D batch: shape={batch.shape} → {batch.shape[0]} docs × {batch.shape[1]}-dim embeddings")

    # 3D tensor — batched attention matrices
    # Shape (2, 3, 3) means: 2 attention heads, each with a 3×3 attention matrix
    attention_scores = np.random.randn(2, 3, 3).astype(np.float32)
    print(f"🔹 3D tensor: shape={attention_scores.shape} → {attention_scores.shape[0]} heads × {attention_scores.shape[1]}×{attention_scores.shape[2]} attention")

    # --- Reshaping ---
    # Reshape is CRITICAL: you constantly reshape between (N,D) and (N,1,D)
    # for broadcasting in batch operations.
    flat = np.arange(12)
    matrix = flat.reshape(3, 4)  # 12 elements → 3 rows × 4 cols
    print(f"\n🔹 Reshape: {flat.shape} → {matrix.shape}")
    print(f"   {matrix}")

    # Add a dimension — needed for broadcasting query against doc batch
    query = np.array([1.0, 2.0, 3.0, 4.0])
    query_2d = query[np.newaxis, :]  # (4,) → (1, 4) — ready for batch dot product
    print(f"\n🔹 Add dimension: {query.shape} → {query_2d.shape}")
    print(f"   This lets us compute query·doc for ALL docs at once via broadcasting")

    # --- Dtype comparison ---
    f64 = np.random.randn(1000).astype(np.float64)
    f32 = f64.astype(np.float32)
    f16 = f64.astype(np.float16)
    print(f"\n🔹 Memory comparison for 1000-dim embedding:")
    print(f"   float64: {f64.nbytes:>6} bytes  (research/training)")
    print(f"   float32: {f32.nbytes:>6} bytes  (standard inference)")
    print(f"   float16: {f16.nbytes:>6} bytes  (quantized/edge deployment)")


def demo_broadcasting_and_vectorization():
    """
    Broadcasting + vectorized operations — why NumPy is 100x faster than loops.

    Why it matters:
      - An agent doing semantic search must compute similarity between a query
        and THOUSANDS of document embeddings. A Python for-loop would take
        seconds. NumPy broadcasting does it in milliseconds.
      - Broadcasting lets arrays of different shapes work together without
        explicit loops — it's the key to writing efficient ML code.
    """
    print(f"\n{'─' * 78}")
    print(f"⚡ Broadcasting & Vectorized Operations")
    print(f"{'─' * 78}")

    # --- Vectorized arithmetic ---
    # Adding a scalar to every element — no loop needed!
    # This is like applying a bias to every embedding dimension.
    embedding = np.array([0.1, -0.3, 0.5, -0.7, 0.9])
    bias = 0.05
    biased = embedding + bias  # Broadcasts scalar to every element
    print(f"\n🔹 Vectorized addition (applying bias):")
    print(f"   Original : {embedding}")
    print(f"   + bias   : {biased}")

    # --- Broadcasting: query vs document batch ---
    # Query: shape (4,)   — one query embedding
    # Docs:  shape (3, 4) — three document embeddings
    # NumPy automatically broadcasts the query across all docs!
    query = np.array([1.0, 0.0, 1.0, 0.0])
    docs = np.array([
        [1.0, 0.0, 1.0, 0.0],   # Identical to query
        [0.0, 1.0, 0.0, 1.0],   # Orthogonal to query
        [0.5, 0.5, 0.5, 0.5],   # Partially similar
    ])

    # Element-wise difference — broadcasts (4,) across (3, 4)
    diff = docs - query  # Shape: (3, 4)
    print(f"\n🔹 Broadcasting: query (4,) - docs (3,4) → diff (3,4)")
    print(f"   Differences from query:")
    for i, d in enumerate(diff):
        print(f"   Doc {i}: {d}")

    # --- Speed comparison: loop vs vectorized ---
    n = 100_000
    a = np.random.randn(n)
    b = np.random.randn(n)

    # Slow: Python loop
    start = time.perf_counter()
    loop_result = sum(a[i] * b[i] for i in range(n))
    loop_time = time.perf_counter() - start

    # Fast: NumPy vectorized
    start = time.perf_counter()
    vec_result = np.dot(a, b)
    vec_time = time.perf_counter() - start

    speedup = loop_time / vec_time if vec_time > 0 else float('inf')
    print(f"\n🔹 Speed: dot product of {n:,}-dim vectors")
    print(f"   Python loop : {loop_time*1000:.2f} ms")
    print(f"   NumPy dot   : {vec_time*1000:.4f} ms")
    print(f"   Speedup     : {speedup:.0f}x faster!")
    print(f"   Results match: {np.isclose(loop_result, vec_result)}")


# ============================================================================
# SECTION 2: Vector Operations — The Building Blocks of Embeddings
# ============================================================================
# Why: Every operation an AI agent performs on embeddings reduces to these
# fundamental vector operations. Dot products measure similarity. Magnitude
# tells you how "strong" an embedding is. Normalization ensures fair
# comparison. Distance metrics tell you how far apart two concepts are.

def demo_dot_product():
    """
    The dot product — the most important operation in all of AI.

    Why it matters:
      - Attention scores in transformers: score = Q · K
      - Cosine similarity = dot(a, b) / (|a| · |b|)
      - Neural network forward pass: output = dot(input, weights) + bias
      - The dot product measures how much two vectors "agree" — how much
        they point in the same direction.
    """
    print("\n" + "=" * 78)
    print("📐 SECTION 2: Vector Operations")
    print("=" * 78)

    print(f"\n{'─' * 78}")
    print(f"🔹 Dot Product — The Heart of AI")
    print(f"{'─' * 78}")

    # Two vectors that "agree" (point same direction) → large positive dot product
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    print(f"\n   a = {a}")
    print(f"   b = {b}")
    print(f"   a · b = {dot}  (large positive → vectors agree/similar)")

    # Orthogonal vectors (perpendicular) → dot product = 0
    # This means they share NO information — completely independent concepts.
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 0.0])
    print(f"\n   c = {c}  (pure x-axis)")
    print(f"   d = {d}  (pure y-axis)")
    print(f"   c · d = {np.dot(c, d)}    (orthogonal → zero similarity)")

    # Opposing vectors → large negative dot product
    e = np.array([1.0, 1.0, 1.0])
    f = np.array([-1.0, -1.0, -1.0])
    print(f"\n   e = {e}")
    print(f"   f = {f}")
    print(f"   e · f = {np.dot(e, f)}  (opposite direction → anti-correlated)")

    # Practical: computing attention score between a query and a key
    query = np.array([0.8, -0.2, 0.5, 0.1])     # "What is machine learning?"
    key_1 = np.array([0.7, -0.1, 0.6, 0.2])      # "AI training methods"
    key_2 = np.array([-0.5, 0.8, -0.3, -0.1])    # "Cooking recipes"

    score_1 = np.dot(query, key_1)
    score_2 = np.dot(query, key_2)
    print(f"\n   Attention-like scoring:")
    print(f"   Query · Key('AI training')   = {score_1:.4f}  (high relevance)")
    print(f"   Query · Key('Cooking')       = {score_2:.4f}  (low relevance)")


def demo_magnitude_and_normalization():
    """
    Magnitude (L2 norm) and normalization — making vectors comparable.

    Why it matters:
      - Raw embeddings have varying magnitudes. A long document might produce
        a "larger" embedding than a short one, but that doesn't mean it's
        more relevant.
      - Normalization (dividing by magnitude) strips away length, keeping
        only DIRECTION — which encodes semantic meaning.
      - After normalization, dot product EQUALS cosine similarity.
        This is why most vector databases normalize embeddings on ingestion.
    """
    print(f"\n{'─' * 78}")
    print(f"🔹 Magnitude & Normalization")
    print(f"{'─' * 78}")

    # Magnitude = sqrt(sum of squares) = L2 norm
    v = np.array([3.0, 4.0])
    magnitude = np.linalg.norm(v)  # sqrt(9 + 16) = sqrt(25) = 5.0
    print(f"\n   v = {v}")
    print(f"   |v| = √(3² + 4²) = √{3**2 + 4**2} = {magnitude}")

    # Normalization: divide by magnitude → unit vector (magnitude = 1)
    v_normalized = v / magnitude
    print(f"\n   v̂ (normalized) = {v_normalized}")
    print(f"   |v̂| = {np.linalg.norm(v_normalized):.6f}  (≈ 1.0, as expected)")

    # Why normalization matters for embeddings:
    # Without normalization, a verbose document could dominate similarity scores.
    short_doc_emb = np.array([0.2, 0.3, 0.1])   # Short doc → small values
    long_doc_emb = np.array([2.0, 3.0, 1.0])     # Long doc → large values (10x)
    query_emb = np.array([0.5, 0.5, 0.5])

    # Raw dot products are misleading — the long doc "wins" just by being bigger
    raw_sim_short = np.dot(query_emb, short_doc_emb)
    raw_sim_long = np.dot(query_emb, long_doc_emb)
    print(f"\n   ⚠️  Raw dot product (unfair):")
    print(f"   Query · ShortDoc = {raw_sim_short:.4f}")
    print(f"   Query · LongDoc  = {raw_sim_long:.4f}  ← Unfairly higher!")

    # After normalization, only direction matters → fair comparison
    short_norm = short_doc_emb / np.linalg.norm(short_doc_emb)
    long_norm = long_doc_emb / np.linalg.norm(long_doc_emb)
    query_norm = query_emb / np.linalg.norm(query_emb)

    norm_sim_short = np.dot(query_norm, short_norm)
    norm_sim_long = np.dot(query_norm, long_norm)
    print(f"\n   ✅ Normalized dot product (fair):")
    print(f"   Query · ShortDoc = {norm_sim_short:.4f}")
    print(f"   Query · LongDoc  = {norm_sim_long:.4f}  ← Same! Direction is identical.")


def demo_distance_metrics():
    """
    Distance metrics — different ways to measure "how far apart" two embeddings are.

    Why it matters:
      - Different vector databases use different distance metrics.
      - Cosine distance = 1 - cosine_similarity (used by Pinecone, Weaviate)
      - Euclidean distance = L2 norm of the difference (used by FAISS default)
      - Manhattan distance = sum of absolute differences (robust to outliers)
      - Choosing the wrong metric can degrade your agent's retrieval quality.
    """
    print(f"\n{'─' * 78}")
    print(f"🔹 Distance Metrics for Embeddings")
    print(f"{'─' * 78}")

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.5, 2.5, 2.5, 4.5])

    # Euclidean distance (L2) — straight-line distance in high-dimensional space
    euclidean = np.linalg.norm(a - b)
    print(f"\n   a = {a}")
    print(f"   b = {b}")
    print(f"\n   📏 Euclidean (L2) distance : {euclidean:.4f}")
    print(f"      Formula: √Σ(aᵢ - bᵢ)²")

    # Manhattan distance (L1) — "city block" distance
    manhattan = np.sum(np.abs(a - b))
    print(f"   📏 Manhattan (L1) distance: {manhattan:.4f}")
    print(f"      Formula: Σ|aᵢ - bᵢ|")

    # Cosine distance = 1 - cosine_similarity
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_dist = 1.0 - cos_sim
    print(f"   📏 Cosine distance        : {cos_dist:.4f}")
    print(f"      Formula: 1 - (a·b)/(|a|·|b|)")

    # Chebyshev distance (L∞) — maximum difference in any dimension
    chebyshev = np.max(np.abs(a - b))
    print(f"   📏 Chebyshev (L∞) distance: {chebyshev:.4f}")
    print(f"      Formula: max|aᵢ - bᵢ|")

    # Comparison table
    print(f"\n   📊 When to use which metric:")
    print(f"   ┌────────────────┬─────────────────────────────────────────────┐")
    print(f"   │ Metric         │ Best for                                    │")
    print(f"   ├────────────────┼─────────────────────────────────────────────┤")
    print(f"   │ Cosine         │ Text embeddings, semantic search (RAG)      │")
    print(f"   │ Euclidean (L2) │ Image embeddings, clustering                │")
    print(f"   │ Manhattan (L1) │ Sparse vectors, robust to outlier dims      │")
    print(f"   │ Dot Product    │ Pre-normalized embeddings (fastest)          │")
    print(f"   └────────────────┴─────────────────────────────────────────────┘")


# ============================================================================
# SECTION 3: Cosine Similarity — THE Metric for Semantic Search
# ============================================================================
# Why: When your RAG agent retrieves documents, it ranks them by cosine
# similarity. When your agent decides which tool to use, it might compare the
# user's intent embedding against tool description embeddings using cosine
# similarity. This is the single most important metric in agentic AI.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(θ) = (a · b) / (|a| × |b|)

    Returns a value in [-1, 1]:
      +1 = identical direction (semantically identical)
       0 = orthogonal (completely unrelated)
      -1 = opposite direction (semantically opposite)

    Args:
        a: First vector (any dimension).
        b: Second vector (same dimension as a).

    Returns:
        Cosine similarity as a float.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Guard against zero vectors (a zero embedding means "no information")
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query and many documents — vectorized.

    This is what vector databases do internally: compute similarity of your
    query against thousands/millions of stored embeddings in one operation.

    Args:
        query: Shape (D,) — single query embedding.
        documents: Shape (N, D) — N document embeddings, each D-dimensional.

    Returns:
        Shape (N,) — cosine similarity of query with each document.
    """
    # Dot product of query with every doc: (N, D) @ (D,) → (N,)
    dot_products = documents @ query

    # Norms: query norm is a scalar, doc norms are (N,)
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(documents, axis=1)

    # Guard against zero norms
    denominator = query_norm * doc_norms
    denominator = np.where(denominator == 0.0, 1.0, denominator)

    return dot_products / denominator


def demo_cosine_similarity():
    """
    Demonstrate cosine similarity with intuitive examples.
    """
    print("\n" + "=" * 78)
    print("📐 SECTION 3: Cosine Similarity — The RAG Agent's Best Friend")
    print("=" * 78)

    # --- Identical vectors → similarity = 1.0 ---
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    sim = cosine_similarity(a, b)
    print(f"\n🔹 Identical vectors:")
    print(f"   a = {a},  b = {b}")
    print(f"   Cosine similarity = {sim:.4f}  (1.0 = perfect match)")

    # --- Scaled vectors → still 1.0! Cosine ignores magnitude ---
    c = np.array([2.0, 4.0, 6.0])  # Same direction as 'a', just 2x longer
    sim_scaled = cosine_similarity(a, c)
    print(f"\n🔹 Scaled vector (same direction, 2x magnitude):")
    print(f"   a = {a},  c = {c}")
    print(f"   Cosine similarity = {sim_scaled:.4f}  (still 1.0 — direction matters, not size)")

    # --- Orthogonal vectors → similarity = 0.0 ---
    d = np.array([1.0, 0.0, 0.0])
    e = np.array([0.0, 1.0, 0.0])
    sim_ortho = cosine_similarity(d, e)
    print(f"\n🔹 Orthogonal vectors:")
    print(f"   d = {d},  e = {e}")
    print(f"   Cosine similarity = {sim_ortho:.4f}  (0.0 = completely unrelated)")

    # --- Opposite vectors → similarity = -1.0 ---
    f = np.array([1.0, 1.0, 1.0])
    g = np.array([-1.0, -1.0, -1.0])
    sim_opp = cosine_similarity(f, g)
    print(f"\n🔹 Opposite vectors:")
    print(f"   f = {f},  g = {g}")
    print(f"   Cosine similarity = {sim_opp:.4f}  (-1.0 = semantically opposite)")

    # --- Real-world analogy: word similarity ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Simulating Semantic Similarity")
    print(f"{'─' * 78}")
    print(f"   (Pretend these are real word embeddings)")

    # Simulated 4-dim embeddings encoding: [royalty, gender, age, power]
    king   = np.array([0.9, 0.1, 0.5, 0.8])
    queen  = np.array([0.9, 0.9, 0.5, 0.8])
    man    = np.array([0.1, 0.1, 0.5, 0.2])
    woman  = np.array([0.1, 0.9, 0.5, 0.2])
    child  = np.array([0.0, 0.5, 0.1, 0.0])

    pairs = [
        ("king", king, "queen", queen),
        ("king", king, "man", man),
        ("queen", queen, "woman", woman),
        ("king", king, "child", child),
        ("man", man, "woman", woman),
    ]

    print(f"\n   {'Pair':<20} {'Cosine Similarity':>18}")
    print(f"   {'─' * 20} {'─' * 18}")
    for name_a, vec_a, name_b, vec_b in pairs:
        s = cosine_similarity(vec_a, vec_b)
        bar = "█" * int(abs(s) * 20)
        print(f"   {name_a:>8} ↔ {name_b:<8} {s:>8.4f}  {bar}")

    # --- Vector arithmetic: king - man + woman ≈ queen ---
    result = king - man + woman
    print(f"\n   🧮 Vector Arithmetic: king - man + woman = ?")
    print(f"      Result vector     : {result}")
    print(f"      Expected (queen)  : {queen}")
    print(f"      Similarity to queen: {cosine_similarity(result, queen):.4f}")
    print(f"      Similarity to king : {cosine_similarity(result, king):.4f}")
    print(f"      → The result is CLOSEST to 'queen'! Vector math captures analogies.")

    # --- Batch cosine similarity ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Batch Cosine Similarity (Vectorized Search)")
    print(f"{'─' * 78}")

    query_vec = king
    doc_vecs = np.array([king, queen, man, woman, child])
    doc_labels = ["king", "queen", "man", "woman", "child"]

    similarities = batch_cosine_similarity(query_vec, doc_vecs)
    ranked = np.argsort(similarities)[::-1]  # Sort descending

    print(f"\n   Query: 'king'")
    print(f"   {'Rank':<6} {'Word':<10} {'Similarity':>10}")
    print(f"   {'─' * 6} {'─' * 10} {'─' * 10}")
    for rank, idx in enumerate(ranked, 1):
        print(f"   {rank:<6} {doc_labels[idx]:<10} {similarities[idx]:>10.4f}")


# ============================================================================
# SECTION 4: Matrix Operations — How Attention Mechanisms Work
# ============================================================================
# Why: The self-attention mechanism in Transformers is:
#   Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
# That's TWO matrix multiplications, a transpose, and a softmax.
# Understanding these operations demystifies how LLMs "pay attention" to
# relevant parts of the input — the core mechanism behind GPT, Claude, etc.

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax — converts raw scores to probabilities.

    Why it matters:
      - After computing Q·Kᵀ attention scores, softmax turns them into
        probability weights that sum to 1.0. This means the model "chooses"
        which keys to pay attention to.
      - We subtract max(x) for numerical stability: exp(large_number) = inf,
        but exp(0) = 1. This trick prevents overflow.

    Args:
        x: Input array (any shape, softmax applied along last axis).

    Returns:
        Probability distribution (same shape, values sum to 1 along last axis).
    """
    # Subtract max for numerical stability (doesn't change the result)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def demo_matrix_operations():
    """
    Matrix operations — the mechanics of attention and neural networks.
    """
    print("\n" + "=" * 78)
    print("📐 SECTION 4: Matrix Operations — How Attention Works")
    print("=" * 78)

    # --- Matrix multiplication ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Matrix Multiplication — The Core of Neural Networks")
    print(f"{'─' * 78}")

    # In a neural network: output = input @ weights
    # Input: batch of 2 samples, each 3 features
    input_data = np.array([
        [1.0, 2.0, 3.0],  # Sample 1
        [4.0, 5.0, 6.0],  # Sample 2
    ])
    # Weights: 3 input features → 2 output features (a layer that compresses)
    weights = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6],
    ])
    output = input_data @ weights  # (2, 3) @ (3, 2) → (2, 2)
    print(f"\n   Input (2×3) @ Weights (3×2) = Output (2×2)")
    print(f"   Input:\n{_indent(input_data)}")
    print(f"   Weights:\n{_indent(weights)}")
    print(f"   Output:\n{_indent(output)}")
    print(f"   Each output value = weighted sum of inputs (a dot product)")

    # --- Transpose ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Transpose — Used in Q·Kᵀ Attention Computation")
    print(f"{'─' * 78}")

    K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    K_T = K.T  # (3, 2) → (2, 3)
    print(f"\n   K (Keys matrix, 3 tokens × 2 dims):\n{_indent(K)}")
    print(f"   Kᵀ (Transposed, 2 dims × 3 tokens):\n{_indent(K_T)}")
    print(f"   Shape: {K.shape} → {K_T.shape}")

    # --- Full Attention Mechanism ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Self-Attention: Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V")
    print(f"{'─' * 78}")

    # Simulating 3 tokens with 4-dimensional Q, K, V embeddings
    np.random.seed(42)
    seq_len, d_k = 3, 4  # 3 tokens, each with 4-dim key/query/value

    Q = np.random.randn(seq_len, d_k).astype(np.float32)  # Queries
    K = np.random.randn(seq_len, d_k).astype(np.float32)  # Keys
    V = np.random.randn(seq_len, d_k).astype(np.float32)  # Values

    print(f"\n   Sequence length: {seq_len} tokens, d_k: {d_k} dimensions")
    print(f"   Q (queries):\n{_indent(Q)}")
    print(f"   K (keys):\n{_indent(K)}")
    print(f"   V (values):\n{_indent(V)}")

    # Step 1: Q · Kᵀ → raw attention scores (seq_len × seq_len)
    scores = Q @ K.T
    print(f"\n   Step 1: Q · Kᵀ (raw attention scores, {scores.shape}):")
    print(f"{_indent(scores)}")
    print(f"   → Each row shows how much each query attends to each key")

    # Step 2: Scale by √d_k to prevent gradients from vanishing
    scale = np.sqrt(d_k)
    scaled_scores = scores / scale
    print(f"\n   Step 2: Scale by √d_k = √{d_k} = {scale:.2f}")
    print(f"{_indent(scaled_scores)}")

    # Step 3: Softmax → attention weights (probabilities that sum to 1)
    attention_weights = softmax(scaled_scores)
    print(f"\n   Step 3: Softmax → attention weights (each row sums to 1.0):")
    print(f"{_indent(attention_weights)}")
    for i in range(seq_len):
        print(f"   Row {i} sum: {attention_weights[i].sum():.6f}")

    # Step 4: Multiply by V → weighted combination of values
    attention_output = attention_weights @ V
    print(f"\n   Step 4: Attention weights @ V → output:")
    print(f"{_indent(attention_output)}")
    print(f"\n   ✅ Each output token is now a weighted blend of all value vectors,")
    print(f"      with weights determined by query-key similarity.")
    print(f"      THIS is how transformers 'understand context'!")

    # --- Inverse matrix (bonus) ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Matrix Inverse — Solving Linear Systems")
    print(f"{'─' * 78}")

    A = np.array([[2.0, 1.0], [5.0, 3.0]])
    A_inv = np.linalg.inv(A)
    identity_check = A @ A_inv

    print(f"\n   A:\n{_indent(A)}")
    print(f"   A⁻¹:\n{_indent(A_inv)}")
    print(f"   A @ A⁻¹ (should be Identity):\n{_indent(np.round(identity_check, 6))}")

    # Determinant
    det = np.linalg.det(A)
    print(f"\n   det(A) = {det:.4f}")
    print(f"   (Non-zero determinant → matrix is invertible)")


def _indent(arr: np.ndarray, spaces: int = 6) -> str:
    """Helper to indent array printing for cleaner output."""
    lines = str(arr).split('\n')
    return '\n'.join(' ' * spaces + line for line in lines)


# ============================================================================
# SECTION 5: Embedding Simulation — How Words Become Vectors
# ============================================================================
# Why: Before you use a real embedding model (OpenAI, Sentence-Transformers,
# etc.), you need to understand what an embedding IS. This section simulates
# how words and sentences get mapped to vectors, and shows that semantic
# relationships are preserved in vector space.

class SimpleEmbeddingModel:
    """
    A rule-based embedding model that maps words to vectors.

    In reality, models like text-embedding-3-small learn these mappings from
    billions of text examples. Our simplified version manually encodes
    semantic features to demonstrate the concept.

    Each dimension represents a semantic feature:
      dim 0: technology/science
      dim 1: nature/biology
      dim 2: food/cooking
      dim 3: emotion/sentiment (positive)
      dim 4: emotion/sentiment (negative)
      dim 5: action/verb-ness
      dim 6: size/scale
      dim 7: abstract/concrete
    """

    EMBEDDING_DIM = 8

    # Hand-crafted word embeddings — each encodes semantic features.
    # Real embeddings have 768-3072 dimensions and are learned, not designed.
    VOCABULARY = {
        # Technology cluster
        "python":       np.array([0.9, 0.1, 0.0, 0.3, 0.0, 0.6, 0.3, 0.7]),
        "code":         np.array([0.9, 0.0, 0.0, 0.2, 0.0, 0.7, 0.3, 0.8]),
        "computer":     np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.3, 0.5, 0.5]),
        "algorithm":    np.array([0.8, 0.0, 0.0, 0.1, 0.0, 0.5, 0.4, 0.9]),
        "software":     np.array([0.9, 0.0, 0.0, 0.2, 0.0, 0.4, 0.5, 0.8]),
        "ai":           np.array([0.9, 0.1, 0.0, 0.4, 0.1, 0.5, 0.6, 0.9]),
        "neural":       np.array([0.7, 0.3, 0.0, 0.2, 0.0, 0.4, 0.4, 0.9]),
        "data":         np.array([0.8, 0.1, 0.0, 0.1, 0.0, 0.3, 0.6, 0.6]),

        # Nature cluster
        "tree":         np.array([0.0, 0.9, 0.1, 0.3, 0.0, 0.1, 0.6, 0.1]),
        "forest":       np.array([0.0, 0.9, 0.0, 0.4, 0.0, 0.0, 0.8, 0.1]),
        "flower":       np.array([0.0, 0.9, 0.1, 0.6, 0.0, 0.1, 0.2, 0.1]),
        "river":        np.array([0.0, 0.8, 0.0, 0.4, 0.0, 0.3, 0.7, 0.1]),
        "mountain":     np.array([0.0, 0.7, 0.0, 0.5, 0.0, 0.0, 0.9, 0.1]),
        "ocean":        np.array([0.0, 0.8, 0.0, 0.5, 0.1, 0.2, 0.9, 0.1]),
        "animal":       np.array([0.0, 0.9, 0.1, 0.3, 0.1, 0.4, 0.5, 0.1]),

        # Food cluster
        "pizza":        np.array([0.0, 0.1, 0.9, 0.6, 0.0, 0.2, 0.3, 0.1]),
        "cook":         np.array([0.0, 0.0, 0.9, 0.3, 0.0, 0.8, 0.2, 0.3]),
        "recipe":       np.array([0.0, 0.0, 0.9, 0.3, 0.0, 0.5, 0.3, 0.5]),
        "restaurant":   np.array([0.0, 0.0, 0.8, 0.5, 0.0, 0.2, 0.5, 0.2]),
        "delicious":    np.array([0.0, 0.0, 0.7, 0.8, 0.0, 0.1, 0.2, 0.3]),
        "kitchen":      np.array([0.0, 0.0, 0.9, 0.2, 0.0, 0.5, 0.4, 0.2]),

        # Emotion words
        "happy":        np.array([0.0, 0.1, 0.1, 0.9, 0.0, 0.3, 0.2, 0.4]),
        "sad":          np.array([0.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.2, 0.4]),
        "love":         np.array([0.0, 0.2, 0.1, 0.9, 0.0, 0.4, 0.3, 0.5]),
        "fear":         np.array([0.0, 0.1, 0.0, 0.0, 0.9, 0.3, 0.2, 0.5]),
        "peace":        np.array([0.0, 0.3, 0.0, 0.8, 0.0, 0.0, 0.3, 0.6]),

        # Action words
        "run":          np.array([0.1, 0.2, 0.0, 0.3, 0.1, 0.9, 0.3, 0.2]),
        "build":        np.array([0.4, 0.0, 0.0, 0.4, 0.0, 0.9, 0.5, 0.4]),
        "learn":        np.array([0.5, 0.1, 0.0, 0.5, 0.0, 0.7, 0.3, 0.7]),
        "create":       np.array([0.3, 0.2, 0.1, 0.6, 0.0, 0.8, 0.4, 0.5]),
    }

    def __init__(self):
        """Initialize with normalized vocabulary embeddings."""
        self._vocab = {}
        for word, vec in self.VOCABULARY.items():
            # Normalize all embeddings to unit length — industry standard
            norm = np.linalg.norm(vec)
            self._vocab[word] = vec / norm if norm > 0 else vec

    def embed_word(self, word: str) -> np.ndarray:
        """
        Get the embedding for a single word.

        If the word is unknown, return a random but deterministic vector
        (simulating subword tokenization fallback in real models).
        """
        word_lower = word.lower().strip()
        if word_lower in self._vocab:
            return self._vocab[word_lower].copy()

        # Unknown word → deterministic random embedding based on hash
        # Real models use subword tokenization (BPE) to handle unknown words
        seed = hash(word_lower) % (2**31)
        rng = np.random.default_rng(seed)
        random_emb = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float64)
        return random_emb / np.linalg.norm(random_emb)

    def embed_sentence(self, sentence: str) -> np.ndarray:
        """
        Embed a sentence by averaging word embeddings.

        This is the simplest sentence embedding technique — mean pooling.
        Real models (Sentence-BERT, text-embedding-3-small) use more
        sophisticated pooling over transformer hidden states, but averaging
        is a surprisingly strong baseline.
        """
        words = sentence.lower().strip().split()
        if not words:
            return np.zeros(self.EMBEDDING_DIM)

        word_embeddings = np.array([self.embed_word(w) for w in words])

        # Mean pooling: average all word vectors into one sentence vector
        sentence_embedding = word_embeddings.mean(axis=0)

        # Re-normalize the averaged vector to unit length
        norm = np.linalg.norm(sentence_embedding)
        if norm > 0:
            sentence_embedding = sentence_embedding / norm

        return sentence_embedding

    def get_vocabulary(self) -> list:
        """Return list of known words."""
        return list(self._vocab.keys())


def demo_embedding_simulation():
    """
    Demonstrate how embeddings capture semantic relationships.
    """
    print("\n" + "=" * 78)
    print("📐 SECTION 5: Embedding Simulation — Words as Vectors")
    print("=" * 78)

    model = SimpleEmbeddingModel()

    # --- Show word embeddings ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Word Embeddings (8-dimensional feature vectors)")
    print(f"{'─' * 78}")
    print(f"\n   Dimensions: [tech, nature, food, positive, negative, action, size, abstract]")

    sample_words = ["python", "tree", "pizza", "happy", "learn"]
    for word in sample_words:
        emb = model.embed_word(word)
        formatted = ", ".join(f"{v:>5.2f}" for v in emb)
        print(f"   {word:>10}: [{formatted}]")

    # --- Semantic clusters ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Semantic Clustering — Similar Words Group Together")
    print(f"{'─' * 78}")

    clusters = {
        "🖥️ Technology": ["python", "code", "computer", "algorithm", "ai"],
        "🌳 Nature":     ["tree", "forest", "flower", "river", "mountain"],
        "🍕 Food":       ["pizza", "cook", "recipe", "restaurant", "delicious"],
        "😊 Emotions":   ["happy", "sad", "love", "fear", "peace"],
    }

    for cluster_name, words in clusters.items():
        embeddings = [model.embed_word(w) for w in words]
        # Compute average intra-cluster similarity
        sims = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                sims.append(cosine_similarity(embeddings[i], embeddings[j]))
        avg_sim = np.mean(sims)
        print(f"\n   {cluster_name}")
        print(f"   Words: {', '.join(words)}")
        print(f"   Avg intra-cluster similarity: {avg_sim:.4f}")

    # Cross-cluster similarity (should be low)
    tech_emb = model.embed_word("python")
    food_emb = model.embed_word("pizza")
    nature_emb = model.embed_word("tree")
    print(f"\n   Cross-cluster similarities (should be low):")
    print(f"   python ↔ pizza : {cosine_similarity(tech_emb, food_emb):.4f}")
    print(f"   python ↔ tree  : {cosine_similarity(tech_emb, nature_emb):.4f}")
    print(f"   pizza  ↔ tree  : {cosine_similarity(food_emb, nature_emb):.4f}")

    # --- Sentence embeddings ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Sentence Embeddings (Mean Pooling)")
    print(f"{'─' * 78}")

    sentences = [
        "python code algorithm",
        "learn python ai",
        "cook delicious pizza",
        "tree forest mountain",
        "happy love peace",
        "sad fear",
    ]

    print(f"\n   Computing sentence embeddings via mean pooling:")
    sentence_embeddings = []
    for s in sentences:
        emb = model.embed_sentence(s)
        sentence_embeddings.append(emb)
        formatted = ", ".join(f"{v:>5.2f}" for v in emb[:4])
        print(f"   '{s}'")
        print(f"     → [{formatted}, ...]  (showing first 4 of 8 dims)")

    # Compare sentence similarities
    print(f"\n   Sentence similarity matrix:")
    print(f"   {'':>30}", end="")
    for i in range(len(sentences)):
        print(f"  S{i}", end="")
    print()

    for i, s_i in enumerate(sentences):
        label = s_i[:28] + ".." if len(s_i) > 30 else s_i
        print(f"   S{i} {label:>26}  ", end="")
        for j in range(len(sentences)):
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
            if i == j:
                print(f" 1.0", end="")
            else:
                print(f" {sim:.1f}", end="")
        print()

    return model


# ============================================================================
# SECTION 6: Mini Semantic Search Engine — A Complete RAG Retriever
# ============================================================================
# Why: This is the capstone — building a functional semantic search engine
# from scratch using only NumPy. This is EXACTLY what the retrieval component
# of a RAG agent does:
#   1. Encode documents into embeddings (offline, at ingestion time)
#   2. Encode the user's query into an embedding (at query time)
#   3. Compute cosine similarity between query and all documents
#   4. Return the top-K most similar documents
# The only difference from production is that we use simulated embeddings
# instead of a real model like text-embedding-3-small.

class VectorStore:
    """
    A minimal vector database — stores embeddings and supports similarity search.

    This is a simplified version of what Pinecone, Weaviate, Chroma, and
    FAISS do under the hood. Production vector stores add:
      - Approximate Nearest Neighbor (ANN) indexes for speed
      - Metadata filtering
      - Persistence to disk
      - Distributed sharding

    But the CORE operation is always: cosine similarity → top-K retrieval.
    """

    def __init__(self, embedding_model: SimpleEmbeddingModel):
        """
        Initialize the vector store.

        Args:
            embedding_model: Model used to convert text to embeddings.
        """
        self.model = embedding_model
        self.documents: list = []           # Original text documents
        self.embeddings: np.ndarray = None  # (N, D) matrix of document embeddings
        self.metadata: list = []            # Optional metadata per document

    def add_documents(self, documents: list, metadata: list = None):
        """
        Ingest documents: convert to embeddings and store.

        This is the "indexing" phase — happens offline, before any queries.
        In production, this might process millions of documents.

        Args:
            documents: List of text strings to index.
            metadata: Optional list of metadata dicts (one per document).
        """
        print(f"\n   📥 Indexing {len(documents)} documents...")

        new_embeddings = []
        for doc in documents:
            emb = self.model.embed_sentence(doc)
            new_embeddings.append(emb)
            self.documents.append(doc)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

        # Stack all embeddings into a single matrix for batch operations
        new_matrix = np.array(new_embeddings)
        if self.embeddings is None:
            self.embeddings = new_matrix
        else:
            self.embeddings = np.vstack([self.embeddings, new_matrix])

        print(f"   ✅ Index size: {len(self.documents)} documents, "
              f"embedding matrix shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for documents most similar to the query.

        This is the "retrieval" phase — happens at query time.
        The core algorithm:
          1. Embed the query
          2. Compute cosine similarity with ALL document embeddings
          3. Sort by similarity (descending)
          4. Return top-K results

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of (document, similarity_score, metadata) tuples.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Step 1: Embed the query
        query_embedding = self.model.embed_sentence(query)

        # Step 2: Compute cosine similarity with ALL documents (vectorized!)
        similarities = batch_cosine_similarity(query_embedding, self.embeddings)

        # Step 3: Get top-K indices (argsort returns ascending, so we reverse)
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Step 4: Build results
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx],
                "rank": len(results) + 1,
            })

        return results

    def stats(self) -> dict:
        """Return statistics about the vector store."""
        if self.embeddings is None:
            return {"num_documents": 0, "embedding_dim": 0}

        return {
            "num_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1],
            "matrix_shape": self.embeddings.shape,
            "memory_bytes": self.embeddings.nbytes,
            "memory_mb": self.embeddings.nbytes / (1024 * 1024),
        }


def demo_semantic_search():
    """
    Build and query a complete semantic search engine.

    This demonstrates the full RAG retrieval pipeline:
      1. Create a knowledge base of documents
      2. Index them into a vector store
      3. Query with natural language
      4. Get ranked results by semantic similarity
    """
    print("\n" + "=" * 78)
    print("📐 SECTION 6: Mini Semantic Search Engine — RAG Retrieval in Action")
    print("=" * 78)

    model = SimpleEmbeddingModel()
    store = VectorStore(model)

    # --- Build a knowledge base ---
    # These are the "documents" an agent would retrieve from.
    # In production, these would be chunks from PDFs, web pages, databases.
    knowledge_base = [
        # Technology documents
        "python code software build algorithm",
        "ai neural data learn algorithm",
        "computer software code data",
        "python learn build code",
        "ai data computer algorithm software",

        # Nature documents
        "tree forest mountain river",
        "flower forest tree animal",
        "ocean river mountain forest",
        "tree animal forest flower",
        "mountain ocean river tree",

        # Food documents
        "cook pizza recipe kitchen",
        "restaurant delicious pizza cook",
        "recipe cook kitchen delicious",
        "pizza restaurant cook recipe",
        "kitchen cook delicious recipe",

        # Emotion documents
        "happy love peace create",
        "sad fear run",
        "happy peace love learn",
        "love happy create build",
        "peace happy love",
    ]

    categories = (
        ["tech"] * 5 + ["nature"] * 5 + ["food"] * 5 + ["emotion"] * 5
    )
    metadata_list = [{"category": cat, "doc_id": i}
                     for i, cat in enumerate(categories)]

    store.add_documents(knowledge_base, metadata_list)

    # --- Show vector store stats ---
    stats = store.stats()
    print(f"\n   📊 Vector Store Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"      {key}: {val:.4f}")
        else:
            print(f"      {key}: {val}")

    # --- Run queries ---
    queries = [
        ("python algorithm code", "Looking for programming content"),
        ("tree mountain forest", "Looking for nature content"),
        ("cook pizza delicious", "Looking for food content"),
        ("happy love peace", "Looking for emotional content"),
        ("ai learn build", "Cross-domain: tech + action"),
        ("river ocean mountain", "Nature/geography query"),
    ]

    for query_text, description in queries:
        print(f"\n{'─' * 78}")
        print(f"   🔍 Query: '{query_text}'")
        print(f"      Intent: {description}")
        print(f"{'─' * 78}")

        results = store.search(query_text, top_k=5)

        print(f"\n   {'Rank':<6} {'Score':>7} {'Category':>10}   Document")
        print(f"   {'─' * 6} {'─' * 7} {'─' * 10}   {'─' * 35}")
        for r in results:
            cat = r['metadata'].get('category', '?')
            bar = "█" * int(r['similarity'] * 30)
            print(f"   #{r['rank']:<5} {r['similarity']:>7.4f} {cat:>10}   "
                  f"{r['document'][:35]}")
            print(f"   {'':>26}   {bar}")

    # --- Demonstrate relevance of results ---
    print(f"\n{'─' * 78}")
    print(f"   🎯 Search Quality Analysis")
    print(f"{'─' * 78}")

    # Check if top results match expected categories
    test_cases = [
        ("python algorithm", "tech"),
        ("forest tree flower", "nature"),
        ("pizza recipe cook", "food"),
        ("happy love", "emotion"),
    ]

    correct = 0
    total = len(test_cases)
    for query_text, expected_cat in test_cases:
        results = store.search(query_text, top_k=3)
        top_cats = [r['metadata']['category'] for r in results]
        is_correct = top_cats[0] == expected_cat
        correct += int(is_correct)
        status = "✅" if is_correct else "❌"
        print(f"   {status} Query: '{query_text}' → "
              f"Top category: {top_cats[0]} "
              f"(expected: {expected_cat})")

    accuracy = correct / total * 100
    print(f"\n   📈 Retrieval Accuracy: {correct}/{total} = {accuracy:.0f}%")
    print(f"      (Even with simulated embeddings, semantic search works!)")

    return store


# ============================================================================
# BONUS: Practical Utilities for Agent Development
# ============================================================================

def demo_practical_utilities():
    """
    Practical NumPy utilities that come up constantly in agent development.
    """
    print("\n" + "=" * 78)
    print("🧰 BONUS: Practical NumPy Utilities for Agent Development")
    print("=" * 78)

    # --- 1. Top-K selection (for retrieval) ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Top-K Selection — Efficient Retrieval")
    print(f"{'─' * 78}")

    scores = np.array([0.3, 0.9, 0.1, 0.7, 0.5, 0.8, 0.2, 0.6])
    k = 3

    # Method 1: Full sort (O(n log n))
    top_k_sorted = np.argsort(scores)[::-1][:k]

    # Method 2: Partial sort with argpartition (O(n)) — faster for large arrays!
    top_k_partition = np.argpartition(scores, -k)[-k:]
    top_k_partition = top_k_partition[np.argsort(scores[top_k_partition])[::-1]]

    print(f"   Scores: {scores}")
    print(f"   Top-{k} indices (argsort)      : {top_k_sorted}")
    print(f"   Top-{k} indices (argpartition) : {top_k_partition}")
    print(f"   Top-{k} values: {scores[top_k_sorted]}")
    print(f"   argpartition is O(n) vs argsort O(n·log·n) — use for large vector DBs!")

    # --- 2. Temperature scaling for LLM sampling ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Temperature Scaling — Controlling LLM Creativity")
    print(f"{'─' * 78}")

    logits = np.array([2.0, 1.0, 0.5, 0.1, -0.5])
    tokens = ["the", "a", "an", "one", "some"]

    print(f"\n   Raw logits: {logits}")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        probs = softmax(logits / temp)
        print(f"\n   Temperature = {temp}:")
        for token, prob in zip(tokens, probs):
            bar = "█" * int(prob * 50)
            print(f"      {token:>5}: {prob:.4f}  {bar}")

    print(f"\n   → Low temp (0.1): deterministic, picks the highest-scoring token")
    print(f"   → High temp (2.0): creative/random, spreads probability more evenly")

    # --- 3. Embedding compression (dimensionality reduction via random projection) ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Embedding Compression — Random Projection")
    print(f"{'─' * 78}")

    original_dim = 1536
    compressed_dim = 256
    num_docs = 100

    # Simulate high-dimensional embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((num_docs, original_dim)).astype(np.float32)

    # Random projection matrix (Johnson-Lindenstrauss lemma guarantees
    # approximate distance preservation!)
    projection = rng.standard_normal(
        (original_dim, compressed_dim)
    ).astype(np.float32) / np.sqrt(compressed_dim)

    # Compress: (100, 1536) @ (1536, 256) → (100, 256)
    compressed = embeddings @ projection

    # Check that pairwise similarities are approximately preserved
    # Pick two random documents
    doc_a, doc_b = 0, 1
    sim_original = cosine_similarity(embeddings[doc_a], embeddings[doc_b])
    sim_compressed = cosine_similarity(compressed[doc_a], compressed[doc_b])

    print(f"\n   Original: {num_docs} docs × {original_dim} dims = "
          f"{embeddings.nbytes / 1024:.0f} KB")
    print(f"   Compressed: {num_docs} docs × {compressed_dim} dims = "
          f"{compressed.nbytes / 1024:.0f} KB")
    print(f"   Compression ratio: {original_dim / compressed_dim:.1f}x")
    print(f"\n   Similarity preservation (docs 0 vs 1):")
    print(f"   Original similarity  : {sim_original:.4f}")
    print(f"   Compressed similarity: {sim_compressed:.4f}")
    print(f"   → Random projection approximately preserves distances!")

    # --- 4. Batch similarity matrix ---
    print(f"\n{'─' * 78}")
    print(f"🔹 Pairwise Similarity Matrix — Finding Clusters")
    print(f"{'─' * 78}")

    model = SimpleEmbeddingModel()
    words = ["python", "code", "ai", "tree", "forest", "pizza", "cook"]
    word_embs = np.array([model.embed_word(w) for w in words])

    # Compute ALL pairwise similarities at once using matrix multiplication!
    # For normalized vectors: similarity_matrix = embeddings @ embeddings.T
    sim_matrix = word_embs @ word_embs.T

    print(f"\n   Pairwise Cosine Similarity Matrix:")
    print(f"   {'':>10}", end="")
    for w in words:
        print(f" {w[:6]:>6}", end="")
    print()
    print(f"   {'':>10}", end="")
    for _ in words:
        print(f" {'─' * 6}", end="")
    print()
    for i, w in enumerate(words):
        print(f"   {w:>10}", end="")
        for j in range(len(words)):
            val = sim_matrix[i, j]
            print(f" {val:>6.2f}", end="")
        print()
    print(f"\n   → Tech words cluster together, nature clusters, food clusters.")
    print(f"      This is how agents identify related documents!")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def demonstrate_all():
    """
    Run all demonstrations in sequence.

    This is the master function that exercises every concept from Day 15.
    Each section builds on the previous one, culminating in a working
    semantic search engine.
    """
    print("\n" + "🔸" * 39)
    print("  🤖 Day 15/150 — NumPy & Vector Mathematics")
    print("     The Math Behind Embeddings & Semantic Search")
    print("     Phase 2: AI/ML Essentials")
    print("🔸" * 39)

    # Section 1: NumPy Fundamentals
    demo_array_creation()
    demo_shapes_and_dtypes()
    demo_broadcasting_and_vectorization()

    # Section 2: Vector Operations
    demo_dot_product()
    demo_magnitude_and_normalization()
    demo_distance_metrics()

    # Section 3: Cosine Similarity
    demo_cosine_similarity()

    # Section 4: Matrix Operations & Attention
    demo_matrix_operations()

    # Section 5: Embedding Simulation
    demo_embedding_simulation()

    # Section 6: Mini Semantic Search Engine
    store = demo_semantic_search()

    # Bonus: Practical Utilities
    demo_practical_utilities()

    # --- Final Summary ---
    print("\n" + "=" * 78)
    print("🎯 Day 15 Summary — What You Learned")
    print("=" * 78)
    print("""
   ✅ Section 1: NumPy arrays, shapes, dtypes, broadcasting — the foundation
   ✅ Section 2: Dot products, magnitude, normalization, distance metrics
   ✅ Section 3: Cosine similarity — THE metric for semantic search
   ✅ Section 4: Matrix multiplication & attention mechanism walkthrough
   ✅ Section 5: Simulated word/sentence embeddings with semantic clustering
   ✅ Section 6: Complete semantic search engine (vector store + retrieval)
   ✅ Bonus: Top-K selection, temperature scaling, embedding compression

   🔗 Connection to Agentic AI:
      • RAG agents use cosine similarity to find relevant documents
      • Attention (Q·Kᵀ) is how transformers understand context
      • Embeddings ARE the universal language between agents and knowledge
      • Vector databases are just organized arrays of embeddings

   📅 Next: Day 16 — Pandas & Data Wrangling for Agents
""")
    print("=" * 78)


if __name__ == '__main__':
    demonstrate_all()

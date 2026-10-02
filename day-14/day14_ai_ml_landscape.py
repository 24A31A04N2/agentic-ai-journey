import sys
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

"""
🤖 Day 14/250 — AI/ML Landscape — Conceptual Foundations
========================================================
Phase 1: Software Engineering + AI Foundations

Concepts:
  1. AI Taxonomy — AI vs ML vs DL vs GenAI vs Agentic AI
  2. Neural Networks — Layers, weights, activations, forward pass
  3. Next-Token Prediction — What LLMs actually do under the hood
  4. Self-Attention — The core mechanism behind transformers
  5. Encoder vs Decoder — BERT-style vs GPT-style architectures
  6. Agentic AI Frontier — Why agents are the current revolution

Why this matters for Agentic AI:
  Before we can build sophisticated agents that orchestrate multi-step
  workflows, we must understand the engine powering them. Knowing how
  next-token prediction works, why self-attention scales so well, and
  how we got from rule-based AI to agentic systems gives us the mental
  models required to debug LLM hallucinations, optimize context windows,
  and structure agent reasoning effectively.
"""

# ==============================================================================
# SECTION 1: AI Taxonomy & Evolution
# ==============================================================================

class AICategory(Enum):
    RULE_BASED = "Rule-Based AI"
    MACHINE_LEARNING = "Machine Learning"
    DEEP_LEARNING = "Deep Learning"
    GENERATIVE_AI = "Generative AI"
    AGENTIC_AI = "Agentic AI"

@dataclass
class AIEra:
    category: AICategory
    era: str
    description: str
    example: str

def demonstrate_ai_taxonomy() -> None:
    print("=" * 60)
    print("SECTION 1: AI Taxonomy & Evolution")
    print("=" * 60)
    
    eras = [
        AIEra(
            category=AICategory.RULE_BASED,
            era="1950s-1980s",
            description="Explicitly programmed if-then logic.",
            example="Deep Blue, Expert Systems (MYCIN)"
        ),
        AIEra(
            category=AICategory.MACHINE_LEARNING,
            era="1990s-2010s",
            description="Algorithms that learn patterns from structured data.",
            example="Random Forests, Support Vector Machines"
        ),
        AIEra(
            category=AICategory.DEEP_LEARNING,
            era="2010s-2020",
            description="Multi-layered neural networks learning from unstructured data.",
            example="CNNs (AlexNet), RNNs"
        ),
        AIEra(
            category=AICategory.GENERATIVE_AI,
            era="2020-2023",
            description="Models generating novel content based on training distribution.",
            example="GPT-3, Midjourney"
        ),
        AIEra(
            category=AICategory.AGENTIC_AI,
            era="2023-Present",
            description="Goal-directed systems capable of planning, tool use, and iteration.",
            example="AutoGPT, Devin, LangChain Agents"
        )
    ]
    
    for era in eras:
        print(f"[{era.category.value}] ({era.era})")
        print(f"  Description: {era.description}")
        print(f"  Example: {era.example}\n")


# ==============================================================================
# SECTION 2: Neural Networks from First Principles
# ==============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def demonstrate_neural_network() -> None:
    print("=" * 60)
    print("SECTION 2: Neural Networks (XOR Problem)")
    print("=" * 60)
    
    # XOR problem
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    expected_output = np.array([[0],[1],[1],[0]])
    
    # Initialize weights
    np.random.seed(42)
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1
    hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
    hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
    output_bias = np.random.uniform(size=(1, outputLayerNeurons))
    
    lr = 0.1
    epochs = 10000
    
    print("Training a 2-layer Neural Network on XOR...\n")
    for _ in range(epochs):
        # Forward Pass
        hidden_layer_activation = np.dot(inputs, hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)
        
        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)
        
        # Backpropagation
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        
        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
        
        # Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
        
    print("Final predictions after training:")
    for i in range(len(inputs)):
        print(f"Input: {inputs[i]} => Predicted: {predicted_output[i][0]:.4f} (Expected: {expected_output[i][0]})")


# ==============================================================================
# SECTION 3: What LLMs Actually Do — Next-Token Prediction
# ==============================================================================

def demonstrate_next_token_prediction() -> None:
    print("\n" + "=" * 60)
    print("SECTION 3: Next-Token Prediction (Bigram Model)")
    print("=" * 60)
    
    corpus = "the agent plans the task and the agent executes the tool"
    tokens = corpus.split()
    
    # Build vocabulary
    vocab = list(set(tokens))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    
    # Transition matrix
    matrix = np.zeros((len(vocab), len(vocab)))
    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i+1]
        matrix[word_to_idx[current_word]][word_to_idx[next_word]] += 1
        
    # Normalize probabilities
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    matrix = np.nan_to_num(matrix)
    
    print("Bigram Transition Matrix Probabilities:")
    for word in vocab:
        idx = word_to_idx[word]
        next_idx = np.argmax(matrix[idx])
        prob = matrix[idx][next_idx]
        if prob > 0:
            print(f"'{word}' -> is most likely followed by -> '{idx_to_word[next_idx]}' (P = {prob:.2f})")
    
    print("\nGenerating text starting with 'the':")
    current = "the"
    generated = [current]
    for _ in range(5):
        idx = word_to_idx[current]
        probs = matrix[idx]
        if np.sum(probs) == 0:
            break
        next_idx = np.random.choice(len(vocab), p=probs)
        current = idx_to_word[next_idx]
        generated.append(current)
        
    print("Generated sequence:", " ".join(generated))


# ==============================================================================
# SECTION 4: Self-Attention Mechanism
# ==============================================================================

def demonstrate_self_attention() -> None:
    print("\n" + "=" * 60)
    print("SECTION 4: Self-Attention Mechanism")
    print("=" * 60)
    
    np.random.seed(42)
    seq_length = 3
    d_model = 4
    
    # Queries, Keys, Values (dummy matrices for 'the', 'agent', 'plans')
    Q = np.random.randn(seq_length, d_model)
    K = np.random.randn(seq_length, d_model)
    V = np.random.randn(seq_length, d_model)
    
    # Scaled dot-product attention
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Output
    output = np.dot(attention_weights, V)
    
    print("Input Tokens: ['the', 'agent', 'plans']")
    print("\nAttention Weights (How much each token attends to others):")
    print(np.round(attention_weights, 3))
    print("\nContextualized Output (Weighted sum of Values):")
    print(np.round(output, 3))


# ==============================================================================
# SECTION 5: Encoder vs Decoder Architectures
# ==============================================================================

def demonstrate_encoder_vs_decoder() -> None:
    print("\n" + "=" * 60)
    print("SECTION 5: Encoder vs Decoder Architectures")
    print("=" * 60)
    
    seq_length = 4
    
    # BERT-style Encoder Mask (Bidirectional - can see everything)
    encoder_mask = np.ones((seq_length, seq_length))
    print("Encoder (BERT-style) Attention Mask:\n(1 means token attends to it)")
    print(encoder_mask)
    
    # GPT-style Decoder Mask (Causal - can only see past/present)
    decoder_mask = np.tril(np.ones((seq_length, seq_length)))
    print("\nDecoder (GPT-style) Causal Attention Mask:\n(0 means future token, masked out)")
    print(decoder_mask)
    
    print("\nKey Difference: Decoders predict the *next* token by only looking at the past.")


# ==============================================================================
# SECTION 6: Why Agentic AI is the Frontier
# ==============================================================================

@dataclass
class AgentCapability:
    paradigm: str
    reasoning: str
    memory: str
    tool_use: str

def demonstrate_agentic_frontier() -> None:
    print("\n" + "=" * 60)
    print("SECTION 6: Why Agentic AI is the Frontier")
    print("=" * 60)
    
    capabilities = [
        AgentCapability(
            paradigm="Traditional ML",
            reasoning="None (Pattern matching)",
            memory="Frozen after training",
            tool_use="None"
        ),
        AgentCapability(
            paradigm="LLMs (Generative)",
            reasoning="Static (One-pass generation)",
            memory="Context window only",
            tool_use="Requires manual prompting"
        ),
        AgentCapability(
            paradigm="Agentic AI",
            reasoning="Dynamic (ReAct, Chain of Thought)",
            memory="Vector DBs, Stateful memory",
            tool_use="Autonomous API/Function calling"
        )
    ]
    
    for cap in capabilities:
        print(f"Paradigm: {cap.paradigm}")
        print(f"  Reasoning: {cap.reasoning}")
        print(f"  Memory:    {cap.memory}")
        print(f"  Tool Use:  {cap.tool_use}\n")


def main():
    demonstrate_ai_taxonomy()
    demonstrate_neural_network()
    demonstrate_next_token_prediction()
    demonstrate_self_attention()
    demonstrate_encoder_vs_decoder()
    demonstrate_agentic_frontier()

if __name__ == "__main__":
    main()

# llama3 Implementation from Scratch

This project implements the llama3 model from scratch, breaking down its complex processes into simpler steps. We leverage tensor and matrix multiplications to build the model piece by piece. The aim is to understand the underlying mechanisms of llama3 by dissecting and recreating it, while utilizing the official model weights provided by Meta. Here's a detailed description of the project:

## Overview

In this project, I implement llama3 from scratch. I start by downloading and loading the model weights from the official source provided by Meta. The implementation covers tokenization, embedding, and the various layers of the transformer architecture, including attention mechanisms and feedforward networks. By the end of this project, I will have a fully functional llama3 model capable of generating text.

## Requirements


- Python 3.x
- TensorFlow or PyTorch

- tiktoken (an OpenAI library for tokenization)
- Meta's llama3 model weights (download link provided in the code)

## Downloading Model Weights

Before running the implementation, you need to download the model weights from the official link provided by Meta. Ensure that these weights are stored in a directory accessible to the code.

## Tokenizer

For tokenization, I use `tiktoken`, an efficient tokenizer from OpenAI. This helps convert text into tokens that the model can process. I also provide a link to Andrej Karpathy's clean implementation of a BPE tokenizer for those interested.

## Reading the Model File

Instead of relying on predefined model classes, I read the model file one tensor at a time. This approach allows us to understand and manipulate the model's inner workings directly.

### Configuration

The model configuration includes:

- 32 transformer layers
- Each multi-head attention block has 32 heads

- A specific vocabulary size

## Converting Text to Tokens

Using `tiktoken`, I convert text inputs into tokens, which are then transformed into embeddings. This step includes normalizing the embeddings using RMS normalization to ensure stable values.

## Building the Transformer Layers

### Normalization

Each layer begins with normalization. I access layer-specific data from the model dictionary to maintain the correct shapes and values.

### Attention Mechanism

I implement attention from scratch:

1. **Loading Attention Heads:** I load query, key, value, and output vectors from the model.
2. **Unwrapping Queries:** Queries are unwrapped to separate attention heads, facilitating parallel processing.

3. **Rotary Positional Embedding (RoPE):** This technique encodes positional information into queries and keys using complex number rotations.

### Self-Attention

After obtaining rotated queries and keys, I perform self-attention by calculating scores that map how each token relates to others. I mask future tokens during training to prevent using future information, adhering to causal language modeling principles.

### Values and Attention Scores

Values are computed using value weights, and attention scores determine how much of the value matrix each token uses. This results in an attention vector for each token.

### Multi-Head Attention

The attention vectors from all heads are combined to form a comprehensive attention matrix. This matrix undergoes further transformations through linear layers and feedforward networks.

### Feedforward Network

I implement a SwiGLU feedforward network, commonly used in modern LLMs for its effectiveness in adding non-linearity.

## Final Embeddings and Decoding

After processing through all layers, I obtain the final embeddings, which encapsulate the model's best guess for the next token. The output decoder converts these embeddings back into token values, completing the text generation process.

## Example Output

Given the prompt "the answer to the ultimate question of life, the universe, and everything is", the model should ideally output "42", referencing Douglas Adams' "The Hitchhiker's Guide to the Galaxy".

## Conclusion

This implementation provides a deep dive into llama3, offering insights into each component of the model. By the end of this project, I successfully replicate the llama3 model's functionality, achieving a comprehensive understanding of its architecture and operations.

I hope you enjoy exploring this detailed reconstruction of llama3!

## References


- Meta's official link for downloading model weights
- Andrej Karpathy's BPE tokenizer implementation: [minBPE](https://github.com/karpathy/minbpe)
- Idea and Inspiration: [naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 
- tiktoken library for tokenization

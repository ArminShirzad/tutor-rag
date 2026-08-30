# Transformers and Attention

## Why attention replaced recurrence

Recurrent networks (RNNs, LSTMs) process a sequence one token at a time, carrying a
hidden state forward. This has two fatal problems at scale. First, it is inherently
sequential, so it cannot exploit parallel hardware. Second, information from early
tokens has to survive many steps of compression to influence a late token, so
long-range dependencies degrade.

**Attention** solves both. Every position can look directly at every other position
in a single operation, and the whole sequence is processed in parallel.

## Self-attention mechanics

Each token is projected into three vectors:

- **Query (Q)**: what this token is looking for
- **Key (K)**: what this token offers
- **Value (V)**: the content this token contributes

The attention output is:

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

Read it in steps. `Q @ K.T` gives every pair of tokens a compatibility score. The
softmax turns each row into weights summing to 1 -- how much each token should attend
to every other. Multiplying by V produces a weighted blend of the value vectors.

The `sqrt(d_k)` scaling is not cosmetic. For large key dimensions the dot products
grow large in magnitude, pushing softmax into a saturated region where gradients
vanish. Dividing by the square root of the key dimension keeps the variance stable.

## Multi-head attention

Instead of one attention operation over the full dimension, the model runs h parallel
attention "heads" over lower-dimensional projections and concatenates the results.
Different heads specialise: some track syntactic dependencies, some track coreference,
some attend to positional neighbours. It is the difference between one opinion and a
committee.

## Positional encoding

Attention is permutation-invariant -- it has no inherent notion of order. Shuffling
the input tokens would produce the same set of outputs. Position information must be
added explicitly.

The original transformer used fixed **sinusoidal** encodings added to the embeddings.
Modern models mostly use **learned** positional embeddings or **RoPE** (Rotary
Position Embedding), which encodes relative position by rotating Q and K vectors and
generalises better to sequences longer than those seen in training.

## Encoder, decoder, and the two families

The original architecture had both an encoder and a decoder. Modern models usually
pick one:

- **Encoder-only** (BERT): bidirectional attention, every token sees all others. Best
  for understanding tasks -- classification, named entity recognition, and producing
  embeddings for retrieval.
- **Decoder-only** (GPT, Llama, Gemini): causal attention, each token sees only
  previous tokens. Best for generation. This is what every modern LLM uses.
- **Encoder-decoder** (T5, BART): best for sequence-to-sequence tasks like translation
  and summarisation.

**Causal masking** is what makes decoder-only training work: positions after the
current one are masked to negative infinity before the softmax, so the model cannot
cheat by looking at the answer it is being asked to predict.

## The quadratic cost problem

Self-attention computes an n x n matrix for a sequence of length n. Doubling the
context length quadruples the compute and memory. This is the fundamental constraint
behind context window limits and why long-context models are expensive.

Mitigations include sparse attention patterns, sliding-window attention, and
**FlashAttention**, which does not change the mathematics but reorders the computation
to avoid writing the huge intermediate matrix to slow GPU memory.

## Tokenization

Models do not see characters or words -- they see **tokens**, produced by a subword
algorithm such as Byte Pair Encoding (BPE) or SentencePiece. Common words become one
token; rare words are split into pieces.

As a rule of thumb for English, one token is about four characters, or roughly 0.75
words. This matters directly for cost and for context budgeting, since APIs bill per
token. Non-English text, and code, generally use more tokens per unit of meaning.

## Temperature and sampling

At generation time the model outputs a probability distribution over the vocabulary.
How you sample from it controls the output's character.

- **Temperature** rescales the logits before softmax. Temperature near 0 makes the
  model nearly deterministic, always picking the most likely token. Higher values
  flatten the distribution and increase diversity and risk.
- **Top-k** sampling restricts choices to the k most likely tokens.
- **Top-p** (nucleus) sampling restricts to the smallest set of tokens whose
  cumulative probability exceeds p. Generally preferred over top-k because the set
  size adapts to how confident the model is.

For factual, grounded tasks such as retrieval-augmented question answering, use a low
temperature. Creativity is a bug when the requirement is faithfulness.

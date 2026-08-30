# Embeddings and Vector Search

## What an embedding is

An embedding is a dense vector of floating point numbers that represents the meaning
of a piece of text. Typical dimensions range from 384 to 3072. The defining property
is that semantically similar inputs produce vectors that are close together in the
space, even when they share no words at all.

This is what lets a search for "how do I stop my model memorising the training data"
retrieve a passage titled "Regularisation and Dropout". Keyword search finds none of
those words; vector search finds the meaning.

Embeddings are produced by a neural network, usually a transformer encoder. The final
token representations are pooled -- most commonly by mean pooling -- into a single
fixed-length vector.

## Measuring similarity

**Cosine similarity** measures the angle between two vectors and ignores their
magnitude:

    cos(a, b) = (a . b) / (|a| * |b|)

It ranges from -1 (opposite) through 0 (unrelated) to 1 (identical direction). It is
the standard choice for text because document length should not affect topical
similarity.

**Dot product** is cosine similarity without normalisation, so magnitude matters.
**Euclidean distance** measures straight-line distance; for normalised vectors it is
monotonically related to cosine similarity, so the ranking is identical.

The practical consequence: if you normalise all vectors to unit length in advance,
cosine similarity reduces to a plain dot product, and searching an entire corpus
becomes a single matrix multiplication.

## Bi-encoders and cross-encoders

A **bi-encoder** encodes the query and each document independently into separate
vectors and compares them with cosine similarity. Because document vectors can be
computed once, in advance, query time is fast and scales to millions of documents.
The cost is precision: the two texts never interact, so the model must compress the
whole document into one vector before it knows what will be asked.

A **cross-encoder** feeds the query and document through the transformer *together*
and outputs a single relevance score. Every query token can attend to every document
token, which is dramatically more accurate. The cost is that nothing can be
precomputed: scoring requires one forward pass per query-document pair, so it cannot
be run over a whole corpus.

This asymmetry produces the standard two-stage retrieval architecture: a bi-encoder
retrieves a wide candidate set cheaply, then a cross-encoder reranks that small set
precisely.

## Vector databases and indexing

Storing vectors is easy; searching them quickly is the hard part.

**Exact search** (also called flat, or brute force) compares the query against every
stored vector. It is O(n*d) and returns perfect results. On modern hardware this is
fast enough for hundreds of thousands of vectors, and it should be the default until
measurements prove otherwise.

**Approximate nearest neighbour (ANN)** search trades a small amount of recall for a
large speedup. The dominant algorithm is **HNSW** (Hierarchical Navigable Small World),
which builds a multi-layer graph and greedily walks it toward the query. Its key
parameters are `M` (graph connectivity, higher means better recall and more memory)
and `ef_search` (how many candidates to explore at query time, higher means better
recall and more latency).

**IVF** partitions vectors into clusters and searches only the nearest few clusters.
**Product Quantization (PQ)** compresses vectors into compact codes, trading accuracy
for a large reduction in memory.

Common vector stores include FAISS (a library, not a server), Chroma, Qdrant,
Weaviate, Pinecone, Milvus, and **pgvector**, which adds a vector column type and
HNSW/IVFFlat indexes to PostgreSQL. pgvector is attractive because it keeps vectors
in the same transactional database as the rest of the application data, so there is no
second system to synchronise and metadata filtering is just SQL.

## Sparse retrieval and BM25

**BM25** is a keyword ranking function and remains a strong baseline decades after its
introduction. It scores a document by how many query terms it contains, weighted by
how rare each term is across the corpus (inverse document frequency), with two
corrections: term frequency saturation, so the tenth occurrence of a word adds far
less than the second, and document length normalisation, so long documents do not win
by sheer size.

BM25 is excellent at exactly what embeddings are worst at: rare proper nouns, product
names, error codes, identifiers, and acronyms that the embedding model never saw
during training.

## Hybrid search

Because dense and sparse retrieval fail in different ways, combining them beats either
alone on almost every benchmark.

The difficulty is that their scores are not comparable -- BM25 scores are unbounded
and corpus-dependent, cosine scores live in a fixed range. Naively adding them is
fragile.

**Reciprocal Rank Fusion (RRF)** avoids the problem entirely by discarding scores and
using only ranks:

    RRF(d) = sum over retrievers of 1 / (k + rank(d))

with k conventionally 60. It requires no normalisation and no tuning, and it is robust
when one retriever returns nonsense. A document ranked highly by both retrievers
naturally rises above one ranked highly by only a single retriever.

## Choosing an embedding model

Considerations, roughly in order of importance:

1. **Dimension**: higher is usually more accurate but costs proportionally more
   storage and search time.
2. **Max sequence length**: text beyond it is silently truncated. `all-MiniLM-L6-v2`
   truncates at 256 word-pieces, which is a frequent and invisible source of lost
   information.
3. **Domain match**: a model trained on web text may perform poorly on legal, medical
   or code corpora.
4. **Symmetric vs asymmetric**: some models are trained so queries and documents are
   embedded differently. Using the wrong mode costs real accuracy and produces no
   error message.

The MTEB leaderboard is the standard public benchmark, but the only number that
matters is performance on your own evaluation set.

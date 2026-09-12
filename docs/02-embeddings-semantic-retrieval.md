# Embeddings and Semantic Retrieval

This guide documents the second retrieval stage of `ai-engineering-lab`: replacing manually designed lexical features with learned dense text embeddings.

The first guide, [Foundations of Lexical Retrieval](01-foundations-lexical-retrieval.md), explained tokenization, TF-IDF, cosine similarity, and lexical ticket retrieval. This guide builds on that foundation and explains the current embedding-based implementation.

The implementation is educational, but the architecture mirrors a production pattern:

```text
text
  -> embedding model
  -> dense vector
  -> vector index or matrix
  -> similarity search
  -> relevant ticket context
```

## Table of Contents

1. [Why Embeddings](#1-why-embeddings)
2. [The Current Architecture](#2-the-current-architecture)
3. [What an Embedding Is](#3-what-an-embedding-is)
4. [TextEmbedder](#4-textembedder)
5. [Embedding Dimensions and Shapes](#5-embedding-dimensions-and-shapes)
6. [Batch Encoding and Indexing](#6-batch-encoding-and-indexing)
7. [SemanticTicketRetriever](#7-semanticticketretriever)
8. [Vectorized Cosine Similarity](#8-vectorized-cosine-similarity)
9. [Lexical Versus Semantic Retrieval](#9-lexical-versus-semantic-retrieval)
10. [Testing Without Repeated Model Downloads](#10-testing-without-repeated-model-downloads)
11. [Educational Versus Production Design](#11-educational-versus-production-design)
12. [Rails Developer Terminology Map](#12-rails-developer-terminology-map)
13. [Current Status](#13-current-status)
14. [What Comes Next](#14-what-comes-next)
15. [Glossary](#15-glossary)

## 1. Why Embeddings

TF-IDF and bag-of-words are lexical representations. They are excellent at exact terms, ticket IDs, error codes, and names, but they mostly depend on word overlap.

For example:

```text
"payment failed"
"transaction was declined"
```

These sentences may be related to a human, but they share few exact words. A lexical retriever may score them weakly. An embedding model can place texts with related meanings near each other in a learned vector space.

This is the transition in the project:

```text
manual lexical retrieval
        -> NumPy vectorized lexical retrieval
        -> dense embeddings
        -> semantic retrieval
```

The goal is not to declare semantic search universally better. It is to understand what it adds, what it loses, and how to evaluate it against the lexical baseline.

## 2. The Current Architecture

The current application-level flow is:

```text
Historical ticket text
        |
        v
TextEmbedder
        |
        v
Dense embedding matrix
        |
        v
SemanticTicketRetriever
        |
        v
Vectorized cosine similarity
        |
        v
Stable top-k ranking
        |
        v
RetrievedTicket objects
```

There are two important phases.

### Indexing time

`SemanticTicketRetriever` receives historical tickets, materializes their IDs and text, loads or receives an embedder, embeds all historical texts in one batch, and stores the embeddings as a NumPy matrix.

This work belongs in an offline indexing job or a refresh operation when the corpus changes. It should not happen for every query.

### Query time

`search()` validates the query, embeds it with the same model, compares it with the stored matrix, ranks scores, and maps selected rows back to ticket IDs and original text.

The same embedding model must be used for historical tickets and queries. If two models or incompatible dimensions are mixed, the vectors do not share a valid feature space.

## 3. What an Embedding Is

An embedding is a learned dense numerical representation of an item. Here the item is text. Instead of assigning one explicit coordinate to each vocabulary word, the model produces a fixed-length vector whose coordinates jointly encode patterns learned from language data.

A simplified illustrative example might look like this:

```text
"password reset failed"
-> [0.12, -0.44, 0.31, 0.08, ...]
```

The numbers are not hand-authored meanings such as “password” or “failure.” Their meaning is distributed across the vector and learned by the model. Nearby vectors can represent related language even when the exact words differ.

The current default model is `all-MiniLM-L6-v2`. Its vectors have dimension 384 in the installed environment. The application discovers that dimension from the loaded model rather than hardcoding it.

Embeddings are not magic labels. They reflect the model's training data, architecture, pooling strategy, preprocessing, and numerical behavior. Their usefulness must be measured on the target support-ticket domain.

## 4. TextEmbedder

The application wrapper lives in:

```text
src/ai_engineering_lab/embeddings.py
```

`TextEmbedder` owns the boundary to `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._embedding_dimension = self._model.get_embedding_dimension()
```

The wrapper has three responsibilities:

- load one model instance in its constructor;
- validate input types at the application boundary;
- convert library output into plain Python lists of floats.

Its public operations are:

- `embed_text(text)` for one string;
- `embed_texts(texts)` for batch encoding;
- `model_name` for the configured model name;
- `embedding_dimension` for the model-provided vector size.

`embed_texts()` preserves input order because each output row corresponds to the input at the same position. It returns `[]` for an empty collection without invoking the model. Empty strings are allowed and passed to the underlying model; the wrapper does not invent special semantics for them.

The model is loaded once per `TextEmbedder` instance. Recreating it for every query would add latency and memory pressure, and could produce inconsistent behavior if model configuration changes between calls.

## 5. Embedding Dimensions and Shapes

A vector's dimension is its number of coordinates. The current real model produces vectors with shape `(384,)` for one text.

When multiple texts are encoded, the result can be represented as a matrix:

```text
number of texts x embedding dimension
```

For three historical tickets using the fake test embedder:

```text
embedding matrix shape: (3, 3)
```

With the real MiniLM model and three tickets, the conceptual shape is:

```text
(3, 384)
```

The first number is the number of rows, where each row represents one ticket. The second number is the shared embedding dimension, where each column is one coordinate in the model's learned space.

Shape validation matters because matrix operations depend on compatible dimensions. A matrix with shape `(N, D)` can compare with a query vector of shape `(D,)`. A query of shape `(D + 1,)` cannot be compared correctly and should fail clearly rather than produce a misleading result.

## 6. Batch Encoding and Indexing

The semantic retriever embeds historical tickets in one batch:

```python
ticket_texts = [ticket["text"] for ticket in self._historical_tickets]
self._embedding_matrix = vectors_to_matrix(
    self._embedder.embed_texts(ticket_texts)
)
```

Batch encoding is useful for two reasons:

1. It avoids repeated model setup and call overhead.
2. Embedding libraries can often process a batch more efficiently than many individual calls.

The batch output preserves historical ticket order. Therefore row `0` of the matrix belongs to historical ticket `0`, row `1` belongs to ticket `1`, and so on. The retriever keeps the original metadata separately and uses the selected row indexes to reconstruct domain results.

This is similar to an offline ingestion job in a Rails application. A job reads records, transforms them, and writes an index that request-time code can query. The request path should not recompute all historical embeddings.

## 7. SemanticTicketRetriever

The implementation lives in:

```text
src/ai_engineering_lab/semantic_retrieval.py
```

The constructor accepts:

```python
SemanticTicketRetriever(
    historical_tickets,
    embedder=None,
)
```

Each ticket must provide `ticket_id` and `text`. The constructor materializes the input so generators and other one-use iterables work consistently. Empty historical collections raise `ValueError` because there is nothing to search.

If no embedder is supplied, the class creates `TextEmbedder()`. If one is supplied, it preserves that exact instance. Dependency injection is important here: production code can choose a model, while tests can provide a deterministic fake without downloading model weights.

The public properties are:

- `embedder`: the model wrapper used for both documents and queries;
- `embedding_matrix`: the stored dense matrix;
- `embedding_dimension`: the model's dimension.

The search flow is:

1. validate that `query` is a string;
2. validate `top_k`;
3. return immediately for `top_k == 0`;
4. generate one query embedding;
5. convert it into a one-dimensional NumPy array;
6. verify its dimension matches the stored matrix;
7. calculate vectorized cosine similarities;
8. sort scores stably in descending order;
9. convert selected indexes to `RetrievedTicket` objects.

The result preserves the ticket ID, original text, and similarity score. This keeps model-oriented numerical data behind a domain-level result that the rest of the application can use.

## 8. Vectorized Cosine Similarity

The semantic retriever uses the same cosine concept as the lexical retriever, but performs the calculation over a dense matrix.

If:

```text
ticket_matrix has shape (N, D)
query_vector has shape (D,)
```

then:

```python
dot_products = ticket_matrix @ query_vector
```

has shape `(N,)`. Each output value is the dot product between one ticket row and the query.

The implementation then calculates:

```python
ticket_norms = np.linalg.norm(ticket_matrix, axis=1)
query_norm = np.linalg.norm(query_vector)
denominator = ticket_norms * query_norm
```

The row norms have shape `(N,)`; the query norm is a scalar. NumPy broadcasting makes the scalar participate in every row denominator. The guarded division is:

```python
similarities = np.zeros(ticket_matrix.shape[0], dtype=float)
np.divide(
    dot_products,
    denominator,
    out=similarities,
    where=denominator != 0.0,
)
```

If a ticket vector or query vector is all zeros, its denominator is zero and the output remains `0.0`. This avoids division-by-zero warnings and gives the application a defined behavior for empty or unrecognized input.

Stable ranking uses:

```python
ranked_indexes = np.argsort(-similarities, kind="stable")[:top_k]
```

The negative sign sorts from highest score to lowest. Stable sorting preserves the original ticket order when scores are tied, making tests and user-visible behavior deterministic.

The numerical calculation does not loop over individual tickets in Python. Only the final conversion of selected indexes into `RetrievedTicket` objects uses a small loop. That is acceptable because it handles metadata construction after the expensive similarity work is complete.

## 9. Lexical Versus Semantic Retrieval

Lexical retrieval represents text through explicit terms. TF-IDF can be strong for:

- exact ticket IDs;
- error codes;
- product names;
- configuration keys;
- rare terms that must match exactly.

Semantic retrieval represents text through learned dense vectors. It can be strong when meaning is shared but wording differs:

```text
"payment failed"
"transaction was declined"
```

A semantic model may recognize these as related even without a shared term. It can also connect paraphrases such as “I cannot sign in” and “login is blocked.”

Semantic retrieval has risks. A model may associate texts that are broadly related but operationally different. It may be weaker for exact identifiers, unusual internal names, or newly introduced product terms. Model behavior can also vary by domain and language.

Real systems often use hybrid retrieval:

```text
lexical candidates
        + semantic candidates
        -> merged candidates
        -> optional reranker
        -> top-k context
```

The current project keeps lexical and semantic retrievers separate so their behavior can be compared before combining them.

## 10. Testing Without Repeated Model Downloads

The semantic tests are in:

```text
tests/test_semantic_retrieval.py
```

Most tests use `FakeEmbedder`. It exposes the same small interface needed by the retriever:

- `embedding_dimension`;
- `embed_text()`;
- `embed_texts()`.

It returns deterministic 3D vectors such as:

```python
"password reset" -> [1.0, 0.0, 0.0]
"invoice payment" -> [0.0, 1.0, 0.0]
"dashboard request" -> [0.0, 0.0, 1.0]
```

This makes expected rankings, zero vectors, dimension mismatches, and stable ties easy to test. It also records batch calls, allowing the test to prove that historical texts are embedded together and that query calls use the single-text method.

The fake avoids downloading the real model for every test and avoids making the entire suite dependent on network access. The real `TextEmbedder` is tested separately in `tests/test_embeddings.py`; those tests load `all-MiniLM-L6-v2` and verify dimensions, types, batch behavior, and input validation without asserting exact model values.

Testing shape and behavior instead of exact embedding coordinates is important. Model implementations, dependency versions, hardware paths, and numerical details can change small floating-point values while preserving the useful contract.

## 11. Educational Versus Production Design

| Current educational design | Typical production alternative |
| --- | --- |
| `TextEmbedder` wraps one local sentence-transformers model | a versioned model service, worker, or managed embedding API |
| model loaded in the application object | model lifecycle managed by a long-lived worker or inference server |
| dense NumPy matrix in memory | vector database or approximate nearest-neighbor index |
| brute-force comparison with every ticket | FAISS, Qdrant, pgvector, Pinecone, Weaviate, or another index |
| model output represented as Python lists | validated numerical arrays or storage-native vector types |
| metadata kept in a parallel Python list | durable records with IDs, filters, permissions, and index metadata |
| fake embedder for unit tests | contract tests plus sampled model integration tests |

The educational implementation is intentionally transparent. It shows exactly where model loading, batch encoding, shape checks, norms, ranking, and metadata mapping occur. Production systems need more: persistent indexes, incremental updates, tenant isolation, access-control filtering, retries, timeouts, observability, model versioning, and evaluation datasets.

Manual understanding remains valuable even when a vector database owns the final search. Engineers still need to reason about model compatibility, dimensions, normalization, stale embeddings, score interpretation, and whether retrieved context is actually relevant.

## 12. Rails Developer Terminology Map

These analogies are useful but not exact equivalents:

| Rails concept | Embedding/AI Engineering concept |
| --- | --- |
| database record | document or retrieval item |
| model callback or background job | offline embedding/indexing job |
| service object | embedding or retrieval orchestration class |
| dependency injection | supplied embedder/model implementation |
| database index | vector index |
| ActiveRecord query scope | metadata filter or retrieval constraint |
| serializer | `RetrievedTicket` domain result |
| request parameter | query text and retrieval options |
| RSpec double | fake embedder |
| application cache | cached model or embedding matrix |
| logs and metrics | retrieval/model observability |
| deploy migration | model/index version migration |

The main difference is that a normal database lookup often asks whether values match a predicate. Semantic retrieval asks which items are nearest in a learned numerical space. That makes evaluation, model versioning, and score interpretation first-class concerns.

## 13. Current Status

The semantic phase currently contains:

- `embeddings.py`: `TextEmbedder` around `sentence-transformers`;
- `semantic_retrieval.py`: dense embedding retrieval with vectorized cosine similarity;
- `tests/test_embeddings.py`: real model wrapper tests;
- `tests/test_semantic_retrieval.py`: deterministic fake-embedder retriever tests.

The default model is:

```text
all-MiniLM-L6-v2
```

Its current embedding dimension is 384. The semantic retriever tests use a fake embedder with a matrix shape of `(3, 3)` so they can verify numerical behavior quickly and deterministically.

The full project suite currently passes 80 tests based on the latest run. The real model-backed tests may take longer on a first run because model files must be downloaded and loaded locally.

## 14. What Comes Next

The next engineering questions are not only about generating vectors. They include:

- how to refresh embeddings when tickets change;
- how to store and query vectors durably;
- how to filter by tenant, permissions, ticket status, or product;
- how to evaluate semantic relevance with labeled queries;
- how to compare lexical, semantic, and hybrid retrieval;
- how to rerank candidates before giving them to an LLM;
- how to monitor latency, empty results, score distributions, and model drift.

The broader progression is:

```text
Manual lexical retrieval
        |
        v
NumPy vectorized lexical retrieval
        |
        v
Dense embeddings
        |
        v
Semantic retrieval
        |
        v
Hybrid retrieval
        |
        v
RAG
        |
        v
LLM-powered SaaS feature
```

## 15. Glossary

- **Embedding:** A learned numerical representation of an item, usually a dense vector.
- **Dense vector:** A vector whose dimensions are generally non-zero and jointly encode information.
- **Embedding model:** A model that maps input items such as text to vectors.
- **Model dimension:** The number of coordinates in each output embedding.
- **Embedding matrix:** A 2D array containing one embedding row per item.
- **Semantic retrieval:** Retrieval based on similarity in a learned representation space.
- **Lexical retrieval:** Retrieval based mainly on explicit token matching and weighting.
- **Corpus:** The collection of historical documents available for retrieval.
- **Indexing:** Preparing and storing representations for later query-time search.
- **Query vector:** The embedding generated for a new search query.
- **Cosine similarity:** A score based on the angle between two vectors.
- **Dot product:** The sum of pairwise coordinate products.
- **Norm:** The numerical length or magnitude of a vector.
- **Broadcasting:** NumPy's rule for applying compatible arrays or scalars across dimensions.
- **Top-k:** The first `k` results after ranking.
- **Stable sorting:** Sorting that preserves input order for equal keys.
- **Reranking:** Applying a second scoring stage to reorder retrieved candidates.
- **Vector index:** A data structure optimized for nearest-neighbor search.
- **Hybrid retrieval:** Combining lexical and semantic retrieval signals.
- **RAG:** Retrieval-augmented generation, where retrieved context is supplied to a generative model.

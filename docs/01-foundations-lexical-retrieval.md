# Foundations of Lexical Retrieval

This guide documents the first stage of `ai-engineering-lab`: building a small support-ticket retrieval system from deterministic rules, text features, similarity math, and evaluation. It is written for an experienced Ruby on Rails developer moving into AI Engineering.

The examples use the APIs that currently exist in this repository. The implementation is intentionally educational: it makes the mechanics visible before production libraries and model services are introduced.

## Table of Contents

1. [Project Context](#1-project-context)
2. [AI Engineering System We Are Building](#2-ai-engineering-system-we-are-building)
3. [Deterministic Classification](#3-deterministic-classification)
4. [Text Preprocessing and Tokenization](#4-text-preprocessing-and-tokenization)
5. [Bag-of-Words and Vocabulary](#5-bag-of-words-and-vocabulary)
6. [TF-IDF](#6-tf-idf)
7. [Vectors and Cosine Similarity](#7-vectors-and-cosine-similarity)
8. [Similar-Ticket Retrieval](#8-similar-ticket-retrieval)
9. [Retrieval Evaluation](#9-retrieval-evaluation)
10. [NumPy Foundations](#10-numpy-foundations)
11. [NumPy-Based Ticket Retrieval](#11-numpy-based-ticket-retrieval)
12. [Sparse Versus Dense Vectors](#12-sparse-versus-dense-vectors)
13. [Lexical Search Versus Semantic Search](#13-lexical-search-versus-semantic-search)
14. [Educational Implementation Versus Production Implementation](#14-educational-implementation-versus-production-implementation)
15. [Rails Developer to AI Engineer Terminology Map](#15-rails-developer-to-ai-engineer-terminology-map)
16. [Current Project Status](#16-current-project-status)
17. [What Comes Next: Embeddings](#17-what-comes-next-embeddings)
18. [Glossary](#18-glossary)

## 1. Project Context

`ai-engineering-lab` is a small, incremental Python project for learning how AI-enabled systems are designed, tested, and operated. It starts with support-ticket data because tickets are concrete domain documents: they have text, identifiers, labels, operational meaning, and a natural path to search and assistance.

The project follows a practical principle:

> Learning + building = production-ready systems.

Learning gives us the mental model to reason about algorithms, tradeoffs, and failures. Building gives us interfaces, tests, data flow, validation, and operational boundaries. Each milestone adds one usable layer rather than producing isolated coding exercises that are difficult to connect later.

The overall direction is:

```text
customer support tickets
        -> retrieval
        -> relevant context
        -> LLM-powered support assistant
```

The eventual assistant may use retrieved historical tickets to answer a new support request, suggest a response, route work, or provide context to a human agent. We are building the retrieval foundation first because an LLM cannot reliably use context that the application cannot select, measure, and explain.

The project is incremental for the same reason Rails applications are usually built around explicit domain boundaries: we want each responsibility to be understandable and testable before adding another dependency or abstraction. A hand-built baseline also makes later improvements measurable. When an embedding model or vector database arrives, we should know what problem it improves and what behavior it changes.

## 2. AI Engineering System We Are Building

The current conceptual architecture is:

```text
Historical support tickets
        |
        v
Data preparation
        |
        v
Text representation
        |
        v
Vectorization
        |
        v
Similarity search
        |
        v
Top-k retrieval
        |
        v
Evaluation
        |
        v
Relevant context for an AI assistant
```

### Offline/indexing time

At indexing time, the system receives the historical corpus. It learns the vocabulary and IDF statistics, transforms every ticket into the shared feature space, and stores the resulting representations. In this repository, `TicketRetriever` and `NumPyTicketRetriever` do that preparation in their constructors.

This is analogous to an ingestion or indexing job in a Rails system. It should not repeat for every web request.

### Query time

At query time, a new ticket or user question is transformed using the already-fitted representation. The system compares the query representation with stored representations, ranks candidates, and returns the top results. The vectorizer is not fitted again: doing so would change the feature space and make stored vectors incomparable.

### Evaluation time

At evaluation time, we compare returned IDs with known relevant IDs. `precision_at_k()` and `recall_at_k()` make retrieval quality measurable across examples rather than relying on a plausible-looking demo.

### Eventual production inference time

In production, retrieval becomes one stage in a larger inference request. The application may retrieve context, apply access control and filtering, build a prompt, call an LLM, validate the output, log the decision, and return a response. Retrieval quality and LLM quality are separate concerns, so each needs its own tests and metrics.

## 3. Deterministic Classification

The classifier lives in `src/ai_engineering_lab/classifier.py`. It classifies ticket text into three structured dimensions:

- `Category`: account, authentication, billing, general, or technical;
- `Priority`: high, low, or medium;
- `Sentiment`: negative, neutral, or positive.

`Category`, `Priority`, and `Sentiment` are `StrEnum` values. They constrain outputs to known values while remaining friendly to serialization. `ClassificationResult` is a frozen, slotted dataclass containing one value from each enum.

The classifier normalizes text with `casefold()` and applies deterministic keyword rules. For example, a ticket containing `password` or `login` is classified as authentication. A ticket containing `invoice` or `payment` is classified as billing. Rule order matters because broader terms can overlap with more specific terms.

This is a rule-based system. Its behavior is deterministic: the same input produces the same output without model weights, random sampling, network calls, or changing training data. The rules are heuristic features because they are useful signals rather than a learned understanding of language.

| Simple concept | AI Engineering terminology |
| --- | --- |
| keyword checks | heuristic features |
| output category | predicted label |
| result object | structured prediction |
| fixed rules | deterministic baseline |

The classifier is a useful baseline because it gives the project a stable domain contract before machine learning is introduced. It also makes failure modes visible. A substring rule may be too broad, a phrase may be ambiguous, and a rule change can be tested directly.

Rule-based logic remains useful in production AI systems. It can handle routing, validation, safety guardrails, fallbacks, permissions, and deterministic business rules. A learned model does not need to replace every rule. In many systems, rules constrain what a model is allowed to do and provide a reliable path when the model is uncertain or unavailable.

## 4. Text Preprocessing and Tokenization

The TF-IDF implementation uses a small regular expression tokenizer in `src/ai_engineering_lab/tfidf.py`:

```python
_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(document: str) -> list[str]:
    return _TOKEN_PATTERN.findall(document.casefold())
```

In plain language, it lowercases the text and extracts word-like pieces. For example:

```text
Before: "Password reset failed!"
After:  ["password", "reset", "failed"]
```

Case folding makes `Password`, `password`, and many other case variants share a feature. Consistent preprocessing matters because tokenization defines the feature space. If indexing turns `Reset!` into one token but querying turns it into another, the system loses an otherwise obvious match.

Preprocessing affects every later step: vocabulary construction, document frequency, TF-IDF values, vector dimensions, and similarity. It is therefore part of the representation contract, not a cosmetic cleanup step.

This tokenizer is intentionally limited. It treats punctuation as boundaries, includes word-like numeric strings, and does not understand all languages or domain syntax. Production preprocessing may need decisions about punctuation, numbers, multilingual text, stemming, lemmatization, domain-specific terms, code, IDs, stack traces, URLs, and error messages. For support tickets, removing an identifier or error code may destroy a valuable exact-match signal. Conversely, preserving every timestamp or random UUID can create noisy features.

Production systems often make preprocessing configurable and versioned. A change to tokenization can change the vocabulary and invalidate an index, so the preprocessing version should be recorded with indexed data.

## 5. Bag-of-Words and Vocabulary

A vocabulary is the mapping from known tokens to fixed vector positions. `BagOfWordsVectorizer` in `src/ai_engineering_lab/vectorizer.py` builds it in first-seen order. The order is deterministic for a given input corpus.

Consider:

```text
Vocabulary: ["payment", "failed", "login"]
Document:   "payment failed"
Vector:     [1, 1, 0]
```

The vector has one coordinate per vocabulary feature. `1` means the token occurred once; `0` means it did not occur. The position has meaning only because the vocabulary defines the mapping. A different ordering would produce a different-looking vector for the same document, even though the document has not changed.

All documents must share the same feature ordering. Otherwise, the first coordinate might mean `payment` in one vector and `login` in another, making arithmetic meaningless. This is why a vectorizer is fitted once and reused for later documents.

The model is called bag-of-words because it records token presence or counts but discards word order and most linguistic structure. It is a sparse concept: a document usually uses a small fraction of a large vocabulary. The educational implementation stores dense Python lists, which is simple to inspect but wasteful for large vocabularies. Production systems often use sparse matrix representations that store only non-zero entries.

## 6. TF-IDF

`src/ai_engineering_lab/tfidf.py` extends the same shared-vocabulary idea by weighting tokens. A token should matter more when it is frequent in the current document but less common across the whole corpus.

### Core terms

- **Term frequency (TF):** how often a term occurs in one document, normalized here by the document's total token count.
- **Document frequency (DF):** how many fitted documents contain the term at least once.
- **Inverse document frequency (IDF):** a weight that is higher for terms appearing in fewer documents.

The conceptual formula is:

```text
TF-IDF(term, document) = TF(term, document) * IDF(term)
```

The implementation uses normalized TF:

```text
TF(term, document) = count(term in document) / total number of tokens in document
```

It uses this smoothed IDF formula:

```text
IDF(term) = log((1 + N) / (1 + DF)) + 1
```

Here, `N` is the number of fitted documents. Smoothing prevents problematic zero divisions and keeps the formula defined at the edges. Common words receive lower IDF because they occur in many documents. Distinctive words receive higher IDF because they help separate one document from the rest.

### The vectorizer lifecycle

- `fit(documents)` clears prior statistics, builds the vocabulary, counts document frequency, and calculates IDF.
- `transform(documents)` uses those learned statistics to create vectors. Repeated terms are counted correctly, unknown terms are ignored as features, and empty documents produce all-zero vectors.
- `fit_transform(documents)` fits and transforms the same input, while preserving support for one-use iterables by materializing them once.

Feature ordering is deterministic because vocabulary entries are assigned in first-seen order. Out-of-vocabulary tokens do not add dimensions. A query must use the historical vocabulary even when it contains words that never appeared during fitting.

The most important operational rule is:

> Fit on the historical corpus once. Transform new queries using the same fitted vectorizer. Do not fit a new vectorizer for every query.

Fitting on every query would change the vocabulary and IDF values. The new query vector would no longer inhabit the same feature space as stored historical vectors. It would also waste work on every request.

These algorithms are manually implemented because this project is teaching the mechanics. Production libraries provide optimized and more configurable implementations, but understanding the manual version helps with debugging, architecture decisions, and interpreting library behavior.

## 7. Vectors and Cosine Similarity

A vector is an ordered collection of numbers. Its dimension is the number of coordinates. In lexical retrieval, each coordinate represents one shared vocabulary feature.

The dot product combines matching coordinates:

```text
dot(a, b) = a1*b1 + a2*b2 + ... + an*bn
```

The vector magnitude, or L2 norm, is its Euclidean length:

```text
||a|| = sqrt(a1² + a2² + ... + an²)
```

Cosine similarity compares direction rather than raw length:

```text
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
```

The existing `cosine_similarity()` in `tfidf.py` validates equal lengths, calculates the dot product and both magnitudes, and returns `0.0` when either vector has zero magnitude. This avoids division by zero for empty or all-unknown text.

Two identical non-zero vectors point in the same direction and have similarity `1.0`. Orthogonal vectors have a dot product of zero and similarity `0.0`. A zero vector has no direction, so the implementation defines its similarity as `0.0` rather than attempting an undefined division.

Magnitude and direction are different concepts. Two documents can contain different amounts of text but have similar relative term weighting. Cosine similarity reduces the effect of total vector length and focuses on the orientation of the feature weights. It is useful for text retrieval because overlapping, distinctive terms tend to point documents in similar directions.

## 8. Similar-Ticket Retrieval

`src/ai_engineering_lab/retrieval.py` defines `TicketRetriever` and the immutable `RetrievedTicket` result dataclass.

The retriever receives historical records with `ticket_id` and `text`. During initialization it stores the records, fits one `TfidfVectorizer` on the historical texts, and stores the transformed vectors. This is the indexing or preparation phase.

During `search(query, top_k=3)`, it transforms the new query with the fitted vectorizer, compares it with every stored vector using cosine similarity, ranks the results, and maps vector indexes back to the original ticket ID and text. The result is a list of `RetrievedTicket` objects rather than anonymous numeric rows.

The production terminology is:

| Project concept | AI Engineering terminology |
| --- | --- |
| historical tickets | retrieval corpus |
| fitted vectorizer | feature encoder |
| stored vectors | indexed representations |
| query vector | query representation |
| `top_k` | retrieval cutoff |
| similarity ranking | candidate ranking |
| `RetrievedTicket` | domain retrieval result |

Fitting during every search would be both incorrect and inefficient. It would allow the query to redefine the feature space, make stored vectors incompatible, and repeat corpus-wide work for every request. The index should be built or refreshed when the corpus changes, not when an individual query arrives.

## 9. Retrieval Evaluation

`src/ai_engineering_lab/evaluation.py` contains simple metrics for comparing retrieval output with relevance labels. A relevance label says which historical ticket IDs are considered useful for a query. Those labels form the ground truth for the evaluation example.

For the first `k` returned IDs:

```text
precision@k = relevant results in first k results / k
```

```text
recall@k = relevant results in first k results / total relevant results
```

Precision asks: “How much of what we returned was relevant?” Recall asks: “How much of the relevant material did we find?”

Precision matters when irrelevant context is costly. In a retrieval-augmented assistant, unrelated tickets can distract an LLM, consume context window space, and increase the chance of a wrong answer. Recall matters when missing a useful ticket is costly, such as omitting a known workaround or incident resolution.

The retrieval cutoff `k` controls the result window. The implementation validates that `k` is greater than zero. `precision_at_k()` uses the documented denominator `k`, even when fewer than `k` IDs are available. `recall_at_k()` considers only the first `k` IDs and raises `ValueError` when the relevant set is empty because the denominator would be undefined.

The metrics are educational and intentionally small. They show why retrieval should be evaluated rather than judged only by intuition or by one attractive demo. Production evaluation usually has many queries, labeled relevance judgments, aggregate statistics, slices by ticket type, and regression thresholds.

## 10. NumPy Foundations

`src/ai_engineering_lab/numpy_basics.py` introduces NumPy's `ndarray`, a typed numerical array. A one-dimensional array represents a vector; a two-dimensional array represents a matrix.

Useful properties include:

- **shape:** the size along each axis, such as `(3,)` for a vector or `(2, 3)` for two rows of three features;
- **dtype:** the numerical type, such as floating-point values;
- **dimension count:** the number of axes, available through `ndim`.

The module's `vector_from_values()` converts a Python iterable into a one-dimensional float array. `dot_product()` uses `np.dot`, `vector_norm()` uses `np.linalg.norm`, and `cosine_similarity_numpy()` combines those operations while returning `0.0` for zero norms. `vectors_to_matrix()` converts nested vectors into a rectangular two-dimensional float matrix and rejects invalid shapes.

The terminology mapping is:

| Python concept | Numerical/AI Engineering terminology |
| --- | --- |
| Python list | numerical array |
| nested lists | feature matrix |
| vector length | feature dimension |
| array shape | tensor shape |
| dot product | vector similarity operation |
| matrix rows | individual documents/items |
| matrix columns | shared features |

For example:

```python
vector = vector_from_values([1, 2.5, 3])
# array([1. , 2.5, 3. ]) with shape (3,) and float dtype

matrix = vectors_to_matrix([[1, 2], [3, 4]])
# array([[1., 2.], [3., 4.]]) with shape (2, 2)
```

Shape correctness is important in machine learning systems because matrix operations depend on compatible dimensions. A `(N, D)` document matrix can multiply a `(D,)` query vector to produce `(N,)` scores. A reversed or inconsistent shape can silently produce a wrong calculation or fail at runtime. Explicit validation turns that error into a useful boundary failure.

## 11. NumPy-Based Ticket Retrieval

`src/ai_engineering_lab/numpy_retrieval.py` defines `NumPyTicketRetriever`. It reuses the existing Python TF-IDF implementation for feature construction, then changes the storage and similarity calculation to NumPy arrays.

Historical vectors are stored as a matrix with shape:

```text
(number_of_tickets, number_of_features)
```

If there are `N` tickets and `D` vocabulary features, then `ticket_matrix` has shape `(N, D)`. The query vector has shape `(D,)`.

The central operation is:

```python
dot_products = ticket_matrix @ query_vector
```

Conceptually:

```text
ticket_matrix:  (N, D)
query_vector:   (D,)
result:         (N,)
```

Each output value is one historical ticket row's dot product with the query. Matrix multiplication calculates all ticket/query dot products together rather than using a Python loop over tickets.

The retriever then computes row-wise ticket norms with `np.linalg.norm(ticket_matrix, axis=1)` and the query norm with `np.linalg.norm(query_vector)`. Multiplying the row norms by the scalar query norm uses broadcasting: NumPy applies the scalar-compatible operation to every row denominator. `np.divide(..., where=...)` writes only where the denominator is non-zero, leaving zero similarity for zero vectors without warnings or division by zero.

Stable `np.argsort` ranks scores in descending order, and slicing applies `top_k`. The final conversion from indexes to `RetrievedTicket` objects uses a small Python loop. That loop handles application metadata mapping, not the expensive numerical calculation, so it is an appropriate boundary between vectorized math and domain objects.

Vectorized operations are generally faster than Python loops over large collections because the array work runs in optimized numerical kernels with less interpreter overhead. This is not a promise that every NumPy expression is faster for every input; it is a reason to represent bulk numerical work as arrays and measure performance at realistic sizes.

The implementation rejects an empty historical collection clearly. It validates query dimension compatibility, handles `top_k == 0`, and assigns zero scores for unknown-term queries.

## 12. Sparse Versus Dense Vectors

### Sparse vectors

Sparse vectors contain mostly zero values. Bag-of-Words and TF-IDF commonly produce sparse vectors because a document uses only a small subset of a potentially large vocabulary. Dimensions still correspond to vocabulary features, so exact word overlap remains visible.

Illustrative example only:

```text
Vocabulary: [payment, failed, login, dashboard, invoice]
Sparse-ish TF-IDF vector: [0.8, 0.6, 0.0, 0.0, 0.0]
```

The numbers are illustrative, not output from a particular ticket in this project.

### Dense vectors

Dense vectors have many non-zero values. They are commonly generated by learned embedding models, where each coordinate does not represent one explicit word. Instead, the dimensions jointly encode distributed information about meaning and usage.

Illustrative example only:

```text
Dense embedding: [0.12, -0.44, 0.31, 0.08, ...]
```

TF-IDF mainly captures word overlap and corpus-level importance. Embeddings attempt to capture semantic relationships in a distributed representation. Sparse lexical features are often valuable for exact terms; dense features are often valuable for paraphrases.

## 13. Lexical Search Versus Semantic Search

Lexical search is based primarily on matching terms and weighting their importance. It can work very well for exact words, ticket IDs, product names, error codes, and configuration keys. It may struggle when two texts express the same idea with different vocabulary.

Semantic search uses learned representations, usually embeddings, to connect related meanings even when wording differs. It may be less reliable for exact identifiers, rare codes, or strings that must match precisely unless lexical signals are combined with it.

For example:

```text
"payment failed"
"transaction was declined"
```

A lexical TF-IDF system may share no important tokens between these phrases. A well-trained embedding model may recognize that they are semantically related. Conversely, a lexical system is likely to be better when the query contains an exact incident code or ticket identifier.

Production systems often use hybrid retrieval:

- lexical search preserves exact-term precision;
- semantic search supplies meaning-based matches;
- a reranker uses richer features to improve ordering.

The right design depends on data, latency, relevance requirements, and failure costs. The current project establishes the lexical baseline so later semantic improvements can be compared against it.

## 14. Educational Implementation Versus Production Implementation

| Educational implementation | Production alternatives |
| --- | --- |
| manually implemented tokenizer | mature preprocessing libraries and versioned pipelines |
| manually implemented TF-IDF | optimized feature-extraction libraries |
| Python-list similarity | optimized numerical kernels and sparse matrices |
| NumPy similarity | batched, compiled, and hardware-optimized operations |
| brute-force ranking | FAISS or other vector indexes |
| local in-memory storage | PostgreSQL with pgvector, Qdrant, Pinecone, Weaviate, or similar systems |
| hand-written tests and examples | monitoring, labeled datasets, evaluation pipelines, and regression dashboards |

The educational implementation deliberately favors transparency over throughput. Manual algorithms make the data flow and formulas inspectable. They are not a claim that a production service should reimplement every mature algorithm.

Understanding the manual version remains valuable for debugging, architecture decisions, failure-mode analysis, library evaluation, performance reasoning, and avoiding black-box usage. When a library returns an unexpected score, an engineer who understands tokenization, feature ordering, norms, and ranking can investigate the right boundary quickly.

## 15. Rails Developer to AI Engineer Terminology Map

These are useful analogies, not exact equivalents:

| Rails concept | AI Engineering concept | Connection |
| --- | --- | --- |
| Rails model/database record | domain document or retrieval item | A stored business object becomes an item represented for search or inference. |
| ActiveRecord scope | filtered retrieval query | Both constrain a larger collection before returning results. |
| background job | offline indexing or ingestion job | Both perform work away from the request path. |
| service object | retrieval/model orchestration service | Both coordinate domain operations behind a focused interface. |
| serializer | structured model output | Both turn internal results into a stable external shape. |
| database index | vector index | Both accelerate lookup over a collection, though their data structures differ. |
| request params | model/query input | Both carry user or application input into a processing boundary. |
| RSpec tests | evaluation and regression tests | Both protect behavior; AI evaluation additionally measures graded relevance or quality. |
| logs/metrics | model and retrieval observability | Both reveal health, latency, errors, and behavior in production. |
| feature flag | model/prompt/version rollout control | Both allow controlled release and comparison of changing behavior. |

The analogy should not hide important differences. A database index usually supports exact or relational access patterns; a vector index supports approximate or similarity-based access. A unit test usually has a crisp expected value; retrieval evaluation often needs relevance judgments and aggregate metrics.

## 16. Current Project Status

The implemented modules are:

- `classifier.py`: deterministic category, priority, and sentiment classification using keyword rules.
- `tfidf.py`: standard-library normalized TF-IDF vectors, document frequency, smoothed IDF, cosine similarity, and ranking helpers.
- `retrieval.py`: cached historical TF-IDF ticket retrieval with `RetrievedTicket` results.
- `evaluation.py`: `precision_at_k()` and `recall_at_k()` for retrieval quality.
- `numpy_basics.py`: NumPy float arrays, matrices, dot products, norms, and cosine similarity.
- `numpy_retrieval.py`: matrix-based ticket retrieval using vectorized cosine similarity and stable ranking.

The latest repository test run contains 60 tests, with all 60 passing:

```text
60 passed in 0.48s
```

That status is a snapshot of the latest documented run. Future changes should rerun the suite before updating this claim.

## 17. What Comes Next: Embeddings

The next stage is embedding-based retrieval. An embedding model converts text into a dense vector whose dimensions encode learned patterns rather than one explicit vocabulary term per coordinate.

The future flow will include:

1. choose a local or hosted embedding model;
2. convert historical ticket text into document embeddings;
3. convert a new query into a query embedding using the same model;
4. verify embedding dimensions and normalization behavior;
5. compare query and document embeddings for semantic retrieval;
6. evaluate embedding retrieval against the TF-IDF baseline;
7. decide whether lexical, semantic, or hybrid retrieval is best for the support domain.

Local embedding models can reduce network dependency and give more control over data handling, but they introduce model packaging, memory, CPU/GPU, versioning, and quality tradeoffs. The query and documents must be embedded consistently. The vector dimension is part of the index contract.

Embeddings matter for RAG because they can retrieve relevant context even when the query and source use different wording. They do not remove the need for evaluation, exact-match signals, permissions, metadata filtering, or prompt and answer validation.

```text
Manual lexical retrieval
    |
    v
NumPy vectorized retrieval
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

## 18. Glossary

- **Corpus:** The collection of documents used for indexing or analysis.
- **Document:** One text item in a corpus, such as a support ticket.
- **Token:** A unit extracted from text for representation, often a word-like piece.
- **Vocabulary:** The mapping from known tokens to feature positions.
- **Feature:** One measurable input coordinate used by an algorithm.
- **Feature space:** The shared coordinate system in which documents become vectors.
- **Vector:** An ordered one-dimensional collection of numeric features.
- **Matrix:** A two-dimensional rectangular collection of numbers.
- **Sparse vector:** A vector with mostly zero values.
- **Dense vector:** A vector with many non-zero values, commonly an embedding.
- **TF:** Term frequency within one document.
- **DF:** Document frequency: the number of documents containing a term.
- **IDF:** Inverse document frequency, which emphasizes terms uncommon in the corpus.
- **TF-IDF:** A representation combining normalized term frequency with IDF.
- **Embedding:** A learned dense vector representation of an item such as text.
- **Lexical search:** Retrieval based primarily on matching and weighting terms.
- **Semantic search:** Retrieval based on learned representations of meaning.
- **Cosine similarity:** Similarity based on the angle between two vectors.
- **Dot product:** The sum of pairwise coordinate products.
- **Norm:** The magnitude or length of a vector.
- **Indexing:** Preparing and storing representations so queries can search efficiently.
- **Query-time transformation:** Converting a new query using the already-fitted representation.
- **Top-k:** The first `k` ranked retrieval results.
- **Precision@k:** The fraction of the first `k` results that are relevant.
- **Recall@k:** The fraction of all known relevant items found in the first `k` results.
- **Retrieval:** Selecting relevant items from a larger corpus.
- **Reranking:** Reordering retrieved candidates with a later or richer scoring step.
- **Vector database:** A system designed to store and search vector representations.
- **RAG:** Retrieval-augmented generation: supplying retrieved context to a generative model before it produces an answer.

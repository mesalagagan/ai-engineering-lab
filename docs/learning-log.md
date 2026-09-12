# AI Engineering Lab Learning Log

This document records what was built, why it was built that way, and which production Python concepts each milestone introduces.

## Project Setup

### Goal

Create a small, dependable Python laboratory for learning production AI engineering incrementally.

### Decisions

- Use Python 3.13+ and `uv` for the environment and dependency workflow.
- Use a `src/` layout so imports exercise the same package boundaries used by installable applications.
- Keep runtime dependencies empty until a learning stage needs one.
- Use pytest, Ruff, and strict mypy as the baseline quality gates.
- Keep the application package small and avoid introducing a web framework before the domain behavior is clear.

### Initial Structure

```text
src/ai_engineering_lab/  # importable application code
tests/                   # automated tests
pyproject.toml           # metadata and tool configuration
```

## Feature 1: Deterministic Support Ticket Classifier

### Goal

Classify support ticket text into a category, priority, and sentiment without machine learning or an LLM.

### Implementation

- `classifier.py` owns the domain model and deterministic keyword rules.
- `ClassificationResult` is an immutable, slotted dataclass.
- `Category`, `Priority`, and `Sentiment` are `StrEnum` values, which keep outputs constrained and JSON-friendly.
- `cli.py` owns command-line parsing and serialization.
- `__main__.py` provides the `python -m ai_engineering_lab` entry point.
- `test_classifier.py` verifies the requested authentication example, a billing case, and default behavior.

### Production Python Concepts

- Pure functions are easy to test because the same input produces the same result.
- Type hints make the domain contract explicit and are checked with strict mypy.
- A frozen dataclass prevents callers from mutating a classification after it is created.
- Separate domain logic from I/O so a future API or worker can reuse the classifier without importing CLI concerns.
- Small rule functions make precedence visible: authentication rules run before broader account rules.

### Validation

```text
pytest: 4 passed
ruff format --check: passed
ruff check: passed
mypy src tests: passed
CLI example: authentication / high / negative
```

### Deliberate Limitations

The classifier uses substring rules and is intentionally not production-complete. It does not yet handle configuration-driven rules, explain its decisions, or measure classification quality against labeled data. Those are useful later lessons, but adding them now would obscure the first domain boundary.

## Feature 2: Numerical Text Representation

### Goal

Represent sentences as small numerical vectors using a bag-of-words model, without introducing NumPy or another runtime dependency.

### Implementation

- `vectorizer.py` owns the `BagOfWordsVectorizer` and uses only Python's standard library.
- Tokenization is intentionally simple: `casefold()` normalizes case, then `split()` separates tokens on whitespace.
- `fit()` builds a vocabulary in first-seen order, assigning each unique token an integer index.
- `transform()` creates one count vector per sentence. Each vector position corresponds to the vocabulary token at that index.
- Tokens not present in the fitted vocabulary are ignored, so new text has the same fixed vector shape.

### Vocabulary, Tokens, and Vectors

For the sentences `"red apple"` and `"blue apple red"`, the vocabulary is `{ "red": 0, "apple": 1, "blue": 2 }`. The vocabulary maps each token to its vector position. The resulting vectors are `[1, 1, 0]` and `[1, 1, 1]`: each number is the count of that token in the sentence. This preserves which words occur and how often, but not word order or meaning.

### Deliberate Limitations

Whitespace tokenization keeps punctuation attached, so `"hello,"` and `"hello"` are different tokens. This keeps the lesson focused on the relationship between text and vectors; punctuation cleanup, weighting such as TF-IDF, and sparse representations can be introduced later.

## Feature 3: Normalized TF-IDF and Similarity

### Goal

Weight document terms by both their frequency in a document and their usefulness across the fitted collection, then compare vectors with cosine similarity.

### Implementation

- `tfidf.py` owns the standard-library-only `TfidfVectorizer`.
- Tokenization lowercases text and extracts word-like tokens with a small regular expression.
- `fit()` builds a stable vocabulary, counts document frequency, and calculates smoothed IDF with `log((1 + n_documents) / (1 + document_frequency)) + 1`.
- `transform()` calculates normalized term frequency as `count / total_tokens`, then multiplies it by IDF. Repeated terms therefore receive proportionally larger weights.
- Unknown tokens are ignored as features, empty documents return zero vectors, and `cosine_similarity()` safely handles zero vectors.
- `most_similar()` ranks document indexes by descending cosine similarity.

### Production Python Concepts

- Fit-time statistics and transform-time inference are separate, which prevents query data from changing the vocabulary or IDF values.
- The implementation uses explicit lists and dictionaries to keep the algorithm inspectable before introducing numerical libraries.
- Small similarity helpers can be reused by later retrieval abstractions without coupling them to ticket metadata.

### Deliberate Limitations

This is a dense educational implementation. It does not yet use sparse vectors, sublinear term frequency, configurable normalization, stemming, or a third-party numerical library. Those tradeoffs are intentionally deferred until the underlying algorithm is familiar.

## Feature 4: Reusable Ticket Retrieval

### Goal

Turn the TF-IDF and cosine-similarity primitives into a reusable search component for historical support tickets.

### Implementation

- `retrieval.py` defines the immutable `RetrievedTicket` result dataclass and the `TicketRetriever` abstraction.
- `TicketRetriever` stores ticket IDs and original text, fits one TF-IDF vectorizer during initialization, and caches historical ticket vectors.
- `search()` transforms each query with the existing fitted vectorizer, ranks cached vectors, and returns the original ticket metadata with similarity scores.
- Search validates `top_k`, supports empty and unknown queries safely, and never refits for an individual query.

### Example

Given historical tickets such as `"password reset failed"` and `"invoice payment failed"`, a query like `"password reset problem"` is transformed into the same feature space and compared against the cached ticket vectors. The result preserves IDs such as `AUTH-1` so a caller can link similarity results back to the source record.

## Feature 5: Retrieval Evaluation

### Goal

Measure whether the retrieval results contain the historical tickets known to be relevant.

### Implementation

- `evaluation.py` provides standard-library-only `precision_at_k()` and `recall_at_k()` functions.
- Precision measures relevant results among the first `k` results; recall measures relevant tickets found among all known relevant tickets.
- Both metrics validate positive `k`. Recall rejects an empty relevant-ticket set because its denominator would be undefined.
- The test suite includes an end-to-end ticket retrieval example that compares returned IDs with known authentication-ticket IDs.

### Deliberate Limitations

Evaluation currently accepts manually supplied relevant IDs. A future lesson can introduce labeled datasets, aggregate metrics across queries, and compare retrieval configurations without embedding labels in the retriever itself.

## Feature 6: NumPy Fundamentals

### Goal

Learn how Python sequences map to NumPy arrays and how common vector operations are expressed with numerical-computing primitives.

### Implementation

- `numpy_basics.py` provides small, typed helpers for creating float vectors and matrices.
- `vector_from_values()` converts an iterable of values into a one-dimensional floating-point array.
- `dot_product()`, `vector_norm()`, and `cosine_similarity_numpy()` demonstrate NumPy dot products, L2 norms, and cosine similarity.
- `vectors_to_matrix()` converts equally sized collections into a two-dimensional float matrix.
- Vector operations validate dimensionality and lengths; cosine similarity returns `0.0` when either vector has zero norm.
- NumPy is now a runtime dependency, while the earlier classifier, TF-IDF, retrieval, and evaluation lessons remain unchanged.

### Production Python Concepts

- Arrays make numerical shape and dtype explicit, which catches incompatible inputs before computation.
- Vectorized library operations express mathematical intent directly and provide a path toward larger-scale numerical workloads.
- Focused tests use `pytest.approx()` for floating-point results and cover both valid shapes and expected validation errors.

### Deliberate Limitations

This module is intentionally a thin educational wrapper around NumPy. It does not yet replace the pure-Python TF-IDF implementation or introduce sparse matrices, batching, or model libraries.

## Feature 7: NumPy Ticket Retrieval

### Goal

Apply NumPy matrix operations to the existing TF-IDF ticket retriever and compare vectorized numerical retrieval with the loop-based implementation.

### Implementation

- `numpy_retrieval.py` fits one existing `TfidfVectorizer`, transforms historical tickets once, and stores the results as a matrix with one ticket per row.
- Search converts the query to an array, calculates all ticket/query dot products with matrix multiplication, and computes row norms with `np.linalg.norm`.
- Broadcasting combines each stored-ticket norm with the query norm. `np.divide(..., where=...)` assigns zero similarity safely whenever either norm is zero.
- `np.argsort` ranks all scores without a Python loop over individual tickets; only the final metadata conversion creates `RetrievedTicket` objects.
- Tests compare ordinary NumPy retrieval results with the existing `TicketRetriever` and cover empty histories, zero-vector queries, shape compatibility, and `top_k` behavior.

### Production Python Concepts

- A two-dimensional matrix gives every ticket the same feature space and lets one array operation process the full corpus.
- Vectorized operations reduce Python interpreter overhead and make the numerical work explicit, while metadata remains handled at the application boundary.
- Stable sorting keeps tie behavior predictable and aligned with the existing retriever.

### Deliberate Limitations

The implementation still uses the educational dense TF-IDF representation. Large production corpora would generally need sparse matrices, batch processing, and an indexed nearest-neighbor strategy.

## Feature 8: Dense Embeddings and Semantic Retrieval

### Goal

Move from lexical word-overlap retrieval toward learned dense representations that can connect related meanings even when the exact words differ.

### Implementation

- `embeddings.py` defines `TextEmbedder`, an application wrapper around `sentence-transformers`.
- The wrapper loads `all-MiniLM-L6-v2` once, exposes the model name and discovered embedding dimension, and supports single or batch encoding into plain Python lists.
- `semantic_retrieval.py` defines `SemanticTicketRetriever`, which accepts an injected embedder, embeds the historical corpus once, and stores one dense embedding row per ticket.
- Search embeds a query once and calculates all cosine similarities with NumPy matrix multiplication, row-wise norms, broadcasting, guarded division, and stable sorting.
- Tests use a deterministic fake embedder for most cases, avoiding repeated model downloads while verifying batch indexing, metadata preservation, dimensions, zero vectors, and tie behavior.

### Production Python Concepts

- Embedding models are feature encoders: the same model must encode both indexed documents and incoming queries.
- Dense vectors distribute meaning across dimensions rather than assigning one coordinate to one vocabulary term.
- Injecting the embedder makes the retriever testable and allows model selection to remain outside the retrieval algorithm.

### Deliberate Limitations

The current retriever uses brute-force dense matrix comparison and an in-memory index. Production systems may use model-specific batching, access-control filters, vector databases, approximate nearest-neighbor indexes, and hybrid lexical-plus-semantic retrieval.

## Delivery Workflow

For each milestone:

1. Create a dated feature branch.
2. Update this log and the README when the project surface changes.
3. Implement the smallest useful slice with tests.
4. Run pytest, Ruff, and mypy.
5. Commit the completed slice and push the branch to GitHub.

GitHub can require reviews, checks, or status approvals through repository settings. The local workflow can consistently validate and push code, but it must not attempt to bypass those repository controls.

## Quality Gate: GitHub Pull Requests

The repository now runs the same local quality commands in `.github/workflows/python-quality.yml` for pull requests targeting `main` or `develop`:

- `uv run pytest`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy src tests`

The workflow uses `uv sync --locked`, so CI honors the committed lock file. To make this equivalent to a required coding standard, configure `Tests, Ruff, and mypy` as a required status check in the branch protection rules for `main` and `develop`.

## Next Milestones

- Add classification explanations while preserving the current domain contract.
- Move rules into explicit, testable configuration.
- Add a labeled retrieval dataset and aggregate metrics across multiple queries.
- Compare retrieval behavior with sparse representations or use NumPy to accelerate a later numerical implementation.
- Introduce an HTTP boundary only after the core domain behavior is stable.
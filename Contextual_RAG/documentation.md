# ContextualRAG Documentation

## Overview

ContextualRAG is an advanced Retrieval-Augmented Generation system that enhances traditional RAG approaches by combining keyword-based and semantic-based retrieval techniques. Unlike simple embedding-based systems, ContextualRAG extracts and leverages keywords with their surrounding context to provide more relevant and precise search results.

## Key Features

- **Hybrid Retrieval Strategy**: Combines traditional keyword-based retrieval with modern embedding-based semantic search
- **Contextual Keyword Processing**: Extracts keywords and preserves their surrounding context for more nuanced understanding
- **Multiple Specialized Databases**: Maintains separate databases for keywords, chunks, and keyword-chunks for optimized retrieval
- **Weighted Search Results**: Allows customizing the balance between keyword, semantic, and hybrid approaches
- **Complete Document Management**: Processes, chunks, indexes, and retrieves from PDF documents
- **Persistence Support**: Save and load functionality to preserve processed data

## Installation

To use ContextualRAG, you'll need to install the following dependencies:
```
bash
pip install transformers torch numpy langchain langchain-community faiss-cpu nltk scikit-learn
```

## Getting Started

### Basic Usage

```
python 
from contextual_rag import ContextualRAG

# Initialize the RAG system
rag = ContextualRAG()

# Process documents from a folder containing PDFs
chunks, metadata = rag.upload_files("path/to/pdfs/")

# Build the databases
rag.make_db(chunks, metadata)

# Query the system
results = rag.process_query("What is the impact of machine learning on materials science?")

# Display results
for result in results:
    print(f"Score: {result['score']}")
    print(f"Document: {result['doc_id']}")
    print(f"Text: {result['chunk_text'][:200]}...")
    print("---")

# Save for later use
rag.save("path/to/save/contextual_rag")

```
### Loading a Saved System
```
from contextual_rag import ContextualRAG

# Initialize an empty system
rag = ContextualRAG()

# Load a previously saved system
rag.load("path/to/saved/contextual_rag")

# Continue using as normal
results = rag.process_query("What are emerging trends in nanomaterials?")

```

## Configuration Options

When initializing a ContextualRAG instance, you can customize several parameters:
```
python 
rag = ContextualRAG( embedding_model_name=
# Emphasize keyword-based retrieval (traditional search)
results = rag.process_query(
    "nanomaterials applications",
    keyword_weight=0.6,       # Higher weight for keyword matching
    chunk_weight=0.2,         # Lower weight for semantic matching
    keyword_chunk_weight=0.2  # Lower weight for hybrid approach
)
```
### Document Processing
- **upload_files(path_folder: str)**: Loads and processes PDF files from a folder
    - Returns: - processed chunks and their metadata `Tuple[List[str], List[ChunkMetadata]]`

- **make_db(chunks: List[str], chunk_metadata: List[ChunkMetadata])**: Creates the vector databases
    - This is a critical step that must be performed after uploading files and before querying

### Querying
- **process_query(query: str, top_k: int = 10, keyword_weight: float = 0.3, chunk_weight: float = 0.3, keyword_chunk_weight: float = 0.4)**: Retrieves the most relevant chunks for a query
    - `query`: The search query
    - : Number of results to return `top_k`
    - , , : Relative importance of each retrieval strategy (must sum to 1.0) `keyword_weight``chunk_weight``keyword_chunk_weight`
    - Returns: List of dictionaries with search results and metadata

- **extract_query_terms(query: str)**: Extracts key terms from a query
    - Returns: List of important terms from the query

### Utility Methods
- **extract_keywords_from_text(text: str, top_n: int = 10)**: Extracts the top keywords from text
    - Returns: List of the most important keywords

- **get_all_unique_keywords()**: Gets statistics on all keywords in the database
    - Returns: Dictionary with keyword statistics

### Persistence
- **save(path: str)**: Saves the system to disk
- **load(path: str)**: Loads a previously saved system

## Advanced Usage
### Customizing Retrieval Weights
The system allows you to adjust the balance between different retrieval strategies:
```
# Emphasize keyword-based retrieval (traditional search)
results = rag.process_query(
    "nanomaterials applications",
    keyword_weight=0.6,       # Higher weight for keyword matching
    chunk_weight=0.2,         # Lower weight for semantic matching
    keyword_chunk_weight=0.2  # Lower weight for hybrid approach
)

# Emphasize semantic retrieval
results = rag.process_query(
    "materials with high thermal conductivity",
    keyword_weight=0.1,        # Lower weight for keyword matching
    chunk_weight=0.7,          # Higher weight for semantic matching
    keyword_chunk_weight=0.2   # Lower weight for hybrid approach
)
```
### Extracting and Analyzing Keywords
```
# Extract keywords from a text
keywords = rag.extract_keywords_from_text(
    "The development of graphene-based nanomaterials has revolutionized energy storage technologies."
)

# Get statistics on all keywords in the database
keyword_stats = rag.get_all_unique_keywords()
```
## Comparison with Traditional RAG
Unlike naive RAG systems that rely solely on semantic embedding similarity, ContextualRAG:
1. Extracts domain-specific keywords using NLP techniques
2. Preserves the context around each keyword occurrence
3. Maintains multiple specialized vector databases for different retrieval strategies
4. Combines results using configurable weights for optimal relevance

This approach is particularly effective for technical and specialized domains where keyword context matters significantly.
## Performance Considerations
- The system performs best with domain-specific content where keyword extraction is meaningful
- Initial processing of documents can be resource-intensive, especially for large document collections
- Consider hardware requirements for embedding models, particularly when using GPU acceleration
- For very large document collections, consider batch processing and incremental database building

## Limitations
- Requires significant memory for storing multiple vector databases
- Initial processing time can be substantial for large document sets
- Keyword extraction quality depends on the characteristics of the document collection
- Performance depends on the quality of the underlying embedding model



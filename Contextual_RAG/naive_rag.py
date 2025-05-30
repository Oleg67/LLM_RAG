#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive RAG System for PDF Document Processing

This module implements a simple Retrieval-Augmented Generation (RAG) system
that uses basic embedding-based retrieval without contextual enhancements.
It provides a baseline for comparison with more sophisticated approaches.
"""

import os
import pickle
import numpy as np
import faiss
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkMetadata(NamedTuple):
    """Metadata for a text chunk."""
    doc_id: str
    page_num: int
    chunk_index: int


class NaiveRAG:
    """
    A simple Retrieval-Augmented Generation system that uses basic vector search
    without contextual enhancements or keyword extraction.
    """

    def __init__(
            self,
            embedding_model_name: str = "BAAI/bge-base-en-v1.5",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            device: Optional[str] = None
    ):
        """
        Initialize the NaiveRAG system.

        Args:
            embedding_model_name: Hugging Face model for embeddings
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        # Storage for vector DB
        self.vector_db = None

        # Data storage
        self.chunk_texts = []
        self.chunk_metadata = []
        self.chunk_embeddings = None

    def upload_files(self, path_folder: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """
        Load and split PDF files from a folder into chunks using LangChain.

        Args:
            path_folder: Path to folder containing PDF files

        Returns:
            Tuple containing:
            - List of text chunks
            - List of metadata for each chunk
        """
        # Create a directory loader for PDF files
        loader = DirectoryLoader(
            path_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        print(f"Loading PDF files from {path_folder}")
        # Load documents
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Extract text and metadata from Document objects
        text_chunks = []
        chunk_metadata = []

        chunk_index_by_doc = {}  # Track chunk indices per document

        for chunk in chunks:
            text_chunks.append(chunk.page_content)

            # Extract document ID from source or filename
            doc_id = os.path.basename(chunk.metadata.get('source', 'unknown'))
            page_num = chunk.metadata.get('page', 0)

            # Increment chunk index for this document
            if doc_id not in chunk_index_by_doc:
                chunk_index_by_doc[doc_id] = 0
            else:
                chunk_index_by_doc[doc_id] += 1

            # Store metadata
            chunk_metadata.append(ChunkMetadata(
                doc_id=doc_id,
                page_num=page_num,
                chunk_index=chunk_index_by_doc[doc_id]
            ))

        print(f"Created {len(text_chunks)} text chunks from {len(chunk_index_by_doc)} documents")
        return text_chunks, chunk_metadata

    def get_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.

        Args:
            chunks: List of text chunks

        Returns:
            Array of chunk embeddings
        """
        chunk_embeddings = []

        for chunk in tqdm(chunks, desc="Generating embeddings"):
            # Tokenize the text
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get embeddings from the model
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)

                # Use pooler output (if available) or mean of last hidden state
                if hasattr(outputs, 'pooler_output'):
                    embedding = outputs.pooler_output[0].cpu().numpy()
                else:
                    embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()

                chunk_embeddings.append(embedding)

        # Convert to numpy array
        embeddings_array = np.array(chunk_embeddings).astype('float32')

        # Normalize embeddings
        normalized_embeddings = normalize(embeddings_array)

        print(f"Generated embeddings for {len(normalized_embeddings)} chunks")
        return normalized_embeddings

    def make_db(self, chunks: List[str], chunk_metadata: List[ChunkMetadata]) -> None:
        """
        Create vector database for chunk embeddings.

        Args:
            chunks: List of text chunks
            chunk_metadata: List of metadata for each chunk
        """
        # Store chunks and metadata
        self.chunk_texts = chunks
        self.chunk_metadata = chunk_metadata

        # Generate embeddings
        chunk_embeddings = self.get_embeddings(chunks)

        # Create FAISS index
        dimension = chunk_embeddings.shape[1]
        self.vector_db = faiss.IndexFlatIP(dimension)
        self.vector_db.add(chunk_embeddings)

        print(f"Created vector database with {len(chunks)} entries")

        # Store the embeddings for later use
        self.chunk_embeddings = chunk_embeddings

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a query.

        Args:
            query: Query text

        Returns:
            Normalized query embedding
        """
        # Tokenize the query
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

            # Use pooler output or mean of last hidden state
            if hasattr(outputs, 'pooler_output'):
                query_embedding = outputs.pooler_output[0].cpu().numpy()
            else:
                query_embedding = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()

        # Normalize the embedding
        return normalize(query_embedding.reshape(1, -1)).astype('float32')

    def process_query(self,
                      query: str,
                      top_k: int = 5,
                      filter_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process a query using simple vector similarity search.

        Args:
            query: Text query
            top_k: Number of results to return
            filter_doc_ids: Optional list of document IDs to filter results

        Returns:
            List of dicts with relevant chunks and metadata
        """
        # Generate query embedding
        query_embedding = self.get_query_embedding(query)

        # Check if vector DB exists
        if self.vector_db is None:
            raise ValueError("Vector database has not been created. Call make_db first.")

        # Get more results to account for filtering
        search_k = top_k * 3 if filter_doc_ids else top_k

        # Search in vector database
        scores, indices = self.vector_db.search(query_embedding, search_k)

        # Build results
        results = []

        for i, idx in enumerate(indices[0]):
            if idx >= len(self.chunk_texts):
                continue

            # Get chunk metadata
            metadata = self.chunk_metadata[idx]

            # Skip if outside filter
            if filter_doc_ids and metadata.doc_id not in filter_doc_ids:
                continue

            # Add to results
            results.append({
                "chunk_text": self.chunk_texts[idx],
                "score": float(scores[0][i]),
                "chunk_id": int(idx),
                "doc_id": metadata.doc_id,
                "page_num": metadata.page_num,
                "chunk_index": metadata.chunk_index
            })

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        """
        Save the RAG system to disk.

        Args:
            path: Path to save the system
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Prepare data to save
        data = {
            "chunk_texts": self.chunk_texts,
            "chunk_metadata": self.chunk_metadata,
            "chunk_embeddings": self.chunk_embeddings if hasattr(self, 'chunk_embeddings') else None
        }

        # Save vector index
        if hasattr(self, 'vector_db') and self.vector_db is not None:
            faiss.write_index(self.vector_db, f"{path}.index")

        # Save other data
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"Saved RAG system to {path}")

    def load(self, path: str) -> None:
        """
        Load the RAG system from disk.

        Args:
            path: Path to load the system from
        """
        # Load vector index
        if os.path.exists(f"{path}.index"):
            self.vector_db = faiss.read_index(f"{path}.index")
        else:
            print(f"Warning: Index file {path}.index not found")

        # Load other data
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunk_texts = data["chunk_texts"]
                self.chunk_metadata = data.get("chunk_metadata", [])
                if "chunk_embeddings" in data and data["chunk_embeddings"] is not None:
                    self.chunk_embeddings = data["chunk_embeddings"]

            print(f"Loaded RAG system from {path}")
        else:
            print(f"Warning: Data file {path}.pkl not found")


def main():
    """Example usage of the NaiveRAG system."""
    # Initialize the RAG system
    rag = NaiveRAG()

    # Process PDF files
    chunks, chunk_metadata = rag.upload_files("path/to/pdfs")

    # Create vector database
    rag.make_db(chunks, chunk_metadata)

    # Save the system
    rag.save("path/to/save/naive_rag_system")

    # Example query
    query = "What is machine learning?"
    results = rag.process_query(query, top_k=3)

    print("\nQuery Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1} (Score: {result['score']:.4f}):")
        print(f"Document ID: {result['doc_id']}")
        print(f"Page: {result['page_num']}")
        print(f"Text: {result['chunk_text'][:200]}...")


if __name__ == "__main__":
    main()
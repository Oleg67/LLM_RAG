#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Contextual RAG System for PDF Document Processing

This module implements a Retrieval-Augmented Generation (RAG) system
that uses contextual information from documents to improve search quality.
It combines keyword-based and semantic-based retrieval approaches.
"""
#
#  contextual_rag.py
#  

import os
import re
import string
import pickle
import hashlib
from typing import List, Tuple, Dict, Any, Set, NamedTuple, Optional
from collections import defaultdict

import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.preprocessing import normalize
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


class ChunkMetadata(NamedTuple):
    """Metadata for a text chunk."""
    doc_id: str
    page_num: int
    chunk_index: int
    
    
class KeywordContext(NamedTuple):
    """Context information for a keyword."""
    context_hash: str
    context_text: str
    chunk_id: int
    doc_id: str


class ContextualRAG:
    """
    A contextual Retrieval-Augmented Generation system that combines
    keyword-based and semantic-based retrieval for improved search quality.
    """
    
    def __init__(
        self, 
        embedding_model_name: str = "BAAI/bge-base-en-v1.5", 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        context_window: int = 50,
        device: Optional[str] = None
    ):
        """
        Initialize the ContextualRAG system.
        
        Args:
            embedding_model_name: Hugging Face model for embeddings
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks
            context_window: Number of tokens around keywords to use for context
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window
        
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
        
        # Ensure NLTK resources are available
        self._ensure_nltk_resources()
        
        # Storage for vector DBs
        self.keyword_db = None
        self.chunk_db = None
        self.vector_db = None  # For backward compatibility
        
        # Data storage
        self.chunk_embeddings = None
        self.chunk_texts = []
        self.chunk_metadata = []
        self.keyword_contexts = {}
        
        # Mapping dictionaries
        self.keyword_context_to_id = {}  # Maps (keyword, context_hash, doc_id) to vector ID
        self.id_to_keyword_context = {}  # Maps vector ID to (keyword, context_hash, doc_id)
        self.id_to_chunk = {}            # Maps vector ID to chunk ID

    def _ensure_nltk_resources(self):
        """Ensure required NLTK resources are available."""
        resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                               else f'taggers/{resource}' if resource == 'averaged_perceptron_tagger'
                               else f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

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
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Convert to lowercase
        return text.lower()
    
    def get_keywords(self, chunks: List[str]) -> List[List[str]]:
        """
        Extract topic-related keywords from each chunk using TF-IDF and POS tagging.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of lists of keywords in each chunk
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # POS tags that are likely to contain domain-specific terms
        relevant_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VB', 'VBN', 'VBG'}
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        
        # Extract candidate keywords using POS tagging
        def extract_candidate_keywords(text):
            # Clean the text
            clean = self._clean_text(text)
            
            # Tokenize and tag parts of speech
            words = word_tokenize(clean)
            tagged = pos_tag(words)
            
            # Filter by relevant POS tags and remove stopwords
            candidates = []
            for word, tag in tagged:
                if (tag[:2] in relevant_pos_tags and 
                    word.lower() not in stop_words and 
                    word not in punctuation and
                    len(word) > 2):  # Filter out very short words
                    candidates.append(word.lower())
            
            return candidates
        
        # Extract candidates from each chunk for TF-IDF
        chunk_candidates = [' '.join(extract_candidate_keywords(chunk)) for chunk in chunks]
        
        # Apply TF-IDF to identify important keywords
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 2)  # Allow for unigrams and bigrams
        )
        
        # Make sure we have enough content for TF-IDF
        if len(chunks) < 3:
            print("Warning: TF-IDF works best with more documents. Using alternative method.")
            all_keywords = []
            for chunk in tqdm(chunks, desc="Extracting keywords"):
                candidates = extract_candidate_keywords(chunk)
                # Count frequency of each candidate
                freq = defaultdict(int)
                for word in candidates:
                    freq[word] += 1
                # Select top keywords by frequency
                sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
                chunk_keywords = [word for word, count in sorted_keywords]
                all_keywords.append(chunk_keywords)
            
            # Count total unique keywords across all chunks
            unique_keywords = set()
            for chunk_keywords in all_keywords:
                unique_keywords.update(chunk_keywords)
            
            print(f"Extracted {len(unique_keywords)} unique keywords across {len(chunks)} chunks")
            return all_keywords
        
        # Compute TF-IDF for candidate keywords
        try:
            tfidf_matrix = vectorizer.fit_transform(chunk_candidates)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top keywords for each chunk
            all_keywords = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="Extracting keywords")):
                if i >= tfidf_matrix.shape[0]:
                    all_keywords.append([])
                    continue
                    
                # Get TF-IDF scores for this chunk
                tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
                sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                
                # Select top keywords by TF-IDF score
                top_n = 15  # Number of keywords per chunk
                chunk_keywords = [word for word, score in sorted_keywords[:top_n] if score > 0]
                
                # Ensure we have enough keywords (fallback to POS-based extraction)
                if len(chunk_keywords) < 5:
                    candidates = extract_candidate_keywords(chunk)
                    freq = defaultdict(int)
                    for word in candidates:
                        freq[word] += 1
                    sorted_by_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    chunk_keywords = list(set(chunk_keywords + [word for word, count in sorted_by_freq]))
                
                all_keywords.append(chunk_keywords)
            
        except ValueError as e:
            # Fallback method if TF-IDF fails
            print(f"TF-IDF extraction failed: {e}. Using alternative method.")
            all_keywords = []
            for chunk in tqdm(chunks, desc="Extracting keywords (fallback)"):
                candidates = extract_candidate_keywords(chunk)
                # Count frequency of each candidate
                freq = defaultdict(int)
                for word in candidates:
                    freq[word] += 1
                # Select top keywords by frequency
                sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
                chunk_keywords = [word for word, count in sorted_keywords]
                all_keywords.append(chunk_keywords)
        
        # Count total unique keywords across all chunks
        unique_keywords = set()
        for chunk_keywords in all_keywords:
            unique_keywords.update(chunk_keywords)
        
        print(f"Extracted {len(unique_keywords)} unique keywords across {len(chunks)} chunks")
        return all_keywords
    
    def get_chunk_emb(self, chunks: List[str]) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Generate token-level embeddings for each chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Tuple containing:
            - List of tensors with token embeddings for each chunk
            - List of lists with token IDs for each chunk
        """
        chunk_embeddings = []
        token_ids_list = []
        
        for chunk in tqdm(chunks, desc="Generating chunk embeddings"):
            # Tokenize the text
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=512).to(self.device)
            
            # Store token IDs for this chunk
            token_ids = inputs['input_ids'][0].cpu().tolist()
            token_ids_list.append(token_ids)
            
            # Get token embeddings from the full model output
            with torch.no_grad():
                outputs = self.embedding_model(**inputs, output_hidden_states=True)
                
                # Get the embeddings from the last hidden state
                # This contains embeddings for all tokens (including special tokens)
                token_embeddings = outputs.hidden_states[-1][0].cpu()
                
            chunk_embeddings.append(token_embeddings)
        
        print(f"Generated token embeddings for {len(chunk_embeddings)} chunks")
        return chunk_embeddings, token_ids_list
    
    def get_full_chunk_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for entire chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Array of chunk embeddings
        """
        chunk_embeddings = []
        
        for chunk in tqdm(chunks, desc="Generating full chunk embeddings"):
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
    
    def get_context_window(self, 
                          token_ids: List[int], 
                          token_start_idx: int, 
                          token_end_idx: int) -> List[int]:
        """
        Get the token IDs for the context window around a keyword.
        
        Args:
            token_ids: List of token IDs for the chunk
            token_start_idx: Start index of the keyword tokens
            token_end_idx: End index of the keyword tokens
            
        Returns:
            List of token IDs representing the context window
        """
        start = max(0, token_start_idx - self.context_window // 2)
        end = min(len(token_ids), token_end_idx + self.context_window // 2)
        
        return token_ids[start:end]
    
    def hash_context(self, context_tokens: List[int]) -> str:
        """
        Create a hash of context tokens to identify unique contexts.
        
        Args:
            context_tokens: List of token IDs representing the context
            
        Returns:
            Hash string of the context
        """
        return hashlib.md5(str(context_tokens).encode()).hexdigest()
    
    def get_keyword_emb(self, 
                       keywords: List[List[str]], 
                       chunk_embeddings: List[torch.Tensor],
                       token_ids_list: List[List[int]],
                       chunks: List[str],
                       chunk_metadata: List[ChunkMetadata]) -> Dict[Tuple[str, str, str], np.ndarray]:
        """
        Generate embeddings for each keyword by extracting the corresponding token embeddings,
        while preserving context and document ID.
        
        Args:
            keywords: List of lists of keywords in each chunk
            chunk_embeddings: List of tensors with token embeddings for each chunk
            token_ids_list: List of lists with token IDs for each chunk
            chunks: List of text chunks for context extraction
            chunk_metadata: List of metadata for each chunk
            
        Returns:
            Dictionary mapping (keyword, context_hash, doc_id) tuples to embeddings
        """
        keyword_context_doc_embeddings = {}
        keyword_contexts = {}  # Maps keyword to a list of KeywordContext objects
        
        # Process each chunk and its keywords
        for chunk_id, chunk_keywords in enumerate(tqdm(keywords, desc="Generating keyword embeddings")):
            if chunk_id >= len(chunk_embeddings):
                continue
                
            # Get the token embeddings and token IDs for this chunk
            token_embs = chunk_embeddings[chunk_id]
            token_ids = token_ids_list[chunk_id]
            chunk_text = chunks[chunk_id]
            doc_id = chunk_metadata[chunk_id].doc_id
            
            # Process each keyword in this chunk
            for keyword in chunk_keywords:
                # Tokenize the keyword to get its token IDs
                keyword_tokens = self.tokenizer.tokenize(keyword)
                keyword_ids = self.tokenizer.convert_tokens_to_ids(keyword_tokens)
                
                # Find all occurrences of the keyword in the chunk
                if len(keyword_ids) == 1:
                    # Single-token keyword case
                    matches = [i for i, token_id in enumerate(token_ids) if token_id == keyword_ids[0]]
                    
                    for idx in matches:
                        if idx < len(token_embs):
                            # Get context window around the keyword
                            context_tokens = self.get_context_window(token_ids, idx, idx + 1)
                            context_hash = self.hash_context(context_tokens)
                            
                            # Get the context text
                            context_text = self.tokenizer.decode(context_tokens)
                            
                            # Store the embedding with context and doc_id
                            keyword_context_doc_embeddings[(keyword, context_hash, doc_id)] = token_embs[idx].numpy()
                            
                            # Store the context information
                            if keyword not in keyword_contexts:
                                keyword_contexts[keyword] = []
                            keyword_contexts[keyword].append(KeywordContext(
                                context_hash=context_hash,
                                context_text=context_text,
                                chunk_id=chunk_id,
                                doc_id=doc_id
                            ))
                            
                else:
                    # Multi-token keyword case
                    for i in range(len(token_ids) - len(keyword_ids) + 1):
                        if all(token_ids[i+j] == keyword_ids[j] for j in range(len(keyword_ids))):
                            # Found the sequence
                            token_range = range(i, i + len(keyword_ids))
                            
                            if max(token_range) < len(token_embs):
                                # Get context window around the keyword
                                context_tokens = self.get_context_window(token_ids, i, i + len(keyword_ids))
                                context_hash = self.hash_context(context_tokens)
                                
                                # Get the context text
                                context_text = self.tokenizer.decode(context_tokens)
                                
                                # Store the embedding with context and doc_id (average of token embeddings)
                                keyword_context_doc_embeddings[(keyword, context_hash, doc_id)] = token_embs[token_range].mean(dim=0).numpy()
                                
                                # Store the context information
                                if keyword not in keyword_contexts:
                                    keyword_contexts[keyword] = []
                                keyword_contexts[keyword].append(KeywordContext(
                                    context_hash=context_hash,
                                    context_text=context_text,
                                    chunk_id=chunk_id,
                                    doc_id=doc_id
                                ))
        
        # Print statistics
        total_contexts = sum(len(contexts) for contexts in keyword_contexts.values())
        unique_keywords = len(keyword_contexts)
        
        print(f"Generated embeddings for {len(keyword_context_doc_embeddings)} keyword-context-document pairs")
        print(f"Average {total_contexts/unique_keywords:.2f} contexts per keyword")
        
        # Store the context information for later use
        self.keyword_contexts = keyword_contexts
        
        return keyword_context_doc_embeddings
    
    def make_db(self, 
               keyword_context_doc_embeddings: Dict[Tuple[str, str, str], np.ndarray], 
               chunks: List[str],
               chunk_metadata: List[ChunkMetadata]) -> None:
        """
        Create vector databases for both keyword and chunk embeddings.
        
        Args:
            keyword_context_doc_embeddings: Dictionary mapping (keyword, context_hash, doc_id) to embeddings
            chunks: List of text chunks
            chunk_metadata: List of metadata for each chunk
        """
        # Store chunks and metadata
        self.chunk_texts = chunks
        self.chunk_metadata = chunk_metadata
        
        # Create mappings for keyword embeddings
        self.keyword_context_to_id = {}
        self.id_to_keyword_context = {}
        self.id_to_chunk = {}
        
        # Create list of keyword-context-doc pairs
        keyword_context_doc_list = list(keyword_context_doc_embeddings.keys())
        keyword_embeddings = []
        
        # Build the keyword vector DB indices
        for i, (keyword, context_hash, doc_id) in enumerate(keyword_context_doc_list):
            self.keyword_context_to_id[(keyword, context_hash, doc_id)] = i
            self.id_to_keyword_context[i] = (keyword, context_hash, doc_id)
            
            # Find the chunk ID for this context
            for context in self.keyword_contexts.get(keyword, []):
                if context.context_hash == context_hash and context.doc_id == doc_id:
                    self.id_to_chunk[i] = context.chunk_id
                    break
            
            keyword_embeddings.append(keyword_context_doc_embeddings[(keyword, context_hash, doc_id)])
        
        # Convert to numpy array
        keyword_emb_array = np.array(keyword_embeddings).astype('float32')
        
        # Normalize embeddings
        normalized_keyword_emb = normalize(keyword_emb_array)
        
        # Create FAISS index for keyword embeddings
        dimension = normalized_keyword_emb.shape[1]
        self.keyword_db = faiss.IndexFlatIP(dimension)
        self.keyword_db.add(normalized_keyword_emb)
        
        print(f"Created keyword vector database with {len(keyword_context_doc_list)} entries")
        
        # Now create embeddings for entire chunks
        chunk_embeddings = self.get_full_chunk_embeddings(chunks)
        
        # Store mapping from chunk index to metadata
        self.chunk_id_to_metadata = {i: metadata for i, metadata in enumerate(chunk_metadata)}
        
        # Create FAISS index for chunk embeddings
        self.chunk_db = faiss.IndexFlatIP(dimension)
        self.chunk_db.add(chunk_embeddings)
        
        print(f"Created chunk vector database with {len(chunks)} entries")
        
        # Store the chunk embeddings for later use
        self.chunk_embeddings = chunk_embeddings
        
        # Set the main vector_db to keyword_db for backward compatibility
        self.vector_db = self.keyword_db
    
    def get_all_unique_keywords(self) -> Dict[str, Dict[str, int]]:
        """
        Get all unique keywords and their document distribution.
        
        Returns:
            Dictionary with keywords as keys and dictionaries mapping document IDs to occurrence count
        """
        if not hasattr(self, 'keyword_contexts') or not self.keyword_contexts:
            raise ValueError("No keywords available. Process some documents first.")
        
        keyword_stats = {}
        
        # Iterate through all the keywords and their contexts
        for keyword, contexts in self.keyword_contexts.items():
            # Initialize the statistics for this keyword
            keyword_stats[keyword] = {}
            
            # Count occurrences per document
            for context in contexts:
                doc_id = context.doc_id
                if doc_id in keyword_stats[keyword]:
                    keyword_stats[keyword][doc_id] += 1
                else:
                    keyword_stats[keyword][doc_id] = 1
        
        # Print a summary
        print(f"Total unique keywords: {len(keyword_stats)}")
        
        # Find keywords with highest document coverage
        keyword_doc_counts = [(k, len(docs)) for k, docs in keyword_stats.items()]
        keyword_doc_counts.sort(key=lambda x: x[1], reverse=True)
        
        if keyword_doc_counts:
            print("\nTop 10 keywords by document coverage:")
            for keyword, doc_count in keyword_doc_counts[:10]:
                total_occurrences = sum(keyword_stats[keyword].values())
                print(f"'{keyword}': {doc_count} documents, {total_occurrences} total occurrences")
        
        return keyword_stats
    
    def extract_query_terms(self, query: str) -> List[str]:
        """
        Extract important terms from a query using POS tagging.
        
        Args:
            query: Query text
            
        Returns:
            List of important terms
        """
        # POS tags that are likely to contain domain-specific terms
        relevant_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VB', 'VBN', 'VBG'}
        # Similar approach as in the get_keywords method
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        
        # Clean and normalize query
        clean_query = self._clean_text(query)
        query_tokens = word_tokenize(clean_query)
        tagged = pos_tag(query_tokens)
        
        # Extract unigrams
        query_keywords = []
        for word, tag in tagged:
            if (tag[:2] in relevant_pos_tags and 
                word.lower() not in stop_words and 
                word not in punctuation and
                len(word) > 2):
                query_keywords.append(word.lower())
        
        # Extract bigrams
        # Get bigrams as well (they might be important for context)
        query_bigrams = []
        for i in range(len(query_tokens) - 1):
            bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
            if (query_tokens[i] not in stop_words and 
                query_tokens[i+1] not in stop_words and
                len(query_tokens[i]) > 2 and 
                len(query_tokens[i+1]) > 2):
                query_bigrams.append(bigram.lower())
        
        # Combine unigrams and bigrams
        query_terms = list(set(query_keywords + query_bigrams))
        return query_terms
    
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
                     filter_doc_ids: Optional[List[str]] = None,
                     keyword_weight: float = 0.7,
                     chunk_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Process a query using both keyword and chunk embeddings.
        
        Args:
            query: Text query
            top_k: Number of results to return
            filter_doc_ids: Optional list of document IDs to filter results
            keyword_weight: Weight for keyword similarity (0-1)
            chunk_weight: Weight for chunk similarity (0-1)
            
        Returns:
            List of dicts with relevant chunks and metadata
        """
        # Normalize weights
        total_weight = keyword_weight + chunk_weight
        keyword_weight = keyword_weight / total_weight
        chunk_weight = chunk_weight / total_weight
        
        # Generate query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Check if vector DBs exist
        if self.keyword_db is None or self.chunk_db is None:
            raise ValueError("Vector databases have not been created. Call make_db first.")
        
        # Get more results to account for filtering and combination
        search_k = top_k * 5
        
        # Search in both keyword and chunk databases
        keyword_scores, keyword_indices = self.keyword_db.search(query_embedding, search_k)
        chunk_scores, chunk_indices = self.chunk_db.search(query_embedding, search_k)
        
        # Extract important terms from query for highlighting in results
        query_terms = self.extract_query_terms(query)
        print(f"Query terms: {query_terms}")
        
        # Combine results from both keyword and chunk searches
        combined_scores = defaultdict(float)
        
        # Process keyword search results
        for i, idx in enumerate(keyword_indices[0]):
            if idx in self.id_to_chunk:
                chunk_id = self.id_to_chunk[idx]
                
                # Skip if outside filter
                if (filter_doc_ids and 
                    chunk_id < len(self.chunk_metadata) and 
                    self.chunk_metadata[chunk_id].doc_id not in filter_doc_ids):
                    continue
                
                # Add normalized score for this chunk
                combined_scores[chunk_id] += keyword_weight * float(keyword_scores[0][i])
    
        # Process chunk search results
        for i, chunk_id in enumerate(chunk_indices[0]):
            # Skip if outside filter
            if (filter_doc_ids and 
                chunk_id < len(self.chunk_metadata) and 
                self.chunk_metadata[chunk_id].doc_id not in filter_doc_ids):
                continue
        
            # Add normalized score for this chunk
            combined_scores[chunk_id] += chunk_weight * float(chunk_scores[0][i])
    
        # Sort chunks by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
        # Build final results
        results = []
    
        for chunk_id, score in sorted_chunks[:top_k]:
            if chunk_id >= len(self.chunk_texts):
                continue
            
            # Get chunk metadata
            metadata = self.chunk_metadata[chunk_id]
            chunk_text = self.chunk_texts[chunk_id]
            
            # Find keywords associated with this chunk
            chunk_keywords = set()
            for key, val in self.id_to_chunk.items():
                if val == chunk_id:
                    keyword, _, _ = self.id_to_keyword_context[key]
                    chunk_keywords.add(keyword)
            
            # Find relevant terms in this chunk
            chunk_lower = chunk_text.lower()
            
            # Check for query term presence in the chunk
            relevant_terms = [term for term in query_terms 
                            if term in chunk_lower]

            # Calculate term frequency in the chunk for ranking
            term_frequencies = {}
            for term in relevant_terms:
                # Count approximate instances (might catch substrings too)
                term_frequencies[term] = chunk_lower.count(term)
            
            # Sort terms by frequency
            sorted_terms = sorted(
                term_frequencies.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            most_relevant_terms = [term for term, freq in sorted_terms if freq > 0]
            
            results.append({
                "chunk_text": chunk_text,
                "keywords": list(chunk_keywords),
                "score": score,
                "chunk_id": int(chunk_id),
                "doc_id": metadata.doc_id,
                "page_num": metadata.page_num,
                "chunk_index": metadata.chunk_index,
                "relevant_terms": most_relevant_terms
            })
        
        return results

    def get_all_unique_keywords(self) -> Dict[str, Dict[str, int]]:
        """
        Get all unique keywords and their document distribution.
        
        Returns:
            Dictionary with keywords as keys and dictionaries mapping document IDs to occurrence count
        """
        if not hasattr(self, 'keyword_contexts') or not self.keyword_contexts:
            raise ValueError("No keywords available. Process some documents first.")
        
        keyword_stats = {}
        
        # Iterate through all the keywords and their contexts
        for keyword, contexts in self.keyword_contexts.items():
            # Initialize the statistics for this keyword
            keyword_stats[keyword] = {}
            
            # Count occurrences per document
            for context in contexts:
                doc_id = context.doc_id
                if doc_id in keyword_stats[keyword]:
                    keyword_stats[keyword][doc_id] += 1
                else:
                    keyword_stats[keyword][doc_id] = 1
        
        # Print a summary
        print(f"Total unique keywords: {len(keyword_stats)}")
        
        # Find keywords with highest document coverage
        keyword_doc_counts = [(k, len(docs)) for k, docs in keyword_stats.items()]
        keyword_doc_counts.sort(key=lambda x: x[1], reverse=True)
        
        if keyword_doc_counts:
            print("\nTop 10 keywords by document coverage:")
            for keyword, doc_count in keyword_doc_counts[:10]:
                total_occurrences = sum(keyword_stats[keyword].values())
                print(f"'{keyword}': {doc_count} documents, {total_occurrences} total occurrences")
        
        return keyword_stats
    
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
            "keyword_context_to_id": self.keyword_context_to_id,
            "id_to_keyword_context": self.id_to_keyword_context,
            "id_to_chunk": self.id_to_chunk,
            "keyword_contexts": self.keyword_contexts,
            "chunk_embeddings": self.chunk_embeddings if hasattr(self, 'chunk_embeddings') else None
        }
        
        # Save keyword index
        if hasattr(self, 'keyword_db') and self.keyword_db is not None:
            faiss.write_index(self.keyword_db, f"{path}_keyword.index")
        
        # Save chunk index
        if hasattr(self, 'chunk_db') and self.chunk_db is not None:
            faiss.write_index(self.chunk_db, f"{path}_chunk.index")
        
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
        # Load keyword index
        if os.path.exists(f"{path}_keyword.index"):
            self.keyword_db = faiss.read_index(f"{path}_keyword.index")
            self.vector_db = self.keyword_db  # For backward compatibility
        else:
            print(f"Warning: Keyword index file {path}_keyword.index not found")
        
        # Load chunk index
        if os.path.exists(f"{path}_chunk.index"):
            self.chunk_db = faiss.read_index(f"{path}_chunk.index")
        else:
            print(f"Warning: Chunk index file {path}_chunk.index not found")
        
        # Load other data
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunk_texts = data["chunk_texts"]
                self.chunk_metadata = data.get("chunk_metadata", [])
                self.keyword_context_to_id = data["keyword_context_to_id"]
                self.id_to_keyword_context = data["id_to_keyword_context"]
                self.id_to_chunk = data["id_to_chunk"]
                self.keyword_contexts = data["keyword_contexts"]
                if "chunk_embeddings" in data and data["chunk_embeddings"] is not None:
                    self.chunk_embeddings = data["chunk_embeddings"]
                    
            print(f"Loaded RAG system from {path}")
        else:
            print(f"Warning: Data file {path}.pkl not found")


def main():
    """Example usage of the ContextualRAG system."""
    # Initialize the RAG system
    rag = ContextualRAG()
    
    # Process PDF files
    chunks, chunk_metadata = rag.upload_files("path/to/pdfs")
    
    # Extract keywords from each chunk
    keywords = rag.get_keywords(chunks)
    
    # Generate chunk embeddings and token IDs
    chunk_embeddings, token_ids_list = rag.get_chunk_emb(chunks)
    
    # Generate keyword embeddings with context and document ID
    keyword_context_doc_embeddings = rag.get_keyword_emb(
        keywords, chunk_embeddings, token_ids_list, chunks, chunk_metadata
    )
    
    # Create vector databases for both keywords and chunks
    rag.make_db(keyword_context_doc_embeddings, chunks, chunk_metadata)
    
    # Save the system
    rag.save("path/to/save/rag_system")
    
    # Example query - search across all documents with hybrid approach
    query = "What is machine learning?"
    results = rag.process_query(
        query, 
        top_k=3, 
        keyword_weight=0.7,  # Emphasize keyword matches
        chunk_weight=0.3     # But also consider whole-chunk similarity
    )
    
    print("\nHybrid Query Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
        print(f"Document ID: {result['doc_id']}")
        print(f"Page: {result['page_num']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"Relevant terms: {', '.join(result['relevant_terms'])}")
        print(f"Text: {result['chunk_text'][:200]}...")


if __name__ == "__main__":
    main()

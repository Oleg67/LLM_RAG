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
from sklearn.feature_extraction.text import TfidfVectorizer
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
        context_window: int = 200,
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

        # Define a vectorizer for keyword extraction
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            max_df=0.75,
            min_df=1,
            stop_words='english',
            use_idf=True,
            ngram_range=(1, 2)
        )

        # Ensure NLTK resources are available
        self._ensure_nltk_resources()
        
        # Storage for vector DBs
        self.keyword_db = None
        self.chunk_db = None
        self.keyword_chunk_db = None
        self.vector_db = None  # For backward compatibility
        
        # Data storage
        self.chunk_embeddings = []
        self.keyword_chunk_embeddings = []
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

    def upload_files(self,
                     path_folder: str
    ) -> Tuple[List[str], List[ChunkMetadata]]:
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
    
    def _clean_text(self,
                    text: str
    ) -> str:
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
    
    def get_keywords(self,
                     chunks: List[str]
    ) -> List[List[str]]:
        """
        Extract topic-related keywords from each chunk using TF-IDF and POS tagging.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of lists of keywords in each chunk
        """
        
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
            tfidf_matrix = self.vectorizer.fit_transform(chunk_candidates)
            feature_names = self.vectorizer.get_feature_names_out()
            
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

    def get_query_keyword_emb(self,
                              query_terms: List[str],
                              query: str = None
                              ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for query keywords, optionally with query context.

        Args:
            query_terms: List of keywords extracted from the query
            query: Optional full query text for context

        Returns:
            Dictionary mapping keywords to their embeddings
        """
        # Create a dictionary to store keyword embeddings
        keyword_embeddings = {}

        # Generate embeddings for each query term
        for keyword in query_terms:
            # If we have query context, create a combined text with the keyword and context
            if query:
                # Create a contextualized representation
                text_to_embed = f"{keyword}: {query}"
            else:
                # Just use the keyword by itself
                text_to_embed = keyword

            try:
                # Step 1: Get token-level embeddings using _get_embedding
                token_embeddings_np, _ = self._get_embedding(text_to_embed, normalize=True)

                # Convert numpy array to torch tensor for pooling
                token_embeddings = torch.tensor(token_embeddings_np)

                # Step 2: Pool token embeddings to get a single vector using _pooler_embeddings
                # We put the tensor in a list since _pooler_embeddings expects a list of tensors
                pooled_embedding = self._pooler_embeddings([token_embeddings], pooling_strategy="mean")[0]

                # Convert to numpy array and ensure correct shape and type
                embedding = pooled_embedding.cpu().numpy().reshape(1, -1).astype('float32')

                # Store the embedding in the dictionary
                keyword_embeddings[keyword] = embedding

            except Exception as e:
                print(f"Error generating keyword embedding for '{keyword}': {e}")
                # Determine appropriate dimension for fallback
                if hasattr(self, 'embedding_dim'):
                    dim = self.embedding_dim
                elif hasattr(self, 'chunk_embeddings') and self.chunk_embeddings is not None:
                    dim = self.chunk_embeddings.shape[1]
                else:
                    # Default dimension if we can't determine it
                    dim = 768

                # Add a zero vector as fallback
                keyword_embeddings[keyword] = np.zeros((1, dim), dtype='float32')

        return keyword_embeddings

    def get_chunk_emb(self,
                      chunks: List[str]
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
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
            # Use the _get_embedding method to get token embeddings and token IDs
            token_embeddings_np, token_ids_np = self._get_embedding(chunk)

            # Convert numpy arrays to the appropriate format
            token_embeddings = torch.tensor(token_embeddings_np)
            token_ids = token_ids_np.tolist()

            # Store the results
            chunk_embeddings.append(token_embeddings)
            token_ids_list.append(token_ids)

        print(f"Generated token embeddings for {len(chunk_embeddings)} chunks")
        print(f"Generated token IDs for {len(token_ids_list)} chunks")
        print(f"Token embedding shape: {chunk_embeddings[0].shape} and Type: {type(chunk_embeddings[0])}")
        return chunk_embeddings, token_ids_list

    def get_full_chunk_embeddings(self,
                                  token_embeddings_list: List[torch.Tensor],
                                  pooling_strategy: str = "mean"
                                  ) -> List[torch.Tensor]:

        """
        Pool token-level embeddings into single embeddings for each chunk.

        Args:
            token_embeddings_list: List of 2D tensors with token embeddings for each chunk
            pooling_strategy: Strategy to use for pooling. Options:
                             - "mean": Use mean of all token embeddings
                             - "pooler": Use the model's pooler_output (if available)
                             - "cls": Use the CLS token embedding

        Returns:
            List of 1D tensors with a single embedding vector for each chunk
        """

        # Pool token-level embeddings into chunk-level embeddings
        chunk_embeddings = self._pooler_embeddings(token_embeddings_list, pooling_strategy)

        print(f"Generated {len(chunk_embeddings)} chunk embeddings using {pooling_strategy} pooling")

        return chunk_embeddings
    
    def get_context_window(self, 
                          token_ids: List[int], 
                          token_start_idx: int, 
                          token_end_idx: int
    ) -> List[int]:
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
    
    def hash_context(self,
                     context_tokens: List[int]
    ) -> str:
        """
        Create a hash of context tokens to identify unique contexts.
        
        Args:
            context_tokens: List of token IDs representing the context
            
        Returns:
            Hash string of the context
        """
        return hashlib.md5(str(context_tokens).encode()).hexdigest()

    def _find_keyword_occurrences(self,
                                  keyword,
                                  token_ids,
                                  token_embs
    ):
        """
        Find all occurrences of a keyword in token IDs and extract their embeddings.

        Args:
            keyword: The keyword to find
            token_ids: List of token IDs for the chunk
            token_embs: Tensor of token embeddings for the chunk

        Returns:
            List of tuples containing (start_idx, end_idx, embedding)
        """
        # Tokenize the keyword to get its token IDs
        keyword_tokens = self.tokenizer.tokenize(keyword)
        keyword_ids = self.tokenizer.convert_tokens_to_ids(keyword_tokens)

        occurrences = []

        if len(keyword_ids) == 1:
            # Single-token keyword case
            matches = [i for i, token_id in enumerate(token_ids) if token_id == keyword_ids[0]]

            for idx in matches:
                if idx < len(token_embs):
                    # Get the embedding for this occurrence
                    emb = token_embs[idx].cpu().numpy()
                    occurrences.append((idx, idx + 1, emb))

        else:
            # Multi-token keyword case
            for i in range(len(token_ids) - len(keyword_ids) + 1):
                if all(token_ids[i + j] == keyword_ids[j] for j in range(len(keyword_ids))):
                    # Found the sequence
                    token_range = range(i, i + len(keyword_ids))

                    if max(token_range) < len(token_embs):
                        # Get the embedding for this occurrence (average of token embeddings)
                        emb = token_embs[token_range].mean(dim=0).cpu().numpy()
                        occurrences.append((i, i + len(keyword_ids), emb))

        return occurrences

    def _process_keyword_in_chunk(self,
                                  keyword,
                                  token_ids,
                                  token_embs,
                                  chunk_id,
                                  doc_id
    ):
        """
        Process a keyword in a chunk, finding all occurrences and extracting context and embeddings.

        Args:
            keyword: The keyword to process
            token_ids: List of token IDs for the chunk
            token_embs: Tensor of token embeddings for the chunk
            chunk_id: ID of the current chunk
            doc_id: Document ID

        Returns:
            Tuple of (keyword_context_doc_embeddings, keyword_contexts, keyword_embeddings)
                keyword_context_doc_embeddings: Dict mapping (keyword, context_hash, doc_id) to embeddings
                keyword_contexts: List of KeywordContext objects for this keyword
                keyword_embeddings: List of embeddings for this keyword in this chunk
        """
        keyword_context_doc_embeddings = {}
        keyword_contexts = []
        keyword_embeddings = []

        # Find all occurrences of the keyword
        occurrences = self._find_keyword_occurrences(keyword, token_ids, token_embs)

        for start_idx, end_idx, emb in occurrences:
            # Get context window around the keyword
            context_tokens = self.get_context_window(token_ids, start_idx, end_idx)
            context_hash = self.hash_context(context_tokens)

            # Get the context text
            context_text = self.tokenizer.decode(context_tokens)

            # Store the embedding with context and doc_id
            keyword_context_doc_embeddings[(keyword, context_hash, doc_id)] = emb

            # Store the context information
            keyword_contexts.append(KeywordContext(
                context_hash=context_hash,
                context_text=context_text,
                chunk_id=chunk_id,
                doc_id=doc_id
            ))

            # Store the embedding for chunk-level aggregation
            keyword_embeddings.append(emb)

        return keyword_context_doc_embeddings, keyword_contexts, keyword_embeddings

    def _get_embedding(self,
                       text,
                       normalize: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an embedding vector for the provided text using the model's tokenizer and embedding model.

        This function tokenizes the input text, processes it through the embedding model,
        and extracts a numerical vector representation. It handles both models that return
        pooler_output and those that only provide last_hidden_state.

        Args:
            text (str): The text to generate an embedding for. Can be a sentence, paragraph,
                        or any text content that needs to be converted to a vector representation.
            normalize (bool): Whether to normalize the token embeddings to unit length (default: False)

        Returns:
            numpy.ndarray: A 2-dimensional float numpy array containing the
                                      embedding vectors of each token. The dimensionality depends on the underlying embedding model and number of tokens
            numpy.ndarray: A 1-dimensional int numpy array containing the
                                      tokens indices
        """
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
            return_attention_mask=True
        )

        # Get the token IDs for return
        token_ids = inputs['input_ids'][0].numpy()

        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding using the model
        with torch.no_grad():
            self.embedding_model.eval()
            outputs = self.embedding_model(**inputs)

            # Extract token embeddings from the model output
            if hasattr(outputs, 'last_hidden_state'):
                # Get the embeddings for all tokens
                token_embeddings = outputs.last_hidden_state[0].cpu()
            elif hasattr(outputs, 'hidden_states'):
                # Some models provide hidden states instead
                token_embeddings = outputs.hidden_states[-1][0].cpu()
            else:
                # Fallback - model may have custom output format
                token_embeddings = outputs[0].cpu()

            # Normalize embeddings if requested
            if normalize:
                # Compute L2 norm along the embedding dimension
                norms = torch.norm(token_embeddings, p=2, dim=1, keepdim=True)
                # Normalize embeddings (add small epsilon to avoid division by zero)
                token_embeddings = token_embeddings / (norms + 1e-8)

            # Convert to numpy for return
            token_embeddings = token_embeddings.numpy()
        #print(f"Embeddins shape: {token_embeddings.shape} tokens shape: {token_ids.shape}")
        return token_embeddings, token_ids

    def _pooler_embeddings(self,
                           token_embeddings_list: List[torch.Tensor],
                           pooling_strategy: str = "mean"
    ) -> List[torch.Tensor]:
        """
        Pool token-level embeddings into single embeddings for each chunk.

        This method converts a list of 2D tensors (token-level embeddings) into a list of 1D tensors
        (chunk-level embeddings) by applying a pooling strategy such as mean pooling or using
        the model's pooler output if available.

        Args:
            token_embeddings_list: List of 2D tensors with token embeddings for each chunk
            pooling_strategy: Strategy to use for pooling. Options:
                             - "mean": Use mean of all token embeddings
                             - "pooler": Use the model's pooler_output (if available)
                             - "cls": Use the CLS token embedding

        Returns:
            List of 1D tensors with a single embedding vector for each chunk
        """
        pooled_embeddings = []

        for token_embeddings in token_embeddings_list:
            if pooling_strategy == "mean":
                # Mean pooling: average all token embeddings
                pooled_embedding = torch.mean(token_embeddings, dim=0)

            elif pooling_strategy == "cls":
                # CLS pooling: use the first token (typically the CLS token)
                pooled_embedding = token_embeddings[0]

            elif pooling_strategy == "pooler" and hasattr(self.embedding_model, "pooler"):
                # Try to use the model's pooler if available
                # Note: This would typically need to be done during the original forward pass,
                # so we're including this option but it may require modification based on how
                # the model is being used

                # Re-process the inputs through the model to get the pooler output
                # This is a placeholder implementation - the actual implementation would
                # depend on how the model is structured
                try:
                    # This assumes token_embeddings has a batch dimension
                    pooled_embedding = self.embedding_model.pooler(token_embeddings.unsqueeze(0)).squeeze(0)
                except:
                    # Fall back to mean pooling if pooler fails
                    pooled_embedding = torch.mean(token_embeddings, dim=0)
                    print("Warning: Pooler strategy failed, falling back to mean pooling.")
            else:
                # Default to mean pooling if the strategy is not recognized
                pooled_embedding = torch.mean(token_embeddings, dim=0)

            pooled_embeddings.append(pooled_embedding)

        return pooled_embeddings

    def get_keyword_emb(self,
                        keywords: List[List[str]],
                        chunk_embeddings: List[torch.Tensor],
                        token_ids_list: List[List[int]],
                        chunks: List[str],
                        chunk_metadata: List[ChunkMetadata]
    ) -> Dict[Tuple[str, str, str], np.ndarray]:
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
            doc_id = chunk_metadata[chunk_id].doc_id

            # Process each keyword in this chunk
            for keyword in chunk_keywords:
                # Process the keyword and get its embeddings and contexts
                kw_embs, kw_contexts, _ = self._process_keyword_in_chunk(
                    keyword, token_ids, token_embs, chunk_id, doc_id
                )

                # Update the global dictionaries
                keyword_context_doc_embeddings.update(kw_embs)

                if keyword not in keyword_contexts:
                    keyword_contexts[keyword] = []
                keyword_contexts[keyword].extend(kw_contexts)

        # Print statistics
        total_contexts = sum(len(contexts) for contexts in keyword_contexts.values())
        unique_keywords = len(keyword_contexts)

        print(f"Generated embeddings for {len(keyword_context_doc_embeddings)} keyword-context-document pairs")
        print(
            f"Average {total_contexts / unique_keywords:.2f} contexts per keyword" if unique_keywords > 0 else "No keywords found")

        # Store the context information for later use
        self.keyword_contexts = keyword_contexts

        return keyword_context_doc_embeddings

    def get_keyword_chunk_emb(self,
                              keywords: List[List[str]],
                              chunk_embeddings: List[torch.Tensor],
                              token_ids_list: List[List[int]],
                              chunks: List[str],
                              chunk_metadata: List[ChunkMetadata]
    ) -> List[np.ndarray]:
        """
        Generate chunk embeddings by averaging the embeddings of keywords in each chunk.

        Args:
            keywords: List of lists of keywords in each chunk
            chunk_embeddings: List of tensors with token embeddings for each chunk
            token_ids_list: List of lists with token IDs for each chunk
            chunks: List of text chunks for context extraction
            chunk_metadata: List of metadata for each chunk

        Returns:
            Array of chunk embeddings based on averaged keyword embeddings
        """
        # Initialize result list to store chunk embeddings
        chunk_emb_list = []
        chunk_keyword_counts = {}  # Count of keywords found in each chunk

        # Process each chunk and its keywords
        for chunk_id, chunk_keywords in enumerate(tqdm(keywords, desc="Generating chunk embeddings")):
            if chunk_id >= len(chunk_embeddings):
                # Skip if we don't have embeddings for this chunk
                if chunk_embeddings:
                    chunk_emb_list.append(np.zeros((chunk_embeddings[0].shape[1],)))
                else:
                    chunk_emb_list.append(np.zeros((768,)))  # Default embedding size
                continue

            # Get the token embeddings and token IDs for this chunk
            token_embs = chunk_embeddings[chunk_id]
            token_ids = token_ids_list[chunk_id]
            doc_id = chunk_metadata[chunk_id].doc_id

            # Collect all keyword embeddings for this chunk
            all_keyword_embeddings = []

            # Process each keyword in this chunk
            for keyword in chunk_keywords:
                # Process the keyword and get its embeddings
                _, _, keyword_embeddings = self._process_keyword_in_chunk(
                    keyword, token_ids, token_embs, chunk_id, doc_id
                )

                # Add the embeddings to our collection
                all_keyword_embeddings.extend(keyword_embeddings)

            # Store the count of keywords found
            chunk_keyword_counts[chunk_id] = len(all_keyword_embeddings)

            # Compute the chunk embedding by averaging all keyword embeddings
            if all_keyword_embeddings:
                # If we found keywords, average their embeddings
                chunk_emb = np.mean(all_keyword_embeddings, axis=0)
            else:
                # If no keywords were found, use a zero vector
                chunk_emb = np.zeros((token_embs.shape[1],))

            chunk_emb_list.append(chunk_emb)

        # Print statistics
        num_chunks_with_keywords = sum(1 for count in chunk_keyword_counts.values() if count > 0)
        total_keywords = sum(chunk_keyword_counts.values())

        print(f"Generated embeddings for {len(chunk_emb_list)} chunks")
        print(f"Found keywords in {num_chunks_with_keywords} chunks")
        if num_chunks_with_keywords > 0:
            print(f"Average {total_keywords / num_chunks_with_keywords:.2f} keywords per chunk with keywords")

        return chunk_emb_list

    def make_db(self,
                chunks: List[str],
                chunk_metadata: List[ChunkMetadata]
    ) -> None:
        """
        Create three vector databases:
        1. Keyword embeddings database
        2. Chunk embeddings database
        3. Averaged keyword chunk embeddings database (main DB)

        Args:
            chunks: List of text chunks
            chunk_metadata: List of metadata for each chunk

        Returns:
            None (databases are stored as instance attributes)
        """
        print("Creating vector databases...")

        # Step 1: Extract keywords from chunks
        print("Extracting keywords from chunks...")
        keywords = self.get_keywords(chunks)
        self.keywords = keywords

        # Step 2: Generate token-level embeddings for each chunk
        print("Generating token-level embeddings...")
        chunk_embeddings, token_ids_list = self.get_chunk_emb(chunks)

        # Step 3: Generate full chunk embeddings (for the chunk database)
        print("Generating full chunk embeddings...")
        full_chunk_embeddings = self.get_full_chunk_embeddings(chunk_embeddings)

        # Step 4: Generate keyword embeddings with context
        print("Generating keyword embeddings...")
        keyword_context_doc_embeddings = self.get_keyword_emb(
            keywords, chunk_embeddings, token_ids_list, chunks, chunk_metadata
        )

        # Step 5: Generate keyword-based chunk embeddings (for the main database)
        print("Generating keyword-based chunk embeddings...")
        keyword_chunk_embeddings = self.get_keyword_chunk_emb(
            keywords, chunk_embeddings, token_ids_list, chunks, chunk_metadata
        )

        # Store the embeddings for later use
        self.chunk_embeddings = full_chunk_embeddings
        self.keyword_chunk_embeddings = keyword_chunk_embeddings
        self.chunk_texts = chunks
        self.chunk_metadata = chunk_metadata

        # Create mapping from chunk ID to metadata
        self.chunk_id_to_metadata = {i: meta for i, meta in enumerate(chunk_metadata)}

        # Create the keyword database
        print("Creating keyword embedding database...")
        keyword_db = {}
        self.id_to_keyword_context = {}
        self.keyword_context_to_id = {}

        for i, ((keyword, context_hash, doc_id), embedding) in enumerate(keyword_context_doc_embeddings.items()):
            # Create a unique ID for this keyword-context-doc combination
            context_id = f"kw_{i}"

            # Store the embedding in the database
            keyword_db[context_id] = embedding

            # Create mappings for later lookup
            keyword_context = KeywordContext(
                context_hash=context_hash,
                context_text=self.keyword_contexts.get(keyword, [])[
                    0].context_text if keyword in self.keyword_contexts and self.keyword_contexts[keyword] else "",
                chunk_id=next((ctx.chunk_id for ctx in self.keyword_contexts.get(keyword, []) if
                               ctx.context_hash == context_hash), None),
                doc_id=doc_id
            )

            self.id_to_keyword_context[context_id] = keyword_context
            self.keyword_context_to_id[(keyword, context_hash, doc_id)] = context_id

        # Create the chunk database
        print("Creating chunk embedding database...")
        chunk_db = {}
        self.id_to_chunk = {}

        for i, embedding in enumerate(full_chunk_embeddings):
            # Create a unique ID for this chunk
            chunk_id = f"chunk_{i}"

            # Store the embedding in the database
            chunk_db[chunk_id] = embedding

            # Create a mapping for later lookup
            self.id_to_chunk[chunk_id] = i

        # Create the keyword chunk database (main DB)
        print("Creating keyword-based chunk embedding database...")
        keyword_chunk_db = {}
        self.keyword_chunk_id_to_metadata = {}

        for i, embedding in enumerate(keyword_chunk_embeddings):
            # Create a unique ID for this keyword-based chunk
            keyword_chunk_id = f"kw_chunk_{i}"

            # Store the embedding in the database
            keyword_chunk_db[keyword_chunk_id] = embedding

            # Create a mapping for later lookup
            self.keyword_chunk_id_to_metadata[keyword_chunk_id] = chunk_metadata[i]

        # Store the databases as instance attributes
        self.keyword_db = keyword_db
        self.chunk_db = chunk_db
        self.keyword_chunk_db = keyword_chunk_db

        # Create the vector database for efficient similarity search
        print("Creating vector database for efficient similarity search...")

        # Combine all databases into one for efficient searching
        vector_db = {}
        vector_db.update(keyword_db)
        vector_db.update(chunk_db)
        vector_db.update(keyword_chunk_db)

        # Convert to numpy arrays for vectorization
        ids = list(vector_db.keys())
        vectors = np.array([vector_db[id] for id in ids])

        # Normalize the vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vectors = vectors / norms

        self.vector_db = {
            'ids': ids,
            'vectors': normalized_vectors
        }

        print(f"Vector databases created with {len(keyword_db)} keyword embeddings, "
              f"{len(chunk_db)} chunk embeddings, and {len(keyword_chunk_db)} keyword-chunk embeddings.")
    
    def extract_query_terms(self,
                            query: str
    ) -> List[str]:
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

    def get_query_embedding(self,
                            query: str,
                            pooling_strategy: str="mean"
    ) -> np.ndarray:
        """
        Generate an embedding for a query using the embedding model and pooling strategy.

        Args:
            query: Query text
            pooling_strategy: Strategy to use for pooling. Options:
                             - "mean": Use mean of all token embeddings
                             - "pooler": Use the model's pooler_output (if available)
                             - "cls": Use the CLS token embedding


        Returns:
            Normalized query embedding as a numpy array
        """
        # Step 1: Get token-level embeddings using _get_embedding
        token_embeddings_np, _ = self._get_embedding(query, normalize=True)

        # Convert numpy array to torch tensor for pooling
        token_embeddings = torch.tensor(token_embeddings_np)

        # Step 2: Pool token embeddings to get a single vector using _pooler_embeddings
        # We put the tensor in a list since _pooler_embeddings expects a list of tensors
        pooled_embedding = self._pooler_embeddings([token_embeddings], pooling_strategy)[0]

        # Convert to numpy array and ensure correct shape and type
        query_embedding = pooled_embedding.cpu().numpy().reshape(1, -1).astype('float32')

        return query_embedding

    def process_query(self,
                      query: str,
                      top_k: int = 10,
                      keyword_weight: float = 0.3,
                      chunk_weight: float = 0.3,
                      keyword_chunk_weight: float = 0.4
    ) -> List[Dict]:
        """
        Process a query using hybrid search across three vector databases.
        Uses the saved vectorizer for keyword extraction.

        Args:
            query: The query string
            top_k: Number of results to return
            keyword_weight: Weight for keyword-based search results (0.0 to 1.0)
            chunk_weight: Weight for chunk-based search results (0.0 to 1.0)
            keyword_chunk_weight: Weight for keyword-chunk-based search results (0.0 to 1.0)

        Returns:
            List of dictionaries containing search results with their scores
        """
        # Ensure weights sum to 1.0
        total_weight = keyword_weight + chunk_weight + keyword_chunk_weight
        if total_weight != 1.0:
            # Normalize weights
            keyword_weight /= total_weight
            chunk_weight /= total_weight
            keyword_chunk_weight /= total_weight
            print(f"Normalized weights: keyword={keyword_weight}, "
                  f"chunk={chunk_weight}, keyword_chunk={keyword_chunk_weight}")

        # Calculate number of results to retrieve from each database
        keyword_count = max(1, int(top_k * keyword_weight)) if keyword_weight > 0 else 0
        chunk_count = max(1, int(top_k * chunk_weight)) if chunk_weight > 0 else 0
        keyword_chunk_count = max(1, top_k - keyword_count - chunk_count)

        # Adjust counts if necessary to ensure we get exactly top_k results
        total_count = keyword_count + chunk_count + keyword_chunk_count
        if total_count < top_k:
            # Prioritize keyword_chunk results for any remaining slots
            keyword_chunk_count += (top_k - total_count)
        elif total_count > top_k:
            # Reduce counts if we've exceeded top_k
            excess = total_count - top_k
            if keyword_count > excess:
                keyword_count -= excess
                excess = 0
            else:
                excess -= keyword_count
                keyword_count = 0

            if excess > 0 and chunk_count > excess:
                chunk_count -= excess
                excess = 0
            elif excess > 0:
                excess -= chunk_count
                chunk_count = 0

            if excess > 0:
                keyword_chunk_count -= excess

        print(f"Retrieving: {keyword_count} keyword results, {chunk_count} chunk results, "
              f"{keyword_chunk_count} keyword-chunk results")

        results = []

        # 1. Extract query keywords and generate query embeddings
        query_terms = self.extract_query_terms(query)
        query_embedding = self.get_query_embedding(query)

        # 2. Search keyword database
        if keyword_count > 0 and hasattr(self, 'keyword_db') and self.keyword_db:
            print(f"Searching keyword database with {len(query_terms)} query terms...")

            # Get keyword embeddings for query terms
            query_keyword_embeddings = self.get_query_keyword_emb(query_terms, query)

            # Search each keyword embedding against the keyword database
            keyword_results = []
            for term, embedding in query_keyword_embeddings.items():
                term_results = self.search_keyword_db(embedding, top_k=keyword_count)

                # Add the query term to each result
                for result in term_results:
                    result['query_term'] = term
                    keyword_results.append(result)

            # Sort by score and take top results
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            keyword_results = keyword_results[:keyword_count]

            # Add result type
            for result in keyword_results:
                result['result_type'] = 'keyword'
                result['weight'] = keyword_weight

                # Get context information
                keyword_context = self.id_to_keyword_context.get(result['id'])
                if keyword_context:
                    result['doc_id'] = keyword_context.doc_id
                    result['text'] = keyword_context.context_text
                    result['chunk_id'] = keyword_context.chunk_id
                    result['metadata'] = self.chunk_metadata[result['chunk_id']]

                results.append(result)

        # 3. Search chunk database
        if chunk_count > 0 and hasattr(self, 'chunk_db') and self.chunk_db:
            print("Searching chunk database...")

            # Search query embedding against chunk database
            chunk_results = self.search_chunk_db(query_embedding, top_k=chunk_count)

            # Add result type
            for result in chunk_results:
                result['result_type'] = 'chunk'
                result['weight'] = chunk_weight

                # Get chunk text and metadata
                chunk_idx = self.id_to_chunk.get(result['id'])
                if chunk_idx is not None and chunk_idx < len(self.chunk_texts):
                    result['text'] = self.chunk_texts[chunk_idx]
                    result['metadata'] = self.chunk_metadata[chunk_idx]
                    result['chunk_id'] = chunk_idx
                    result['doc_id'] = result['metadata'].doc_id

                results.append(result)

        # 4. Search keyword-chunk database (main DB)
        if keyword_chunk_count > 0 and hasattr(self, 'keyword_chunk_db') and self.keyword_chunk_db:
            print("Searching keyword-chunk database...")

            # Generate keyword-based query embedding
            query_keyword_chunk_emb = self.get_query_keyword_chunk_emb(query_terms, query)

            # Search query embedding against keyword-chunk database
            keyword_chunk_results = self.search_keyword_chunk_db(query_keyword_chunk_emb, top_k=keyword_chunk_count)

            # Add result type
            for result in keyword_chunk_results:
                result['result_type'] = 'keyword_chunk'
                result['weight'] = keyword_chunk_weight

                # Get metadata and text
                metadata = self.keyword_chunk_id_to_metadata.get(result['id'])
                if metadata:
                    result['metadata'] = metadata
                    result['doc_id'] = metadata.doc_id

                    # Find the corresponding chunk text
                    chunk_idx = next((i for i, meta in enumerate(self.chunk_metadata)
                                      if meta.doc_id == metadata.doc_id and
                                      meta.chunk_index == metadata.chunk_index), None)

                    if chunk_idx is not None and chunk_idx < len(self.chunk_texts):
                        result['text'] = self.chunk_texts[chunk_idx]
                        result['chunk_id'] = chunk_idx

                results.append(result)

        # 5. Combine and normalize scores
        # Scale scores by their weights
        for result in results:
            result['original_score'] = result['score']
            result['score'] = result['score'] * result['weight']

        # Sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Take top_k results
        final_results = results[:top_k]

        # Format the final results
        formatted_results = []
        for result in final_results:
            formatted_result = {
                'id': result['id'],
                'score': result['score'],
                'original_score': result.get('original_score', result['score']),
                'result_type': result['result_type'],
                'text': result.get('text', ''),
                'keyword': None, 
                'doc_id': result.get('doc_id', None),
                'chunk_id': result.get('chunk_id', None),
                'metadata': result.get('metadata', None),
            }

            # Add text content based on result type
            if result['result_type'] == 'keyword':
                formatted_result['keyword'] = result.get('query_term', '')

            formatted_results.append(formatted_result)

        print(f"Retrieved {len(formatted_results)} results")
        return formatted_results


    def extract_keywords_from_text(self,
                                   text: str,
                                   top_n: int = 10
    ) -> List[str]:
            """
            Extract keywords from text using the TF-IDF vectorizer.

            Args:
                text: Input text to extract keywords from
                top_n: Number of top keywords to return

            Returns:
                List of extracted keywords
            """
            # Check if vectorizer is fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                # If not fitted, initialize and fit a new one with default params
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(
                    max_df=0.85, min_df=2, stop_words='english',
                    use_idf=True, ngram_range=(1, 2)
                )
                # Fit on a single document (not ideal, but prevents errors)
                self.vectorizer.fit([text])
                print("Warning: Vectorizer not fitted. Initialized a new one.")

            try:
                # Transform the text using the vectorizer
                feature_names = self.vectorizer.get_feature_names_out()
                tf_idf_vector = self.vectorizer.transform([text])

                # Convert to coordinate format for easier sorting
                coo_matrix = tf_idf_vector.tocoo()

                # Sort by TF-IDF score
                sorted_items = sorted(
                    zip(coo_matrix.col, coo_matrix.data),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Extract the top N keywords
                keywords = [feature_names[idx] for idx, _ in sorted_items[:top_n]]
                return keywords

            except (ValueError, AttributeError) as e:
                print(f"Error using vectorizer for keyword extraction: {e}")
                # Fallback to a simple keyword extraction approach
                import re
                from collections import Counter

                # Simple tokenization and cleaning
                words = re.findall(r'\b\w+\b', text.lower())

                # Filter out very short words and common English stop words
                stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'and', 'or', 'but', 'is', 'are',
                              'was', 'were', 'be', 'been', 'being', 'to', 'for', 'of', 'by', 'with'}
                words = [word for word in words if len(word) > 2 and word not in stop_words]

                # Count and get most frequent
                word_counts = Counter(words)
                keywords = [word for word, _ in word_counts.most_common(top_n)]

                return keywords

    def _sort_coo(self, coo_matrix):
        
        """Sort a sparse matrix by value and return sorted (idx, value) tuples."""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: x[1], reverse=True)

    def search_keyword_db(self, embedding, top_k):
        """
        Search the keyword database for the most similar embeddings.

        Args:
            embedding: Query embedding to search for
            top_k: Number of results to return

        Returns:
            List of dictionaries containing search results with their scores
        """
        if not self.keyword_db:
            print("Warning: Keyword database is empty")
            return []

        results = []

        # Calculate cosine similarity between query and all vectors in the keyword database
        for id, vector in self.keyword_db.items():
            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(embedding)
            vector_norm = np.linalg.norm(vector)

            # Avoid division by zero
            if query_norm == 0 or vector_norm == 0:
                similarity = 0
            else:
                # Calculate cosine similarity
                similarity = np.dot(embedding.flatten(), vector.flatten()) / (query_norm * vector_norm)

            results.append({
                'id': id,
                'score': float(similarity)
            })

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k results
        return results[:top_k]

    def search_chunk_db(self, embedding, top_k):
        """
        Search the chunk database for the most similar embeddings.

        Args:
            embedding: Query embedding to search for
            top_k: Number of results to return

        Returns:
            List of dictionaries containing search results with their scores
        """
        if not self.chunk_db:
            print("Warning: Chunk database is empty")
            return []

        results = []

        # Calculate cosine similarity between query and all vectors in the chunk database
        for id, vector in self.chunk_db.items():
            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(embedding)
            vector_norm = np.linalg.norm(vector)

            # Avoid division by zero
            if query_norm == 0 or vector_norm == 0:
                similarity = 0
            else:
                # Calculate cosine similarity
                similarity = np.dot(embedding.flatten(), vector.flatten()) / (query_norm * vector_norm)

            results.append({
                'id': id,
                'score': float(similarity)
            })

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k results
        return results[:top_k]

    def search_keyword_chunk_db(self, embedding, top_k):
        """
        Search the keyword-chunk database for the most similar embeddings.

        Args:
            embedding: Query embedding to search for
            top_k: Number of results to return

        Returns:
            List of dictionaries containing search results with their scores
        """
        if not self.keyword_chunk_db:
            print("Warning: Keyword-chunk database is empty")
            return []

        results = []

        # Calculate cosine similarity between query and all vectors in the keyword-chunk database
        for id, vector in self.keyword_chunk_db.items():
            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(embedding)
            vector_norm = np.linalg.norm(vector)

            # Avoid division by zero
            if query_norm == 0 or vector_norm == 0:
                similarity = 0
            else:
                # Calculate cosine similarity
                similarity = np.dot(embedding.flatten(), vector.flatten()) / (query_norm * vector_norm)

            results.append({
                'id': id,
                'score': float(similarity)
            })

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k results
        return results[:top_k]

    def get_query_keyword_chunk_emb(self,
                                    query_terms: List[str],
                                    query: str = None,
                                    pooling_strategy: str = "mean"
    ) -> np.ndarray:
        """
        Generate a pooled embedding for query keywords, optionally with query context.

        This method first generates embeddings for each keyword in the query and then
        pools them together using the specified pooling strategy.

        Args:
            query_terms: List of keywords extracted from the query
            query: Optional full query text for context
            pooling_strategy: Strategy for pooling keyword embeddings ("mean", "max", etc.)

        Returns:
            A single pooled embedding vector representing the entire query
        """
        # Early return for empty query terms
        if not query_terms:
            # Determine appropriate dimension for empty case
            if hasattr(self, 'embedding_dim'):
                dim = self.embedding_dim
            elif hasattr(self, 'chunk_embeddings') and self.chunk_embeddings is not None:
                dim = self.chunk_embeddings.shape[1]
            else:
                # Default dimension if we can't determine it
                dim = 768
            return np.zeros((1, dim), dtype='float32')

        # Step 1: Get embeddings for each keyword
        keyword_embeddings_dict = self.get_query_keyword_emb(query_terms, query)

        # Step 2: Convert dictionary of embeddings to list of tensors for pooling
        keyword_embeddings_list = []
        for keyword, embedding in keyword_embeddings_dict.items():
            # Convert numpy array to torch tensor
            embedding_tensor = torch.tensor(embedding.squeeze())
            keyword_embeddings_list.append(embedding_tensor)

        # Step 3: Directly calculate mean of keyword embeddings instead of using _pooler_embeddings
        if len(keyword_embeddings_list) > 0:
            # Convert list of embeddings to a tensor if needed
            if isinstance(keyword_embeddings_list[0], torch.Tensor):
                # Stack tensors along dimension 0
                stacked_embeddings = torch.stack(keyword_embeddings_list, dim=0)
                # Calculate mean along dimension 0
                pooled_embedding = torch.mean(stacked_embeddings, dim=0)
            else:
                # If not tensors, convert to tensor first
                keyword_embeddings_tensor = torch.tensor(keyword_embeddings_list)
                pooled_embedding = torch.mean(keyword_embeddings_tensor, dim=0)

            # Step 4: Convert the pooled embedding to numpy and reshape
            pooled_embedding_np = pooled_embedding.cpu().numpy().reshape(1, -1).astype('float32')
            return pooled_embedding_np
        else:
            # Handle the case when there are no embeddings
            # Return an appropriate default value or raise an error
            raise ValueError("No keyword embeddings available to pool")
            return np.zeros((1, dim), dtype='float32')

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

    def save(self, path: str):
        """
        Save the contextual RAG model to disk.

        This method saves all necessary components:
        - Vector databases (keyword, chunk, and keyword chunk)
        - Mappings and metadata
        - Tokenizer and embedding model configurations
        - Cached embeddings and texts
        - Vectorizer for TF-IDF
        - Embedding model name

        Args:
            path: Path to save the model

        Returns:
            None
        """
        import os
        import pickle
        import json
        from pathlib import Path
        import datetime

        print(f"Saving model to {path}...")

        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Create dictionaries to save
        data = {
            # Vector databases
            'keyword_db': self.keyword_db,
            'chunk_db': self.chunk_db,
            'keyword_chunk_db': self.keyword_chunk_db,
            'vector_db': self.vector_db,

            # Mappings
            'id_to_keyword_context': self.id_to_keyword_context,
            'keyword_context_to_id': self.keyword_context_to_id,
            'id_to_chunk': self.id_to_chunk,
            'chunk_id_to_metadata': self.chunk_id_to_metadata,
            'keyword_chunk_id_to_metadata': self.keyword_chunk_id_to_metadata,

            # Cached data
            'chunk_texts': self.chunk_texts,
            'chunk_metadata': self.chunk_metadata,
            'chunk_embeddings': self.chunk_embeddings,
            'keyword_chunk_embeddings': self.keyword_chunk_embeddings,
            'keyword_contexts': self.keyword_contexts,

            # TF-IDF vectorizer
            'vectorizer': self.vectorizer,
        }

        # Save configuration as JSON for easy inspection
        config = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'context_window': self.context_window,
            'device': str(self.device),
            'embedding_model_name': self.embedding_model_name if hasattr(self, 'embedding_model_name') else '',
            'embedding_model': self.embedding_model.config._name_or_path if hasattr(self.embedding_model,
                                                                                    'config') else str(
                self.embedding_model),
            'tokenizer': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else str(
                self.tokenizer),
        }

        # Create subdirectories for different components
        vectors_path = os.path.join(path, 'vectors')
        os.makedirs(vectors_path, exist_ok=True)

        mappings_path = os.path.join(path, 'mappings')
        os.makedirs(mappings_path, exist_ok=True)

        cache_path = os.path.join(path, 'cache')
        os.makedirs(cache_path, exist_ok=True)

        models_path = os.path.join(path, 'models')
        os.makedirs(models_path, exist_ok=True)

        # Save config as JSON
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Save vector databases (these can be large)
        for db_name in ['keyword_db', 'chunk_db', 'keyword_chunk_db', 'vector_db']:
            if db_name in data and data[db_name] is not None:
                with open(os.path.join(vectors_path, f"{db_name}.pkl"), 'wb') as f:
                    pickle.dump(data[db_name], f)

        # Save mappings
        for mapping_name in ['id_to_keyword_context', 'keyword_context_to_id', 'id_to_chunk',
                             'chunk_id_to_metadata', 'keyword_chunk_id_to_metadata']:
            if mapping_name in data and data[mapping_name] is not None:
                with open(os.path.join(mappings_path, f"{mapping_name}.pkl"), 'wb') as f:
                    pickle.dump(data[mapping_name], f)

        # Save cached data
        for cache_name in ['chunk_texts', 'chunk_metadata', 'chunk_embeddings',
                           'keyword_chunk_embeddings', 'keyword_contexts']:
            if cache_name in data and data[cache_name] is not None:
                with open(os.path.join(cache_path, f"{cache_name}.pkl"), 'wb') as f:
                    pickle.dump(data[cache_name], f)

        # Save models (vectorizer)
        if 'vectorizer' in data and data['vectorizer'] is not None:
            with open(os.path.join(models_path, "vectorizer.pkl"), 'wb') as f:
                pickle.dump(data['vectorizer'], f)

        # Save a manifest file for validation during loading
        manifest = {
            'version': '1.0',
            'saved_components': {
                'vectors': [f"{db_name}.pkl" for db_name in ['keyword_db', 'chunk_db', 'keyword_chunk_db', 'vector_db']
                            if db_name in data and data[db_name] is not None],
                'mappings': [f"{mapping_name}.pkl" for mapping_name in
                             ['id_to_keyword_context', 'keyword_context_to_id',
                              'id_to_chunk', 'chunk_id_to_metadata',
                              'keyword_chunk_id_to_metadata']
                             if mapping_name in data and data[mapping_name] is not None],
                'cache': [f"{cache_name}.pkl" for cache_name in ['chunk_texts', 'chunk_metadata', 'chunk_embeddings',
                                                                 'keyword_chunk_embeddings', 'keyword_contexts']
                          if cache_name in data and data[cache_name] is not None],
                'models': ['vectorizer.pkl'] if 'vectorizer' in data and data['vectorizer'] is not None else []
            },
            'embedding_model_name': self.embedding_model_name if hasattr(self, 'embedding_model_name') else '',
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(os.path.join(path, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Model saved successfully to {path}")
        print(f"- Saved {len(self.keyword_db) if hasattr(self, 'keyword_db') and self.keyword_db else 0} keyword embeddings")
        print(f"- Saved {len(self.chunk_db) if hasattr(self, 'chunk_db') and self.chunk_db else 0} chunk embeddings")
        print(f"- Saved {len(self.keyword_chunk_db) if hasattr(self, 'keyword_chunk_db') and self.keyword_chunk_db else 0} keyword-chunk embeddings")
        print(f"- Saved vectorizer: {data['vectorizer'] is not None if 'vectorizer' in data else False}")
        print(f"- Embedding model name: {self.embedding_model_name if hasattr(self, 'embedding_model_name') else 'Not specified'}")

    def load(self, path: str):
        """
        Load a contextual RAG model from disk.

        This method loads all necessary components:
        - Vector databases (keyword, chunk, and keyword chunk)
        - Mappings and metadata
        - Tokenizer and embedding model configurations
        - Cached embeddings and texts
        - Vectorizer for TF-IDF
        - Embedding model name

        Args:
            path: Path to load the model from

        Returns:
            self: The loaded ContextualRAG instance
        """
        import os
        import pickle
        import json
        from pathlib import Path
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading model from {path}...")

        # Check if the path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path {path} does not exist")

        # Check if manifest exists
        manifest_path = os.path.join(path, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        print(f"Loading model version {manifest.get('version', 'unknown')}")
        print(f"Model timestamp: {manifest.get('timestamp', 'unknown')}")

        # Load configuration
        config_path = os.path.join(path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Set configuration parameters
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.context_window = config.get('context_window', 3)

        # Load or initialize embedding model and tokenizer
        embedding_model_name = config.get('embedding_model_name', manifest.get('embedding_model_name', ''))
        if embedding_model_name:
            print(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model_name = embedding_model_name
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        else:
            print("Warning: No embedding model name found. Using default model.")
            # Use default initialization from __init__

        # Set device
        device_str = config.get('device', 'cpu')
        self.device = torch.device(device_str if torch.cuda.is_available() and 'cuda' in device_str else 'cpu')
        if hasattr(self, 'embedding_model'):
            self.embedding_model.to(self.device)

        # Define paths for different components
        vectors_path = os.path.join(path, 'vectors')
        mappings_path = os.path.join(path, 'mappings')
        cache_path = os.path.join(path, 'cache')
        models_path = os.path.join(path, 'models')

        # Function to load pickle files
        def load_pickle(component_path, filename):
            file_path = os.path.join(component_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Warning: File {file_path} not found")
                return None

        # Load vector databases
        if os.path.exists(vectors_path):
            print("Loading vector databases...")
            self.keyword_db = load_pickle(vectors_path, 'keyword_db.pkl')
            self.chunk_db = load_pickle(vectors_path, 'chunk_db.pkl')
            self.keyword_chunk_db = load_pickle(vectors_path, 'keyword_chunk_db.pkl')
            self.vector_db = load_pickle(vectors_path, 'vector_db.pkl')

            print(f"Loaded {len(self.keyword_db) if self.keyword_db else 0} keyword embeddings")
            print(f"Loaded {len(self.chunk_db) if self.chunk_db else 0} chunk embeddings")
            print(f"Loaded {len(self.keyword_chunk_db) if self.keyword_chunk_db else 0} keyword-chunk embeddings")
        else:
            print("Warning: Vector databases directory not found")
            self.keyword_db = {}
            self.chunk_db = {}
            self.keyword_chunk_db = {}
            self.vector_db = {'ids': [], 'vectors': []}

        # Load mappings
        if os.path.exists(mappings_path):
            print("Loading mappings...")
            self.id_to_keyword_context = load_pickle(mappings_path, 'id_to_keyword_context.pkl')
            self.keyword_context_to_id = load_pickle(mappings_path, 'keyword_context_to_id.pkl')
            self.id_to_chunk = load_pickle(mappings_path, 'id_to_chunk.pkl')
            self.chunk_id_to_metadata = load_pickle(mappings_path, 'chunk_id_to_metadata.pkl')
            self.keyword_chunk_id_to_metadata = load_pickle(mappings_path, 'keyword_chunk_id_to_metadata.pkl')
        else:
            print("Warning: Mappings directory not found")
            self.id_to_keyword_context = {}
            self.keyword_context_to_id = {}
            self.id_to_chunk = {}
            self.chunk_id_to_metadata = {}
            self.keyword_chunk_id_to_metadata = {}

        # Load cached data
        if os.path.exists(cache_path):
            print("Loading cached data...")
            self.chunk_texts = load_pickle(cache_path, 'chunk_texts.pkl')
            self.chunk_metadata = load_pickle(cache_path, 'chunk_metadata.pkl')
            self.chunk_embeddings = load_pickle(cache_path, 'chunk_embeddings.pkl')
            self.keyword_chunk_embeddings = load_pickle(cache_path, 'keyword_chunk_embeddings.pkl')
            self.keyword_contexts = load_pickle(cache_path, 'keyword_contexts.pkl')
        else:
            print("Warning: Cache directory not found")
            self.chunk_texts = []
            self.chunk_metadata = []
            self.chunk_embeddings = []
            self.keyword_chunk_embeddings = []
            self.keyword_contexts = {}

        # Load models (vectorizer)
        if os.path.exists(models_path):
            print("Loading models...")
            self.vectorizer = load_pickle(models_path, 'vectorizer.pkl')
            print(f"Loaded vectorizer: {self.vectorizer is not None}")
        else:
            print("Warning: Models directory not found")
            self.vectorizer = None

        # Initialize text splitter if needed
        if not hasattr(self, 'text_splitter') or self.text_splitter is None:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # Ensure NLTK resources are available
        self._ensure_nltk_resources()

        print("Model loaded successfully")


def main():
    """
    Main entry point for the Contextual RAG application.

    This function demonstrates the workflow for creating, training, and using
    a Contextual Retrieval Augmented Generation (RAG) system:
    1. Initialize the ContextualRAG with appropriate parameters
    2. Upload and process documents
    3. Create the knowledge database
    4. Process queries against the knowledge base
    5. Save the model for future use

    Returns:
        None
    """
    import os
    import time
    import torch
    from pathlib import Path

    # Define configuration parameters
    config = {
        # Model parameters
        "chunk_size": 1000,  # Size of text chunks for processing
        "chunk_overlap": 200,  # Overlap between chunks to maintain context
        "context_window": 200,  # Size of context window for keywords
        "embedding_model_name": "BAAI/bge-base-en-v1.5",  # Model for embeddings
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available

        # File paths
        "docs_dir": Path("./documents"),  # Directory containing source documents
        "model_path": Path("./rag_model"),  # Directory to save/load the RAG model

        # Query parameters
        "top_k": 10,  # Number of top results to return for each query
    }

    # Initialize the ContextualRAG system
    print("Initializing ContextualRAG system...")
    rag = ContextualRAG(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        context_window=config["context_window"],
        embedding_model_name=config["embedding_model_name"],
        device=config["device"]
    )

    # Check if a saved model exists
    if config["model_path"].exists():
        print(f"Loading existing model from {config['model_path']}...")
        rag.load(config["model_path"])
    else:
        # Process documents if no model exists
        if not config["docs_dir"].exists():
            print(f"Document directory {config['docs_dir']} not found!")
            return

        # Upload and process documents
        print(f"Processing documents from {config['docs_dir']}...")
        start_time = time.time()
        document_files = [str(f) for f in config["docs_dir"].glob("**/*")
                          if f.is_file() and f.suffix in ['.txt', '.pdf', '.docx']]

        if not document_files:
            print("No documents found to process!")
            return

        rag.upload_files(document_files)

        # Create knowledge database
        print("Building knowledge database...")
        rag.make_db()

        # Save the model for future use
        print(f"Saving model to {config['model_path']}...")
        rag.save(config["model_path"])

        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")

    # Interactive query loop
    print("\nContextual RAG system ready for queries (type 'exit' to quit)")
    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ['exit', 'quit']:
            break

        if not query.strip():
            continue

        # Process the query and get results
        start_time = time.time()
        results = rag.process_query(query, top_k=config["top_k"])
        elapsed_time = time.time() - start_time

        # Display results
        print(f"\nResults (retrieved in {elapsed_time:.2f} seconds):")

        if not results:
            print("No relevant information found.")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result.get('doc_id', 'Unknown')}")
            print(f"   Score: {result.get('score', 0):.4f}")
            print(f"   Context: {result.get('text', '')[:200]}...")

    print("Exiting Contextual RAG system.")


if __name__ == "__main__":
    main()

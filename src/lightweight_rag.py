"""
Lightweight RAG Engine Module

This module implements a simple, memory-efficient RAG system
that uses basic text matching instead of heavy embeddings.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger


class LightweightRAGEngine:
    """Lightweight RAG engine using basic text matching."""
    
    def __init__(self, persist_directory: str = "lightweight_rag_db"):
        """Initialize the lightweight RAG engine.
        
        Args:
            persist_directory: Path for storing document data
        """
        self.persist_directory = Path(persist_directory)
        self.documents = []
        self.persist_directory.mkdir(exist_ok=True)
        self._load_documents()
    
    def _load_documents(self):
        """Load existing documents from disk."""
        try:
            import pickle
            db_file = self.persist_directory / "documents.pkl"
            if db_file.exists():
                with open(db_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} existing documents")
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")
            self.documents = []
    
    def _save_documents(self):
        """Save documents to disk."""
        try:
            import pickle
            db_file = self.persist_directory / "documents.pkl"
            with open(db_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def add_documents(self, text_chunks: List[Dict[str, Any]]):
        """Add text chunks to the lightweight RAG system.
        
        Args:
            text_chunks: List of text chunks with metadata
        """
        if not text_chunks:
            logger.warning("No text chunks provided")
            return
        
        try:
            logger.info(f"Adding {len(text_chunks)} text chunks to lightweight RAG")
            
            # Process chunks in small batches
            batch_size = 5
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} chunks")
                
                for chunk in batch:
                    # Create simple document entry
                    doc_entry = {
                        'id': f"{chunk['filename']}_{chunk['chunk_id']}",
                        'text': chunk['text'],
                        'metadata': {
                            'filename': chunk['filename'],
                            'chunk_id': chunk['chunk_id'],
                            'start_char': chunk['start_char'],
                            'end_char': chunk['end_char']
                        }
                    }
                    self.documents.append(doc_entry)
                
                # Save after each batch
                self._save_documents()
                
                # Small delay to prevent memory buildup
                import time
                time.sleep(0.1)
            
            logger.info(f"Successfully added {len(text_chunks)} documents to lightweight RAG")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using basic text matching.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.documents:
                logger.warning("No documents in lightweight RAG system")
                return []
            
            # Simple keyword-based search
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            
            # Score documents based on word overlap
            scored_docs = []
            for doc in self.documents:
                doc_words = set(re.findall(r'\b\w+\b', doc['text'].lower()))
                
                # Calculate simple similarity score
                common_words = query_words.intersection(doc_words)
                score = len(common_words) / max(len(query_words), 1)
                
                if score > 0:  # Only include documents with some relevance
                    scored_docs.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': score
                    })
            
            # Sort by score and return top results
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            results = scored_docs[:n_results]
            
            logger.info(f"Retrieved {len(results)} relevant documents using lightweight search")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def get_context_for_prompt(self, prompt: str, max_chars: int = 2000) -> str:
        """Get relevant context for a given prompt.
        
        Args:
            prompt: The prompt to find context for
            max_chars: Maximum number of characters to include in context
            
        Returns:
            Formatted context string
        """
        # Search for relevant documents
        results = self.search_documents(prompt, n_results=10)
        
        if not results:
            logger.warning("No relevant documents found for prompt")
            return ""
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for result in results:
            text = result['text']
            filename = result['metadata'].get('filename', 'Unknown')
            score = result.get('score', 0)
            
            # Add document info and text
            context_line = f"[From {filename}, relevance: {score:.2f}]: {text}"
            
            if current_length + len(context_line) <= max_chars:
                context_parts.append(context_line)
                current_length += len(context_line)
            else:
                break
        
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context with {len(context)} characters from {len(context_parts)} documents")
        
        return context
    
    def clear_documents(self):
        """Clear all documents from the system."""
        try:
            self.documents = []
            self._save_documents()
            logger.info("Cleared all documents from lightweight RAG system")
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            return {
                'total_documents': len(self.documents),
                'collection_type': 'lightweight_rag',
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {} 
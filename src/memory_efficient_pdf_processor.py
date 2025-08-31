"""
Memory-Efficient PDF Processor Module

This module provides a memory-efficient way to process PDF documents
by reading and processing them in smaller chunks.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


class MemoryEfficientPDFProcessor:
    """Memory-efficient PDF text extraction and processing."""
    
    def __init__(self, dataset_folder: str = "dataset"):
        """Initialize the memory-efficient PDF processor.
        
        Args:
            dataset_folder: Path to folder containing PDF files
        """
        self.dataset_folder = Path(dataset_folder)
        self.extracted_texts = {}
    
    def extract_text_from_pdf_chunked(self, pdf_path: Path, chunk_size: int = 10000) -> str:
        """Extract text from PDF in chunks to avoid memory issues.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Number of characters to process at once
            
        Returns:
            Extracted text
        """
        try:
            logger.info(f"Processing PDF in chunks: {pdf_path.name}")
            
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    
                    # Process pages in smaller batches
                    total_pages = len(reader.pages)
                    logger.info(f"PDF has {total_pages} pages, processing in chunks")
                    
                    for page_num in range(total_pages):
                        try:
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            
                            if page_text:
                                # Process page text in chunks
                                for i in range(0, len(page_text), chunk_size):
                                    chunk = page_text[i:i + chunk_size]
                                    text_parts.append(chunk)
                                    
                                    # Small delay to prevent memory buildup
                                    import time
                                    time.sleep(0.01)
                            
                            logger.info(f"Processed page {page_num + 1}/{total_pages}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to process page {page_num + 1}: {e}")
                            continue
                    
                    full_text = ''.join(text_parts)
                    logger.info(f"Successfully extracted {len(full_text)} characters using PyPDF2")
                    return full_text
                    
            except ImportError:
                logger.info("PyPDF2 not available, trying pypdf")
            
            # Fallback to pypdf
            try:
                import pypdf
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    text_parts = []
                    
                    total_pages = len(reader.pages)
                    logger.info(f"PDF has {total_pages} pages, processing in chunks")
                    
                    for page_num in range(total_pages):
                        try:
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            
                            if page_text:
                                # Process page text in chunks
                                for i in range(0, len(page_text), chunk_size):
                                    chunk = page_text[i:i + chunk_size]
                                    text_parts.append(chunk)
                                    
                                    # Small delay to prevent memory buildup
                                    import time
                                    time.sleep(0.01)
                            
                            logger.info(f"Processed page {page_num + 1}/{total_pages}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to process page {page_num + 1}: {e}")
                            continue
                    
                    full_text = ''.join(text_parts)
                    logger.info(f"Successfully extracted {len(full_text)} characters using pypdf")
                    return full_text
                    
            except ImportError:
                logger.error("Neither PyPDF2 nor pypdf available")
                raise ImportError("No PDF processing library available")
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    def process_all_pdfs_chunked(self, chunk_size: int = 10000) -> Dict[str, str]:
        """Process all PDFs in the dataset folder using chunked processing.
        
        Args:
            chunk_size: Number of characters to process at once
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        try:
            pdf_files = list(self.dataset_folder.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.dataset_folder}")
                return {}
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Processing {pdf_file.name}")
                    text = self.extract_text_from_pdf_chunked(pdf_file, chunk_size)
                    
                    if text:
                        self.extracted_texts[pdf_file.name] = text
                        logger.info(f"Successfully extracted {len(text)} characters from {pdf_file.name}")
                    
                    # Force garbage collection after each PDF
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(self.extracted_texts)} PDF files")
            return self.extracted_texts
            
        except Exception as e:
            logger.error(f"Failed to process PDFs: {e}")
            return {}
    
    def get_text_chunks_efficient(self, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """Create text chunks efficiently with minimal memory usage.
        
        Args:
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        try:
            if not self.extracted_texts:
                logger.warning("No extracted texts available")
                return []
            
            text_chunks = []
            chunk_id = 0
            
            for filename, text in self.extracted_texts.items():
                logger.info(f"Creating chunks for {filename} ({len(text)} characters)")
                
                # Process text in smaller segments
                segment_size = 5000  # Process 5K characters at a time
                
                for start in range(0, len(text), segment_size):
                    end = min(start + segment_size, len(text))
                    segment = text[start:end]
                    
                    # Create chunks from this segment
                    for i in range(0, len(segment), chunk_size - overlap):
                        chunk_start = start + i
                        chunk_end = min(chunk_start + chunk_size, len(text))
                        
                        if chunk_end - chunk_start >= chunk_size // 2:  # Only include substantial chunks
                            chunk_text = text[chunk_start:chunk_end]
                            
                            chunk_data = {
                                'text': chunk_text,
                                'filename': filename,
                                'chunk_id': chunk_id,
                                'start_char': chunk_start,
                                'end_char': chunk_end,
                                'length': len(chunk_text)
                            }
                            
                            text_chunks.append(chunk_data)
                            chunk_id += 1
                            
                            # Small delay to prevent memory buildup
                            import time
                            time.sleep(0.001)
                
                logger.info(f"Created {chunk_id} chunks from {filename}")
                
                # Force garbage collection after each file
                import gc
                gc.collect()
            
            logger.info(f"Total chunks created: {len(text_chunks)}")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Failed to create text chunks: {e}")
            return []
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about PDF files without processing them.
        
        Returns:
            Dictionary with file information
        """
        try:
            pdf_files = list(self.dataset_folder.glob("*.pdf"))
            file_info = {}
            
            for pdf_file in pdf_files:
                try:
                    size = pdf_file.stat().st_size
                    file_info[pdf_file.name] = {
                        'size_bytes': size,
                        'size_mb': size / (1024 * 1024),
                        'path': str(pdf_file)
                    }
                except Exception as e:
                    logger.warning(f"Could not get info for {pdf_file.name}: {e}")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {} 
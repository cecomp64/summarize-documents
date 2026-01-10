"""
Document Summarization Tool

A Python tool that processes text files to extract articles, generate AI-powered summaries,
and create structured JSON output with metadata.
"""

__version__ = "0.1.0"

from .processor import DocumentProcessor, Article, Document

__all__ = ["DocumentProcessor", "Article", "Document", "__version__"]

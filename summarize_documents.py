#!/usr/bin/env python3
"""
Document Summarization Tool - CLI Wrapper

This is a thin wrapper around the document_summarizer package.
All logic is in src/document_summarizer/ to avoid code duplication.
"""

import sys
from pathlib import Path

# Add src directory to path so we can import the package
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from document_summarizer.cli import main

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Document Summarization Tool

Processes text files to extract articles, generate summaries, and create structured JSON output.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv


class Article:
    """Represents an article within a document."""

    def __init__(self, title: str, content: str, page_start: Optional[int] = None, page_end: Optional[int] = None):
        self.title = title
        self.content = content
        self.page_start = page_start
        self.page_end = page_end
        self.summary = ""
        self.categories = []
        self.keywords = []

    def to_dict(self, document_id: str) -> Dict:
        """Convert article to dictionary format."""
        article_id = self.generate_id(document_id)
        return {
            "id": article_id,
            "documentId": document_id,
            "title": self.title,
            "summary": self.summary,
            "categories": self.categories,
            "keywords": self.keywords,
            "pageStart": self.page_start,
            "pageEnd": self.page_end
        }

    def generate_id(self, document_id: str) -> str:
        """Generate article ID from document ID and title."""
        title_slug = re.sub(r'[^a-z0-9]+', '-', self.title.lower()).strip('-')
        return f"{document_id}-{title_slug}"


class Document:
    """Represents a document (text file)."""

    def __init__(self, txt_path: Path, root_dir: Path):
        self.txt_path = txt_path
        self.root_dir = root_dir
        self.id = self.generate_id()
        self.title = ""
        self.issue_date = None
        self.pdf_path = None
        self.articles = []

    def generate_id(self) -> str:
        """Generate document ID from filename."""
        stem = self.txt_path.stem
        # Convert filename to lowercase slug
        doc_id = re.sub(r'[^a-z0-9]+', '-', stem.lower()).strip('-')
        return doc_id

    def find_pdf(self) -> Optional[str]:
        """Find corresponding PDF file."""
        # Look for PDF with same name in same directory
        pdf_path = self.txt_path.with_suffix('.pdf')
        if pdf_path.exists():
            # Return relative path from root_dir
            return str(pdf_path.relative_to(self.root_dir))
        return None

    def to_dict(self) -> Dict:
        """Convert document to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "issueDate": self.issue_date,
            "pdfPath": self.pdf_path,
            "txtPath": str(self.txt_path.relative_to(self.root_dir))
        }


class DocumentProcessor:
    """Processes documents and extracts articles."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = Anthropic(api_key=api_key) if api_key else None
        self.page_marker_pattern = re.compile(r'\[?(?:page|pg\.?|p\.?)\s*(\d+)\]?', re.IGNORECASE)

    def find_files(self, directory: Path, pattern: str = "*.txt") -> List[Path]:
        """Find all files matching the pattern recursively."""
        return list(directory.rglob(pattern))

    def extract_articles(self, content: str, document: Document) -> List[Article]:
        """Split document content into articles."""
        articles = []
        current_page = None

        # Split by common section markers (headers in all caps, numbered sections, etc.)
        # This is a heuristic approach - can be customized based on document structure
        sections = self._split_into_sections(content)

        for section in sections:
            if not section.strip():
                continue

            # Extract title (first line or detected header)
            title, body, page_start, page_end = self._parse_section(section)

            if title:
                article = Article(title, body, page_start, page_end)
                articles.append(article)

        return articles

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into sections based on headers."""
        lines = content.split('\n')
        sections = []
        current_section = []

        for line in lines:
            # Detect section headers (all caps lines, numbered headers, etc.)
            if self._is_section_header(line):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        return sections

    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is a section header."""
        stripped = line.strip()
        if not stripped:
            return False

        # Check for all caps (at least 3 words)
        if stripped.isupper() and len(stripped.split()) >= 2:
            return True

        # Check for numbered sections (e.g., "1. Introduction", "I. Overview")
        if re.match(r'^(?:\d+\.|[IVX]+\.|\([a-z]\))\s+[A-Z]', stripped):
            return True

        return False

    def _parse_section(self, section: str) -> tuple:
        """Parse section into title, body, and page numbers."""
        lines = section.split('\n')
        title = lines[0].strip() if lines else "Untitled"
        body = '\n'.join(lines[1:]).strip()

        # Extract page numbers from content
        page_start = None
        page_end = None
        page_numbers = []

        for match in self.page_marker_pattern.finditer(section):
            page_num = int(match.group(1))
            page_numbers.append(page_num)

        if page_numbers:
            page_start = min(page_numbers)
            page_end = max(page_numbers)

        return title, body, page_start, page_end

    def process_with_ai(self, article: Article, document: Document) -> None:
        """Use Claude API to generate summary, categories, and keywords."""
        if not self.client:
            # Fallback without AI
            article.summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            article.categories = ["uncategorized"]
            article.keywords = []
            return

        prompt = f"""Analyze the following article and provide:
1. A concise summary (2-3 sentences)
2. Relevant categories (e.g., events, observing reports, news, technical, etc.)
3. Key keywords or topics

Article Title: {article.title}
Document: {document.title}

Content:
{article.content[:2000]}

Respond in JSON format:
{{
  "summary": "...",
  "categories": ["category1", "category2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            result = json.loads(response_text)

            article.summary = result.get("summary", "")
            article.categories = result.get("categories", [])
            article.keywords = result.get("keywords", [])

        except Exception as e:
            print(f"Error processing article '{article.title}': {e}")
            # Fallback
            article.summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            article.categories = ["uncategorized"]
            article.keywords = []

    def extract_document_metadata(self, content: str, document: Document) -> None:
        """Extract document-level metadata."""
        lines = content.split('\n')

        # Try to extract title from first non-empty line or first substantial line
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 10:
                document.title = stripped
                break

        # Try to extract date from filename or content
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(document.txt_path))
        if date_match:
            document.issue_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        else:
            # Try to find date in content
            date_match = re.search(r'(\d{4})\s*[-/]\s*(\d{2})\s*[-/]\s*(\d{2})', content[:500])
            if date_match:
                document.issue_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

        # Find corresponding PDF
        document.pdf_path = document.find_pdf()

    def process_document(self, txt_path: Path, root_dir: Path) -> Document:
        """Process a single document."""
        print(f"Processing: {txt_path}")

        document = Document(txt_path, root_dir)

        # Read content
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract document metadata
        self.extract_document_metadata(content, document)

        # Extract articles
        articles = self.extract_articles(content, document)

        # Process each article with AI
        for article in articles:
            print(f"  - Processing article: {article.title}")
            self.process_with_ai(article, document)

        document.articles = articles

        return document

    def generate_output(self, documents: List[Document]) -> Dict:
        """Generate final JSON output."""
        output = {
            "articles": [],
            "documents": []
        }

        for doc in documents:
            # Add document
            output["documents"].append(doc.to_dict())

            # Add articles
            for article in doc.articles:
                output["articles"].append(article.to_dict(doc.id))

        return output

    def save_json(self, output: Dict, txt_path: Path) -> None:
        """Save JSON output next to the text file."""
        json_path = txt_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process text files and extract articles with summaries"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for text files"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for files to process (default: *.txt)"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env variable)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create a single combined JSON file instead of one per document"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for combined JSON (used with --combined)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("Warning: No API key provided. Summaries will be basic excerpts.")
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key option.")

    # Initialize processor
    processor = DocumentProcessor(api_key)

    # Find files
    directory = args.directory.resolve()
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1

    txt_files = processor.find_files(directory, args.pattern)

    if not txt_files:
        print(f"No files matching '{args.pattern}' found in {directory}")
        return 1

    print(f"Found {len(txt_files)} file(s) to process\n")

    # Process documents
    documents = []
    for txt_path in txt_files:
        doc = processor.process_document(txt_path, directory)
        documents.append(doc)

        # Save individual JSON unless combined output requested
        if not args.combined:
            output = processor.generate_output([doc])
            processor.save_json(output, txt_path)

        print()

    # Save combined output if requested
    if args.combined:
        output = processor.generate_output(documents)
        output_path = args.output or directory / "combined_output.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved combined output: {output_path}")

    print(f"\nProcessed {len(documents)} document(s) successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

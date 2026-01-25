"""
Core document processing functionality.
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from anthropic import Anthropic
import ollama
from openai import OpenAI
from google import genai


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
        self.embedding = None  # Will store embedding vector if generated

    def to_dict(self, document_id: str, include_embedding: bool = True) -> Dict:
        """Convert article to dictionary format."""
        article_id = self.generate_id(document_id)
        result = {
            "id": article_id,
            "documentId": document_id,
            "title": self.title,
            "summary": self.summary,
            "categories": self.categories,
            "keywords": self.keywords,
            "pageStart": self.page_start,
            "pageEnd": self.page_end
        }

        # Include embedding if requested and available
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding

        return result

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

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514",
                 model_provider: str = "anthropic",
                 generate_embeddings: bool = False, embedding_provider: str = "openai",
                 embedding_model: Optional[str] = None, openai_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        self.model_provider = model_provider
        self.model = model
        self.api_key = api_key
        self.generate_embeddings = generate_embeddings
        self.embedding_provider = embedding_provider

        # Set default embedding models based on provider
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            # Defaults for each provider
            if embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-small"
            elif embedding_provider == "anthropic":
                self.embedding_model = "voyage-3"  # Anthropic uses Voyage AI
            elif embedding_provider == "ollama":
                self.embedding_model = "embeddinggemma"  # Default Ollama embedding model
            elif embedding_provider == "gemini":
                self.embedding_model = "models/text-embedding-004"  # Gemini embedding model
            else:
                self.embedding_model = "text-embedding-3-small"

        # Initialize summarization client
        if model_provider == "anthropic":
            self.client = Anthropic(api_key=api_key) if api_key else None
        elif model_provider == "ollama":
            self.client = None  # Ollama uses direct API calls
        elif model_provider == "gemini":
            self.client = genai.Client(api_key=api_key) if api_key else None
        else:
            raise ValueError(f"Unknown model provider: {model_provider}")

        # Initialize embedding client
        self.embedding_client = None
        if generate_embeddings:
            if embedding_provider == "openai":
                self.embedding_client = OpenAI(api_key=openai_api_key)
            elif embedding_provider == "anthropic":
                # Anthropic uses Voyage AI for embeddings
                import voyageai
                self.embedding_client = voyageai.Client(api_key=api_key)
            elif embedding_provider == "ollama":
                # Ollama embeddings use the same interface as chat
                self.embedding_client = None
            elif embedding_provider == "gemini":
                # Configure Gemini API with new package
                self.embedding_client = genai.Client(api_key=gemini_api_key)
            else:
                raise ValueError(f"Unknown embedding provider: {embedding_provider}")

        self.page_marker_pattern = re.compile(r'\[?(?:page|pg\.?|p\.?)\s*(\d+)\]?', re.IGNORECASE)

    def find_files(self, directory: Path, pattern: str = "*.txt") -> List[Path]:
        """Find all files matching the pattern recursively."""
        return list(directory.rglob(pattern))

    def load_existing_json(self, json_path: Path) -> Optional[Dict]:
        """Load existing JSON summary file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None

    def load_document_from_json(self, json_path: Path, txt_path: Path, root_dir: Path) -> Optional[Document]:
        """Load a Document object from existing JSON file."""
        data = self.load_existing_json(json_path)
        if not data or "documents" not in data or "articles" not in data:
            return None

        # Create Document object
        document = Document(txt_path, root_dir)

        # Load document metadata
        if data["documents"]:
            doc_data = data["documents"][0]
            document.id = doc_data.get("id", document.id)
            document.title = doc_data.get("title", "")
            document.issue_date = doc_data.get("issueDate")
            document.pdf_path = doc_data.get("pdfPath")

        # Load articles
        for article_data in data["articles"]:
            article = Article(
                title=article_data.get("title", ""),
                content="",  # Content not stored in JSON
                page_start=article_data.get("pageStart"),
                page_end=article_data.get("pageEnd")
            )
            article.summary = article_data.get("summary", "")
            article.categories = article_data.get("categories", [])
            article.keywords = article_data.get("keywords", [])
            document.articles.append(article)

        return document

    def generate_embeddings_from_json(self, json_path: Path) -> Optional[Path]:
        """Generate embeddings from existing JSON summary file."""
        print(f"\nProcessing embeddings for: {json_path}")
        start_time = time.time()

        # Load existing JSON
        data = self.load_existing_json(json_path)
        if not data or "articles" not in data:
            print("  Skipping: No articles found in JSON")
            return None

        # Generate embeddings for each article
        print(f"  Generating embeddings for {len(data['articles'])} article(s)...")
        embeddings_list = []

        for i, article_data in enumerate(data['articles'], 1):
            summary = article_data.get("summary", "")
            if not summary:
                print(f"  [{i}/{len(data['articles'])}] Skipping '{article_data.get('title', 'Unknown')}': No summary")
                continue

            # Create temporary Article object to generate embedding
            temp_article = Article(
                title=article_data.get("title", ""),
                content="",  # Not needed for embeddings
            )
            temp_article.summary = summary

            print(f"  [{i}/{len(data['articles'])}] {temp_article.title[:60]}...", end=" ")
            self.generate_embedding(temp_article)

            if temp_article.embedding:
                embeddings_list.append({
                    "articleId": article_data["id"],
                    "vector": temp_article.embedding
                })
                print("✓")
            else:
                print("✗")

        # Save embeddings file
        if embeddings_list:
            embeddings_path = json_path.parent / f"{json_path.stem}-embeddings.json"
            embeddings_output = {"embeddings": embeddings_list}

            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_output, f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time
            print(f"  Saved embeddings: {embeddings_path}")
            print(f"  Processing completed in {elapsed:.2f}s")
            return embeddings_path
        else:
            print("  No embeddings generated")
            return None

    def extract_articles(self, content: str, document: Document) -> List[Article]:
        """Split document content into articles using markdown headers or AI-based segmentation."""
        # First try markdown-based extraction
        articles = self._extract_articles_from_markdown(content, document)

        # If markdown extraction found articles, use them
        if articles:
            print(f"  Found {len(articles)} article(s) using markdown headers")
        else:
            # Otherwise use AI to intelligently detect article boundaries
            if self.model_provider == "anthropic" and self.client:
                articles = self._extract_articles_with_ai_anthropic(content, document)
            elif self.model_provider == "ollama":
                articles = self._extract_articles_with_ai_ollama(content, document)
            elif self.model_provider == "gemini" and self.client:
                articles = self._extract_articles_with_ai_gemini(content, document)
            else:
                # Fallback to page-based segmentation
                articles = self._extract_articles_by_page(content, document)

        # Filter out articles that are too short (less than 100 chars)
        MIN_ARTICLE_LENGTH = 100
        articles = [a for a in articles if len(a.content) >= MIN_ARTICLE_LENGTH]

        return articles

    def _extract_articles_from_markdown(self, content: str, document: Document) -> List[Article]:
        """Extract articles based on markdown headers (# and ##)."""
        articles = []
        lines = content.split('\n')

        # Find all markdown headers (# or ##) that look like article titles
        header_positions = []
        for i, line in enumerate(lines):
            # Match markdown headers: # Title or ## Title
            if re.match(r'^#{1,2}\s+\*?\*?[A-Z]', line):
                # Extract title by removing markdown syntax
                title = re.sub(r'^#{1,2}\s+', '', line).strip()
                title = re.sub(r'\*\*|\*', '', title)  # Remove bold markers
                header_positions.append((i, title))

        # If we found headers, split content by them
        if len(header_positions) >= 2:  # Need at least 2 to make meaningful articles
            for idx in range(len(header_positions)):
                start_line, title = header_positions[idx]

                # Determine end line
                if idx + 1 < len(header_positions):
                    end_line = header_positions[idx + 1][0]
                else:
                    end_line = len(lines)

                # Extract article content
                article_lines = lines[start_line:end_line]
                article_content = '\n'.join(article_lines).strip()

                # Extract page numbers from content
                page_numbers = []
                for match in self.page_marker_pattern.finditer(article_content):
                    page_num = int(match.group(1))
                    page_numbers.append(page_num)

                page_start = min(page_numbers) if page_numbers else None
                page_end = max(page_numbers) if page_numbers else None

                articles.append(Article(title, article_content, page_start, page_end))

        return articles

    def _extract_articles_with_ai_anthropic(self, content: str, document: Document) -> List[Article]:
        """Use Claude AI to intelligently segment document into articles."""
        print("  Using AI to identify article boundaries...")
        start_time = time.time()

        # Use full content for analysis
        content_for_analysis = content
        print(f"  Analyzing {len(content_for_analysis):,} characters")

        prompt = f"""Analyze this document and identify distinct articles or sections within it.

Document content:
{content_for_analysis}

Instructions:
- Identify natural article boundaries based on content, topics, and structure
- Each article should have a clear topic or theme
- Avoid breaking up content that belongs together
- Watch out for numbered lists or bullet points - these are NOT article boundaries
- If no clear articles exist, segment by major topics or by page (look for [Page N] markers)
- Each article should be at least 100 characters long
- Provide a descriptive title for each article

Respond with a JSON array of articles:
[
  {{
    "title": "Article title",
    "start_marker": "First few words of the article (10-20 words)",
    "end_marker": "Last few words of the article (10-20 words)"
  }}
]"""

        try:
            api_start = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            api_time = time.time() - api_start
            print(f"  AI API call completed in {api_time:.2f}s")

            response_text = response.content[0].text

            # Extract JSON from response
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            article_specs = json.loads(response_text)
            print(f"  AI identified {len(article_specs)} potential article(s)")

            # Extract articles based on AI-identified boundaries
            articles = []
            for i, spec in enumerate(article_specs, 1):
                title = spec.get("title", "Untitled")
                start_marker = spec.get("start_marker", "")
                end_marker = spec.get("end_marker", "")

                print(f"    [{i}] Extracting: {title[:60]}...")

                # Find article content using markers
                article_content, page_start, page_end = self._extract_article_by_markers(
                    content, start_marker, end_marker
                )

                if article_content:
                    article = Article(title, article_content, page_start, page_end)
                    articles.append(article)
                    print(f"        ✓ Extracted {len(article_content):,} characters", end="")
                    if page_start:
                        print(f" (pages {page_start}-{page_end})")
                    else:
                        print()
                else:
                    print(f"        ✗ Failed to extract content")

            elapsed = time.time() - start_time
            print(f"  Article identification completed in {elapsed:.2f}s")

            # If AI extraction failed or returned no articles, fallback
            if not articles:
                print("  AI segmentation returned no articles, using page-based fallback")
                return self._extract_articles_by_page(content, document)

            return articles

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Error in AI article extraction after {elapsed:.2f}s: {e}")
            print("  Falling back to page-based segmentation")
            return self._extract_articles_by_page(content, document)

    def _extract_articles_with_ai_ollama(self, content: str, document: Document) -> List[Article]:
        """Use Ollama to intelligently segment document into articles."""
        print("  Using Ollama to identify article boundaries...")
        start_time = time.time()

        # Use full content for analysis
        content_for_analysis = content
        print(f"  Analyzing {len(content_for_analysis):,} characters")

        prompt = f"""Analyze this document and identify distinct articles or sections within it.

Document content:
{content_for_analysis}

Instructions:
- Identify natural article boundaries based on content, topics, and structure
- Each article should have a clear topic or theme
- Avoid breaking up content that belongs together
- Watch out for numbered lists or bullet points - these are NOT article boundaries
- If no clear articles exist, segment by major topics or by page (look for [Page N] markers)
- Each article should be at least 100 characters long
- Provide a descriptive title for each article

Respond with a JSON array of articles:
[
  {{
    "title": "Article title",
    "start_marker": "First few words of the article (10-20 words)",
    "end_marker": "Last few words of the article (10-20 words)"
  }}
]"""

        try:
            api_start = time.time()
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            api_time = time.time() - api_start
            print(f"  Ollama API call completed in {api_time:.2f}s")

            response_text = response['message']['content']

            article_specs = json.loads(response_text)

            # Handle case where response is wrapped in an object
            if isinstance(article_specs, dict) and 'articles' in article_specs:
                article_specs = article_specs['articles']

            # Ensure we have a list
            if not isinstance(article_specs, list):
                raise ValueError(f"Expected list of articles, got {type(article_specs)}")

            print(f"  AI identified {len(article_specs)} potential article(s)")

            # Extract articles based on AI-identified boundaries
            articles = []
            for i, spec in enumerate(article_specs, 1):
                # Handle both dict and non-dict items
                if not isinstance(spec, dict):
                    print(f"    [{i}] Skipping invalid spec: {spec}")
                    continue

                title = spec.get("title", "Untitled")
                start_marker = spec.get("start_marker", "")
                end_marker = spec.get("end_marker", "")

                print(f"    [{i}] Extracting: {title[:60]}...")

                article_content, page_start, page_end = self._extract_article_by_markers(
                    content, start_marker, end_marker
                )

                if article_content:
                    article = Article(title, article_content, page_start, page_end)
                    articles.append(article)
                    print(f"        ✓ Extracted {len(article_content):,} characters", end="")
                    if page_start:
                        print(f" (pages {page_start}-{page_end})")
                    else:
                        print()
                else:
                    print(f"        ✗ Failed to extract content")

            elapsed = time.time() - start_time
            print(f"  Article identification completed in {elapsed:.2f}s")

            # If AI extraction failed or returned no articles, fallback
            if not articles:
                print("  AI segmentation returned no articles, using page-based fallback")
                return self._extract_articles_by_page(content, document)

            return articles

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Error in AI article extraction with Ollama after {elapsed:.2f}s: {e}")
            print("  Falling back to page-based segmentation")
            return self._extract_articles_by_page(content, document)

    def _extract_articles_with_ai_gemini(self, content: str, document: Document) -> List[Article]:
        """Use Gemini to intelligently segment document into articles."""
        print("  Using Gemini to identify article boundaries...")
        start_time = time.time()

        # Use full content for analysis
        content_for_analysis = content
        print(f"  Analyzing {len(content_for_analysis):,} characters")

        prompt = f"""Analyze this document and identify distinct articles or sections within it.

Document content:
{content_for_analysis}

Instructions:
- Identify natural article boundaries based on content, topics, and structure
- Each article should have a clear topic or theme
- Avoid breaking up content that belongs together
- Watch out for numbered lists or bullet points - these are NOT article boundaries
- If no clear articles exist, segment by major topics or by page (look for [Page N] markers)
- Each article should be at least 100 characters long
- Provide a descriptive title for each article

Respond with a JSON array of articles:
[
  {{
    "title": "Article title",
    "start_marker": "First few words of the article (10-20 words)",
    "end_marker": "Last few words of the article (10-20 words)"
  }}
]"""

        try:
            api_start = time.time()

            # Try with JSON mode first
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )
            except Exception as json_error:
                # Check if it's a JSON mode not supported error
                error_str = str(json_error)
                if 'JSON mode is not enabled' in error_str or 'INVALID_ARGUMENT' in error_str:
                    print(f"  Note: JSON mode not supported by {self.model}, using text mode")
                    # Fall back to text mode
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                else:
                    raise

            api_time = time.time() - api_start
            print(f"  Gemini API call completed in {api_time:.2f}s")

            response_text = response.text

            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            article_specs = json.loads(response_text)

            # Handle case where response is wrapped in an object
            if isinstance(article_specs, dict) and 'articles' in article_specs:
                article_specs = article_specs['articles']

            # Ensure we have a list
            if not isinstance(article_specs, list):
                raise ValueError(f"Expected list of articles, got {type(article_specs)}")

            print(f"  AI identified {len(article_specs)} potential article(s)")

            # Extract articles based on AI-identified boundaries
            articles = []
            for i, spec in enumerate(article_specs, 1):
                # Handle both dict and non-dict items
                if not isinstance(spec, dict):
                    print(f"    [{i}] Skipping invalid spec: {spec}")
                    continue

                title = spec.get("title", "Untitled")
                start_marker = spec.get("start_marker", "")
                end_marker = spec.get("end_marker", "")

                print(f"    [{i}] Extracting: {title[:60]}...")

                article_content, page_start, page_end = self._extract_article_by_markers(
                    content, start_marker, end_marker
                )

                if article_content:
                    article = Article(title, article_content, page_start, page_end)
                    articles.append(article)
                    print(f"        ✓ Extracted {len(article_content):,} characters", end="")
                    if page_start:
                        print(f" (pages {page_start}-{page_end})")
                    else:
                        print()
                else:
                    print(f"        ✗ Failed to extract content")

            elapsed = time.time() - start_time
            print(f"  Article identification completed in {elapsed:.2f}s")

            # If AI extraction failed or returned no articles, fallback
            if not articles:
                print("  AI segmentation returned no articles, using page-based fallback")
                return self._extract_articles_by_page(content, document)

            return articles

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Error in AI article extraction with Gemini after {elapsed:.2f}s: {e}")
            print("  Falling back to page-based segmentation")
            return self._extract_articles_by_page(content, document)

    def _extract_article_by_markers(self, content: str, start_marker: str, end_marker: str) -> tuple:
        """Extract article content between start and end markers."""
        # Normalize whitespace in markers for better matching
        start_marker = ' '.join(start_marker.split())
        end_marker = ' '.join(end_marker.split())

        # Find start position
        start_pos = content.find(start_marker)
        if start_pos == -1:
            # Try fuzzy matching - find closest match
            content_normalized = ' '.join(content.split())
            start_pos = content_normalized.find(start_marker)
            if start_pos == -1:
                return None, None, None

        # Find end position
        end_pos = content.find(end_marker, start_pos)
        if end_pos == -1:
            # Take rest of document
            end_pos = len(content)
        else:
            # Include the end marker
            end_pos += len(end_marker)

        article_content = content[start_pos:end_pos].strip()

        # Extract page numbers from content
        page_numbers = []
        for match in self.page_marker_pattern.finditer(article_content):
            page_num = int(match.group(1))
            page_numbers.append(page_num)

        page_start = min(page_numbers) if page_numbers else None
        page_end = max(page_numbers) if page_numbers else None

        return article_content, page_start, page_end

    def _extract_articles_by_page(self, content: str, document: Document) -> List[Article]:
        """Fallback: segment document by page markers."""
        articles = []

        # Find all page markers
        page_splits = []
        for match in self.page_marker_pattern.finditer(content):
            page_num = int(match.group(1))
            page_splits.append((match.start(), page_num))

        if not page_splits:
            # No page markers - treat entire document as one article
            title = document.title or "Complete Document"
            articles.append(Article(title, content, None, None))
            return articles

        # Split by pages
        for i in range(len(page_splits)):
            start_pos, page_num = page_splits[i]
            end_pos = page_splits[i + 1][0] if i + 1 < len(page_splits) else len(content)

            page_content = content[start_pos:end_pos].strip()

            # Create article title from first line or use generic title
            lines = page_content.split('\n')
            first_line = next((line.strip() for line in lines if line.strip()), "")
            title = first_line[:50] if first_line else f"Page {page_num}"

            if len(title) > 50:
                title = title[:47] + "..."

            articles.append(Article(title, page_content, page_num, page_num))

        return articles

    def process_with_ai(self, article: Article, document: Document) -> None:
        """Use AI (Anthropic, Ollama, or Gemini) to generate summary, categories, and keywords."""
        if self.model_provider == "anthropic":
            self._process_with_anthropic(article, document)
        elif self.model_provider == "ollama":
            self._process_with_ollama(article, document)
        elif self.model_provider == "gemini":
            self._process_with_gemini(article, document)
        else:
            # Fallback without AI
            article.summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            article.categories = ["uncategorized"]
            article.keywords = []

    def _process_with_anthropic(self, article: Article, document: Document) -> None:
        """Use Anthropic Claude API to generate summary, categories, and keywords."""
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
{article.content}

Respond in JSON format:
{{
  "summary": "...",
  "categories": ["category1", "category2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
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

    def _process_with_ollama(self, article: Article, document: Document) -> None:
        """Use Ollama API to generate summary, categories, and keywords."""
        prompt = f"""Analyze the following article and provide:
1. A concise summary (2-3 sentences)
2. Relevant categories (e.g., events, observing reports, news, technical, etc.)
3. Key keywords or topics

Article Title: {article.title}
Document: {document.title}

Content:
{article.content}

Respond in JSON format:
{{
  "summary": "...",
  "categories": ["category1", "category2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )

            response_text = response['message']['content']

            # Parse JSON response
            result = json.loads(response_text)

            article.summary = result.get("summary", "")
            article.categories = result.get("categories", [])
            article.keywords = result.get("keywords", [])

        except Exception as e:
            print(f"Error processing article '{article.title}' with Ollama: {e}")
            # Fallback
            article.summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            article.categories = ["uncategorized"]
            article.keywords = []

    def _process_with_gemini(self, article: Article, document: Document) -> None:
        """Use Gemini API to generate summary, categories, and keywords."""
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
{article.content}

Respond in JSON format:
{{
  "summary": "...",
  "categories": ["category1", "category2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""

        try:
            # Try with JSON mode first
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )
            except Exception as json_error:
                # Check if it's a JSON mode not supported error
                error_str = str(json_error)
                if 'JSON mode is not enabled' in error_str or 'INVALID_ARGUMENT' in error_str:
                    # Fall back to text mode (only print once per processor instance)
                    if not hasattr(self, '_gemini_json_warning_shown'):
                        print(f"  Note: JSON mode not supported by {self.model}, using text mode")
                        self._gemini_json_warning_shown = True
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                else:
                    raise

            response_text = response.text

            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            # Parse JSON response
            result = json.loads(response_text)

            article.summary = result.get("summary", "")
            article.categories = result.get("categories", [])
            article.keywords = result.get("keywords", [])

        except Exception as e:
            print(f"Error processing article '{article.title}' with Gemini: {e}")
            # Fallback
            article.summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            article.categories = ["uncategorized"]
            article.keywords = []

    def generate_embedding(self, article: Article) -> None:
        """Generate embedding for article summary."""
        if not self.generate_embeddings or not article.summary:
            return

        try:
            if self.embedding_provider == "openai":
                self._generate_embedding_openai(article)
            elif self.embedding_provider == "anthropic":
                self._generate_embedding_voyage(article)
            elif self.embedding_provider == "ollama":
                self._generate_embedding_ollama(article)
            elif self.embedding_provider == "gemini":
                self._generate_embedding_gemini(article)
        except Exception as e:
            print(f"Warning: Failed to generate embedding for '{article.title}': {e}")

    def _generate_embedding_openai(self, article: Article) -> None:
        """Generate embedding using OpenAI API."""
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=article.summary
        )
        article.embedding = response.data[0].embedding

    def _generate_embedding_voyage(self, article: Article) -> None:
        """Generate embedding using Voyage AI (Anthropic's embedding provider)."""
        response = self.embedding_client.embed(
            texts=[article.summary],
            model=self.embedding_model
        )
        article.embedding = response.embeddings[0]

    def _generate_embedding_ollama(self, article: Article) -> None:
        """Generate embedding using Ollama."""
        response = ollama.embed(
            model=self.embedding_model,
            input=article.summary
        )
        article.embedding = response['embeddings'][0]

    def _generate_embedding_gemini(self, article: Article) -> None:
        """Generate embedding using Google Gemini API."""
        response = self.embedding_client.models.embed_content(
            model=self.embedding_model,
            contents=article.summary
        )
        article.embedding = response.embeddings[0].values

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

    def process_document(self, txt_path: Path, root_dir: Path, skip_summary: bool = False,
                        skip_embeddings: bool = False, existing_json_path: Optional[Path] = None) -> Document:
        """Process a single document."""
        doc_start_time = time.time()

        # If skipping summary, load from existing JSON
        if skip_summary and existing_json_path:
            print(f"  Loading existing summary from: {existing_json_path}")
            document = self.load_document_from_json(existing_json_path, txt_path, root_dir)
            if not document:
                print(f"  Failed to load existing JSON, processing from scratch")
                skip_summary = False

        # If not skipping summary or load failed, process normally
        if not skip_summary:
            document = Document(txt_path, root_dir)

            # Read content
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract document metadata
            self.extract_document_metadata(content, document)

            # Extract articles
            extraction_start = time.time()
            articles = self.extract_articles(content, document)
            extraction_time = time.time() - extraction_start
            print(f"  Total extraction time: {extraction_time:.2f}s")

            # Process each article with AI
            print(f"\n  Generating summaries for {len(articles)} article(s)...")
            summary_start = time.time()
            for i, article in enumerate(articles, 1):
                article_start = time.time()
                print(f"  [{i}/{len(articles)}] {article.title[:60]}...", end=" ")
                self.process_with_ai(article, document)
                article_time = time.time() - article_start
                print(f"({article_time:.2f}s)")

            summary_time = time.time() - summary_start
            print(f"  Total summary generation time: {summary_time:.2f}s")

            document.articles = articles

        # Generate embeddings if enabled and not skipping
        if self.generate_embeddings and not skip_embeddings:
            print(f"\n  Generating embeddings for {len(document.articles)} article(s)...")
            embedding_start = time.time()
            for i, article in enumerate(document.articles, 1):
                print(f"  [{i}/{len(document.articles)}] {article.title[:60]}...", end=" ")
                self.generate_embedding(article)
                print("✓")

            embedding_time = time.time() - embedding_start
            print(f"  Total embedding generation time: {embedding_time:.2f}s")

        total_time = time.time() - doc_start_time
        print(f"  Document processing completed in {total_time:.2f}s")

        return document

    def generate_output(self, documents: List[Document], include_embeddings: bool = True) -> Dict:
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
                output["articles"].append(article.to_dict(doc.id, include_embedding=include_embeddings))

        return output

    def save_json(self, output: Dict, txt_path: Path, output_dir: Optional[Path] = None) -> None:
        """Save JSON output next to the text file or in specified output directory."""
        if output_dir:
            # Save in output directory with same filename
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / f"{txt_path.stem}.json"
            embeddings_path = output_dir / f"{txt_path.stem}-embeddings.json"
        else:
            # Save next to text file
            json_path = txt_path.with_suffix('.json')
            embeddings_path = txt_path.parent / f"{txt_path.stem}-embeddings.json"

        # Extract embeddings if present
        embeddings_output = None
        if self.generate_embeddings and output.get("articles"):
            embeddings_list = []
            for article in output["articles"]:
                if "embedding" in article:
                    embeddings_list.append({
                        "articleId": article["id"],
                        "vector": article["embedding"]
                    })
                    # Remove embedding from main output
                    del article["embedding"]

            if embeddings_list:
                embeddings_output = {"embeddings": embeddings_list}

        # Save main JSON (without embeddings)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # Save embeddings to separate file if they exist
        if embeddings_output:
            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_output, f, indent=2, ensure_ascii=False)
            print(f"Saved embeddings: {embeddings_path}")

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python tool that processes text files to extract articles, generate AI-powered summaries, and create structured JSON output. It supports both Anthropic Claude API and Ollama for local model inference.

## Critical: Virtual Environment Usage

**ALWAYS use the virtual environment when running Python commands:**

```bash
# Activate the venv first
. .venv/bin/activate

# Then run commands
python summarize_documents.py examples
pip install -e .
```

Or use inline activation:
```bash
. .venv/bin/activate && python summarize_documents.py examples
```

## Development Commands

### Setup and Installation
```bash
# Install package in editable mode
. .venv/bin/activate && pip install -e .

# Install dependencies
. .venv/bin/activate && pip install -r requirements.txt
```

### Running the Tool

**IMPORTANT: For development and testing, ALWAYS use Ollama with llama3.1 to avoid API costs!**

There are two entry points:

1. **Direct script** (for development/testing):
```bash
# PREFERRED: Use Ollama with llama3.1 (free, local inference - no API costs!)
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider ollama

# With custom model
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider ollama --model llama3.2

# With embeddings (using Ollama - recommended for testing)
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider ollama --generate-embeddings --embedding-provider ollama

# Use Gemini Flash 2.5 (fast and cheap)
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider gemini

# Only use Anthropic when specifically testing Claude API features
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider anthropic
```

2. **Installed CLI** (after `pip install -e .`):
```bash
# PREFERRED: Use Ollama
. .venv/bin/activate && document-summarizer examples/pdfs --model-provider ollama

# Use Gemini
. .venv/bin/activate && document-summarizer examples/pdfs --model-provider gemini

# With embeddings (OpenAI)
. .venv/bin/activate && document-summarizer examples/pdfs --model-provider ollama --generate-embeddings --embedding-provider openai

# Only use Anthropic when necessary
. .venv/bin/activate && document-summarizer examples/pdfs --model-provider anthropic
```

### Testing
```bash
# Run basic functionality test (no API key required)
. .venv/bin/activate && python test_basic.py

# Test with a single document using Ollama (recommended for testing)
. .venv/bin/activate && python summarize_documents.py examples/pdfs --pattern "Eph75_04.txt" --model-provider ollama

# Test with Gemini
. .venv/bin/activate && python summarize_documents.py examples/pdfs --pattern "Eph75_04.txt" --model-provider gemini
```

## Architecture

### Unified Code Structure

The project uses a **single implementation** in the package to avoid duplication:

1. **Main script**: `summarize_documents.py` (root directory)
   - Thin wrapper that imports and calls the package's CLI
   - Used for direct execution: `python summarize_documents.py`
   - No logic duplication - just imports from the package

2. **Package**: `src/document_summarizer/`
   - Contains ALL the logic: `cli.py`, `processor.py`, `__init__.py`
   - Used by both the main script and installed CLI: `document-summarizer`
   - Single source of truth for all functionality

**Important**: When adding features, implement them ONLY in `src/document_summarizer/`. The main script will automatically use them.

### Core Classes

- **Article**: Represents a section/article within a document
  - Contains: title, content, summary, categories, keywords, page range
  - Generates unique IDs from document ID + title slug

- **Document**: Represents a text file being processed
  - Tracks: title, issue date, PDF path, list of articles
  - Automatically finds corresponding PDF files

- **DocumentProcessor**: Main processing engine
  - Handles file discovery, article extraction, AI processing, and embedding generation
  - Supports three model providers: Anthropic (`anthropic`), Ollama (`ollama`), and Gemini (`gemini`)
  - Supports four embedding providers: OpenAI, Anthropic (Voyage AI), Ollama, and Gemini

### Article Detection Logic

**Multi-tier Article Segmentation Strategy**

Articles are identified using a three-tier approach with automatic fallback:

1. **Primary Method**: Structural parsing + LLM classification
   - Parses `#` headers into candidate sections with metadata (word count, page range, byline, preview)
   - `##` headers are kept as sub-sections within their parent `#` section
   - LLM classifies which sections are real articles vs. non-content (TOC, contacts, calendars, forms, footers)
   - LLM can merge sections split by page breaks (continuation fragments)
   - Token-efficient: sends section metadata (~2,500 tokens) instead of full document (~22,000 tokens)
   - Heuristic fallback when LLM is unavailable (filters by word count, title patterns, continuation detection)

2. **Fallback Method 1**: AI-based semantic analysis (when no markdown headers found)
   - Analyzes full document content to identify article boundaries
   - Understands context, topics, and natural section breaks
   - Avoids common pitfalls (e.g., numbered lists vs. headings)
   - Works with Anthropic Claude, Ollama, and Gemini models

3. **Fallback Method 2**: Page-based segmentation (when AI fails)
   - Splits by page markers: `[page 1]`, `Page 5`, `pg. 10`, `p. 3`, `<!-- PAGE N -->`, etc.
   - Creates one article per page
   - Ensures content is always processed even if everything else fails

4. **Post-processing**: Minimum length filter removes articles under 100 characters

### Model Provider Support

Three providers are supported for summarization:

- **Ollama** (RECOMMENDED for testing): Free local inference via `ollama` package
  - Default model: `llama3.1`
  - No API costs
  - Runs locally on your machine
  - Good for development and testing

- **Gemini** (fast and affordable): Google Gemini API via `google-genai` package
  - Default model: `gemini-2.0-flash-exp` (Flash 2.5)
  - Requires API key (low cost)
  - Fast inference
  - Good quality results

- **Anthropic** (highest quality): Claude API via `anthropic` package
  - Default model: `claude-sonnet-4-20250514`
  - Requires API key (higher cost)
  - Best quality results
  - Use for production or when you need highest quality

All providers use a unified `--model` argument to specify the model name.

Key methods:
- `_extract_articles_from_markdown()`: Structural parsing + LLM classification (primary)
- `_parse_markdown_sections()`: Parses `#` headers into `Section` objects with metadata
- `_classify_sections_with_llm()`: Sends section summaries to LLM for article classification
- `_classify_sections_heuristic()`: Heuristic fallback (word count, title patterns)
- `_call_llm_for_classification()`: Unified LLM dispatch for all providers
- `_build_articles_from_sections()`: Builds `Article` objects from classified sections
- `_extract_articles_with_ai_anthropic()`: Uses Claude to segment documents (fallback 1)
- `_extract_articles_with_ai_ollama()`: Uses Ollama to segment documents (fallback 1)
- `_extract_articles_with_ai_gemini()`: Uses Gemini to segment documents (fallback 1)
- `_extract_articles_by_page()`: Page-based segmentation (fallback 2)
- `_process_with_anthropic()`: Calls Anthropic API for summaries
- `_process_with_ollama()`: Calls Ollama API for summaries
- `_process_with_gemini()`: Calls Gemini API for summaries
- `generate_embedding()`: Generates embedding vector for article summary
- `_generate_embedding_openai()`: OpenAI embeddings
- `_generate_embedding_voyage()`: Anthropic/Voyage AI embeddings
- `_generate_embedding_ollama()`: Ollama embeddings
- `_generate_embedding_gemini()`: Gemini embeddings

### Page Marker Detection

Regex patterns recognize: `[page 1]`, `Page 5`, `pg. 10`, `p. 3`, `<!-- PAGE N -->`, etc.

## Environment Variables

Configure via `.env` file (see `.env.example`):

```bash
ANTHROPIC_API_KEY=your_api_key_here  # For Anthropic provider (summaries & Voyage embeddings)
GEMINI_API_KEY=your_gemini_key_here  # For Gemini provider (summaries & embeddings)
OPENAI_API_KEY=your_openai_key_here  # For OpenAI embeddings
```

## Embedding Generation

The tool supports generating vector embeddings for article summaries, useful for semantic search and similarity matching.

### Embedding Providers

Four providers are supported:

1. **OpenAI** (recommended for production)
   - Default model: `text-embedding-3-small` (1536 dimensions)
   - Requires `OPENAI_API_KEY` environment variable
   - High quality, cost-effective
   - Usage: `--generate-embeddings --embedding-provider openai`

2. **Ollama** (recommended for testing)
   - Default model: `embeddinggemma` (768 dimensions)
   - Free, runs locally
   - Good for development and testing
   - Requires Ollama to be running locally with an embedding model
   - Usage: `--generate-embeddings --embedding-provider ollama`

3. **Google Gemini**
   - Default model: `models/text-embedding-004` (768 dimensions)
   - Requires `GEMINI_API_KEY` environment variable
   - Competitive pricing, good quality
   - Usage: `--generate-embeddings --embedding-provider gemini`

4. **Anthropic/Voyage AI**
   - Default model: `voyage-3`
   - Uses same API key as Anthropic (for summaries)
   - Requires `voyageai` Python package
   - Usage: `--generate-embeddings --embedding-provider anthropic`

### Embedding CLI Options

```bash
--generate-embeddings              # Enable embedding generation
--embedding-provider {openai|anthropic|ollama|gemini}  # Choose provider (default: openai)
--embedding-model MODEL_NAME       # Override default model for provider
--openai-api-key KEY              # OpenAI API key (or use OPENAI_API_KEY env var)
--gemini-api-key KEY              # Gemini API key (or use GEMINI_API_KEY env var)
```

### Example Commands

```bash
# Test with Ollama embeddings (free, local)
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider ollama --generate-embeddings --embedding-provider ollama

# Use Gemini for both summaries and embeddings
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider gemini --generate-embeddings --embedding-provider gemini

# Production with Anthropic summaries and OpenAI embeddings
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider anthropic --generate-embeddings --embedding-provider openai

# Custom embedding model
. .venv/bin/activate && python summarize_documents.py examples/pdfs --model-provider ollama --generate-embeddings --embedding-provider openai --embedding-model text-embedding-3-large

# Embeddings-only mode: Generate embeddings from existing JSON files
. .venv/bin/activate && python summarize_documents.py examples/pdfs --embeddings-only --embedding-provider ollama
```

### Embeddings-Only Mode

If you already have JSON summary files and just want to generate embeddings (useful for batch processing or when changing embedding providers), use the `--embeddings-only` flag:

```bash
# Generate embeddings for all existing JSON files in a directory
. .venv/bin/activate && python summarize_documents.py examples/pdfs --embeddings-only --embedding-provider ollama

# Generate embeddings for a specific JSON file
. .venv/bin/activate && python summarize_documents.py examples/pdfs --pattern "Eph75_04.json" --embeddings-only --embedding-provider gemini
```

This mode:
- Skips text file processing and summary generation
- Reads existing JSON summary files
- Generates embeddings from the summaries
- Saves to separate `*-embeddings.json` files

## Output Format

### Summary JSON

Generates JSON with two top-level arrays:

- **documents**: Document-level metadata (id, title, issueDate, pdfPath, txtPath)
- **articles**: Article-level data (id, documentId, title, summary, categories, keywords, pageStart, pageEnd)

JSON files are generated:
- Next to each input file (default): `Eph75_04.txt` → `Eph75_04.json`
- Or as combined output with `--combined`: `combined_output.json`

### Embeddings JSON

When `--generate-embeddings` is enabled, embeddings are saved to a **separate file** to keep the main JSON clean and optimized for vector databases:

- **Per-document mode**: `Eph75_04-embeddings.json` next to `Eph75_04.json`
- **Combined mode**: `combined_output-embeddings.json` next to `combined_output.json`

Embeddings file schema:
```json
{
  "embeddings": [
    {
      "articleId": "eph-1975-04-01-rec-activities",
      "vector": [0.0123, -0.0876, 0.2345, ...]
    }
  ]
}
```

## Generated Files

The following files are generated and should be ignored:
- `*.json` (next to .txt files in examples/ and elsewhere)
- `*-embeddings.json` (embedding vectors, when `--generate-embeddings` is used)
- `__pycache__/`
- `.venv/`

See `.gitignore` for the full list.

## Dependencies

- `anthropic>=0.40.0` - Claude API client
- `python-dotenv>=1.0.0` - Environment variable management
- `ollama>=0.4.0` - Ollama local model client
- `openai>=1.0.0` - OpenAI API client (for embeddings)
- `voyageai>=0.2.0` - Voyage AI client (optional, for Anthropic embeddings)
- `google-genai>=1.0.0` - Google Gemini client (optional, for Gemini embeddings)

## Recent Improvements

1. **Embedding Generation** (2026-01): Vector embeddings for semantic search
   - Generate embeddings from article summaries
   - Three provider options: OpenAI, Anthropic/Voyage AI, Ollama
   - Ollama support enables free local embedding generation
   - Embeddings stored in JSON output for vector search integration

2. **Unified Architecture** (2026-01): Eliminated code duplication
   - Main script is now just a thin wrapper
   - All logic lives in the package (`src/document_summarizer/`)
   - Single source of truth prevents divergence

3. **Smart Article Detection** (2026-02): Structural parsing + LLM classification
   - Parses `#` headers into candidate sections with metadata (word count, byline, preview)
   - LLM classifies which sections are real articles vs. non-content (TOC, contacts, calendars, etc.)
   - Handles page-break continuations by merging split sections
   - Token-efficient: sends section metadata (~2,500 tokens) instead of full document
   - Heuristic fallback when LLM is unavailable
   - Replaced naive header-per-article approach that created ~40+ false articles from newsletters

4. **Multi-tier Fallback Strategy** (2026-01): Robust article segmentation
   - Structural parsing + LLM → AI semantic analysis → Page-based splitting
   - Ensures articles are always extracted, even on failures
   - Improved error handling for malformed AI responses

5. **Detailed Timing Output** (2026-01): Performance visibility
   - API call duration
   - Article extraction time
   - Summary generation time (per article and total)
   - Embedding generation time (if enabled)
   - Total document processing time

## Known Issues

1. **No automated tests**: Only has `test_basic.py` which is a manual integration test, not a proper test suite.

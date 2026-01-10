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

There are two entry points:

1. **Direct script** (for development/testing):
```bash
. .venv/bin/activate && python summarize_documents.py examples
. .venv/bin/activate && python summarize_documents.py examples --model-provider ollama
```

2. **Installed CLI** (after `pip install -e .`):
```bash
. .venv/bin/activate && document-summarizer examples
```

### Testing
```bash
# Run basic functionality test (no API key required)
. .venv/bin/activate && python test_basic.py
```

## Architecture

### Two Code Paths

The project has **two separate implementations** that need to be kept in sync:

1. **Main script**: `summarize_documents.py` (root directory)
   - Standalone script with all logic inline
   - Used for direct execution: `python summarize_documents.py`
   - Contains Ollama support

2. **Package**: `src/document_summarizer/`
   - Modular package structure: `cli.py`, `processor.py`, `__init__.py`
   - Used for installed CLI: `document-summarizer`
   - **Does NOT have Ollama support yet** - needs to be synced with main script

**Important**: When adding features, they need to be implemented in BOTH locations or the implementations will diverge.

### Core Classes

- **Article**: Represents a section/article within a document
  - Contains: title, content, summary, categories, keywords, page range
  - Generates unique IDs from document ID + title slug

- **Document**: Represents a text file being processed
  - Tracks: title, issue date, PDF path, list of articles
  - Automatically finds corresponding PDF files

- **DocumentProcessor**: Main processing engine
  - Handles file discovery, article extraction, AI processing
  - Supports two model providers: Anthropic (`anthropic`) and Ollama (`ollama`)

### Article Detection Logic

Articles are detected by identifying section headers using these heuristics (in `_is_section_header()`):

1. **All-caps lines** with 2+ words (e.g., "RECENT ACTIVITIES")
2. **Numbered sections** matching: `1.`, `I.`, `(a)` followed by capital letter
3. Can be customized based on document structure

### Model Provider Support

The main script (`summarize_documents.py`) supports two providers:

- **Anthropic** (default): Uses Claude API via `anthropic` package
- **Ollama** (local): Uses local Ollama server via `ollama` package

Key methods:
- `_process_with_anthropic()`: Calls Anthropic API
- `_process_with_ollama()`: Calls Ollama API with JSON format

### Page Marker Detection

Regex pattern recognizes: `[page 1]`, `Page 5`, `pg. 10`, `p. 3`, etc.

## Environment Variables

Configure via `.env` file (see `.env.example`):

```bash
ANTHROPIC_API_KEY=your_api_key_here  # For Anthropic provider
OLLAMA_HOST=http://localhost:11434  # Optional, for custom Ollama host
```

## Output Format

Generates JSON with two top-level arrays:

- **documents**: Document-level metadata (id, title, issueDate, pdfPath, txtPath)
- **articles**: Article-level data (id, documentId, title, summary, categories, keywords, pageStart, pageEnd)

JSON files are generated:
- Next to each input file (default): `Eph75_04.txt` → `Eph75_04.json`
- Or as combined output with `--combined`

## Generated Files

The following files are generated and should be ignored:
- `*.json` (next to .txt files in examples/ and elsewhere)
- `__pycache__/`
- `.venv/`

See `.gitignore` for the full list.

## Dependencies

- `anthropic>=0.40.0` - Claude API client
- `python-dotenv>=1.0.0` - Environment variable management
- `ollama>=0.4.0` - Ollama local model client

## Known Issues

1. **Two implementations diverging**: The main script has Ollama support, but the package version (`src/document_summarizer/`) does not. They need to be kept in sync manually.

2. **No automated tests**: Only has `test_basic.py` which is a manual integration test, not a proper test suite.

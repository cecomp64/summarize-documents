# Document Summarization Tool

A Python tool that processes text files to extract articles, generate AI-powered summaries, and create structured JSON output with metadata. Supports multiple AI providers (Anthropic Claude, Google Gemini, Ollama) and optional vector embedding generation.

## Features

- **Multi-provider AI support** - Choose between Anthropic Claude, Google Gemini, or Ollama (local) for summarization
- **Multi-tier article detection** - Markdown headers, AI-based semantic analysis, and page-based segmentation with automatic fallback
- **Recursive file discovery** with configurable glob patterns (default: `*.txt`)
- **AI-powered processing** - Generates concise summaries, categories, and keywords for each article
- **Vector embedding generation** - Optional embeddings via OpenAI, Gemini, Ollama, or Voyage AI for semantic search
- **Embeddings-only mode** - Generate embeddings from existing JSON summary files without reprocessing
- **Incremental processing** - Skips already-processed files unless `--force` is used
- **PDF linking** - Automatically finds and links corresponding PDF files
- **Flexible output** - Individual JSON files per document, combined output, or custom output directory
- **Detailed timing** - Performance metrics for API calls, extraction, summarization, and embedding generation

## Installation

### Option 1: Install from source (recommended for development)

```bash
git clone https://github.com/yourusername/document-summarizer.git
cd document-summarizer
pip install -e .
```

### Option 2: Install as a package

```bash
pip install document-summarizer
```

### Option 3: Install from local directory

```bash
cd /path/to/document-summarizer
pip install .
```

## Configuration

### API Keys

Set up API keys for your chosen provider(s). You can use environment variables or a `.env` file:

```bash
cp .env.example .env
```

| Variable | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude summaries, Voyage AI embeddings |
| `GEMINI_API_KEY` | Google Gemini summaries and embeddings |
| `OPENAI_API_KEY` | OpenAI embeddings |

Ollama requires no API key -- it runs locally.

## Usage

Once installed, use the `document-summarizer` command (or run `python summarize_documents.py` directly).

### Basic usage

Process all `.txt` files using the default provider (Anthropic):

```bash
document-summarizer /path/to/documents
```

### Choose a model provider

```bash
# Ollama - free, local inference (recommended for testing)
document-summarizer /path/to/documents --model-provider ollama

# Google Gemini - fast and affordable
document-summarizer /path/to/documents --model-provider gemini

# Anthropic Claude - highest quality (default)
document-summarizer /path/to/documents --model-provider anthropic
```

### Custom model

Override the default model for any provider:

```bash
document-summarizer /path/to/documents --model-provider ollama --model llama3.2
document-summarizer /path/to/documents --model-provider gemini --model gemini-2.5-pro
document-summarizer /path/to/documents --model-provider anthropic --model claude-sonnet-4-20250514
```

### Custom file pattern

```bash
document-summarizer /path/to/documents --pattern "*.md"
document-summarizer /path/to/documents --pattern "Eph*.txt"
```

### Output options

```bash
# Individual JSON files next to each source file (default)
document-summarizer /path/to/documents

# Individual JSON files in a custom output directory
document-summarizer /path/to/documents --output /path/to/output

# Single combined JSON file
document-summarizer /path/to/documents --combined --output combined.json
```

### With embeddings

```bash
# Ollama embeddings (free, local)
document-summarizer /path/to/documents --model-provider ollama --generate-embeddings --embedding-provider ollama

# Gemini for both summaries and embeddings
document-summarizer /path/to/documents --model-provider gemini --generate-embeddings --embedding-provider gemini

# OpenAI embeddings (default embedding provider)
document-summarizer /path/to/documents --generate-embeddings --embedding-provider openai

# Custom embedding model and dimensions
document-summarizer /path/to/documents --generate-embeddings --embedding-provider gemini --embedding-model gemini-embedding-001 --embedding-dimensions 512
```

### Embeddings-only mode

Generate embeddings from existing JSON summary files without reprocessing:

```bash
# Generate embeddings for all existing JSON files
document-summarizer /path/to/documents --embeddings-only --embedding-provider ollama

# Generate embeddings for a specific file
document-summarizer /path/to/documents --pattern "Eph75_04.json" --embeddings-only --embedding-provider gemini
```

### Force regeneration

By default, existing summary and embedding files are skipped. Use `--force` to regenerate:

```bash
document-summarizer /path/to/documents --force
document-summarizer /path/to/documents --embeddings-only --embedding-provider ollama --force
```

## Command-Line Reference

```
usage: document-summarizer [-h] [--version] [--pattern PATTERN]
                           [--api-key API_KEY]
                           [--model-provider {anthropic,ollama,gemini}]
                           [--model MODEL] [--combined] [--output OUTPUT]
                           [--generate-embeddings]
                           [--embeddings-only]
                           [--embedding-provider {openai,anthropic,ollama,gemini}]
                           [--embedding-model EMBEDDING_MODEL]
                           [--embedding-dimensions EMBEDDING_DIMENSIONS]
                           [--openai-api-key OPENAI_API_KEY]
                           [--gemini-api-key GEMINI_API_KEY]
                           [--force]
                           [directory]
```

| Argument | Description |
|---|---|
| `directory` | Directory to search for text files |
| `--version` | Show version and exit |
| `--pattern PATTERN` | Glob pattern for files to process (default: `*.txt`) |
| `--api-key API_KEY` | API key for model provider (or use env variables) |
| `--model-provider` | Model provider: `anthropic` (default), `ollama`, or `gemini` |
| `--model MODEL` | Model name (defaults: anthropic=`claude-sonnet-4-20250514`, ollama=`llama3.1`, gemini=`gemini-2.5-flash`) |
| `--combined` | Create a single combined JSON file instead of one per document |
| `--output OUTPUT` | Output path. With `--combined`: path to combined JSON file. Without: directory for individual JSON files |
| `--generate-embeddings` | Enable embedding generation for article summaries |
| `--embeddings-only` | Only generate embeddings from existing JSON files (skips summarization) |
| `--embedding-provider` | Embedding provider: `openai` (default), `anthropic` (Voyage AI), `ollama`, or `gemini` |
| `--embedding-model MODEL` | Override default embedding model (defaults: openai=`text-embedding-3-small`, anthropic=`voyage-3`, ollama=`embeddinggemma`, gemini=`gemini-embedding-001`) |
| `--embedding-dimensions N` | Output dimensionality for embeddings (default: 768). Supported by Gemini provider |
| `--openai-api-key KEY` | OpenAI API key for embeddings (or use `OPENAI_API_KEY` env var) |
| `--gemini-api-key KEY` | Gemini API key (or use `GEMINI_API_KEY` env var) |
| `--force` | Force regeneration of existing summary and embedding files |

## Model Providers

### Summarization

| Provider | Default Model | API Key | Notes |
|---|---|---|---|
| **Ollama** | `llama3.1` | None (local) | Free, runs locally. Best for development/testing |
| **Gemini** | `gemini-2.5-flash` | `GEMINI_API_KEY` | Fast, affordable. Good quality |
| **Anthropic** | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | Highest quality. Best for production |

### Embeddings

| Provider | Default Model | Dimensions | API Key |
|---|---|---|---|
| **OpenAI** | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` |
| **Ollama** | `embeddinggemma` | 768 | None (local) |
| **Gemini** | `gemini-embedding-001` | 768 (configurable) | `GEMINI_API_KEY` |
| **Anthropic/Voyage AI** | `voyage-3` | varies | `ANTHROPIC_API_KEY` |

## Article Detection

Articles are identified using a three-tier approach with automatic fallback:

1. **Markdown header parsing** (primary) - Detects `#` and `##` headers followed by capitalized text. Instant, no API calls.
2. **AI-based semantic analysis** (fallback) - Analyzes full document content to identify article boundaries using the configured model provider.
3. **Page-based segmentation** (last resort) - Splits by page markers (`[page 1]`, `Page 5`, `pg. 10`, `p. 3`, etc.). Creates one article per page.

A minimum length filter (100 characters) removes trivially short articles after extraction.

## Output Format

### Summary JSON

Generated next to each input file (e.g., `Eph75_04.txt` -> `Eph75_04.json`) or as combined output:

```json
{
  "documents": [
    {
      "id": "eph-1975-04",
      "title": "SAN JOSE AMATEUR ASTRONOMERS BULLETIN",
      "issueDate": "1975-04-01",
      "pdfPath": "pdfs/Eph75_04.pdf",
      "txtPath": "pdfs/Eph75_04.txt"
    }
  ],
  "articles": [
    {
      "id": "eph-1975-04-01-rec-activities",
      "documentId": "eph-1975-04",
      "title": "RECENT ACTIVITIES",
      "summary": "Report on the March 7, 1975 general meeting...",
      "categories": ["events", "observing reports"],
      "keywords": ["virgo cluster", "galaxies", "telescope"],
      "pageStart": 1,
      "pageEnd": 1
    }
  ]
}
```

### Embeddings JSON

When `--generate-embeddings` is enabled, embeddings are saved to a **separate file** (e.g., `Eph75_04-embeddings.json`):

```json
{
  "embeddings": [
    {
      "articleId": "eph-1975-04-01-rec-activities",
      "vector": [0.0123, -0.0876, 0.2345, "..."]
    }
  ]
}
```

## Working Without an API Key

If no API key is provided, the tool will still process documents with basic functionality:
- Summaries will be simple excerpts (first 200 characters)
- Categories will be set to `["uncategorized"]`
- Keywords will be empty

## Project Structure

```
document-summarizer/
├── src/
│   └── document_summarizer/
│       ├── __init__.py          # Package initialization
│       ├── processor.py         # Core processing logic
│       └── cli.py               # Command-line interface
├── summarize_documents.py       # Thin wrapper for direct execution
├── examples/
│   └── pdfs/                    # Example files
├── pyproject.toml               # Package configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── CLAUDE.md                    # Claude Code instructions
└── README.md                    # This file
```

The main script (`summarize_documents.py`) is a thin wrapper that imports from the package. All logic lives in `src/document_summarizer/`.

## Requirements

- Python 3.9+
- `anthropic>=0.40.0` - Claude API client
- `python-dotenv>=1.0.0` - Environment variable management
- `ollama>=0.4.0` - Ollama local model client
- `openai>=1.0.0` - OpenAI API client (for embeddings)
- `google-genai>=1.0.0` - Google Gemini client (optional)
- `voyageai>=0.2.0` - Voyage AI client (optional, for Anthropic embeddings)

## License

MIT License - Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

# Document Summarization Tool

A Python tool that processes text files to extract articles, generate AI-powered summaries, and create structured JSON output with metadata.

## Features

- **Recursive file discovery** with configurable glob patterns (default: `*.txt`)
- **Article extraction** - Automatically splits documents into sections/articles based on headers
- **Page tracking** - Detects and tracks page markers within documents
- **AI-powered processing** - Uses Claude API to generate:
  - Concise summaries
  - Relevant categories
  - Key keywords
- **PDF linking** - Automatically finds and links corresponding PDF files
- **Document metadata** - Extracts titles, dates, and other metadata
- **Flexible output** - Generate individual JSON files per document or a combined output

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

### Setting up the API key

Set up your Anthropic API key (optional, but required for AI-powered summaries):

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Or create a `.env` file in your working directory:

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

## Usage

Once installed, you can use the `document-summarizer` command from anywhere:

### Basic usage

Process all `.txt` files in a directory:

```bash
document-summarizer /path/to/documents
```

### Custom file pattern

Process files matching a specific pattern:

```bash
document-summarizer /path/to/documents --pattern "*.md"
```

### Combined output

Generate a single JSON file for all documents:

```bash
document-summarizer /path/to/documents --combined --output output.json
```

### Command-line options

```
usage: document-summarizer [-h] [--version] [--pattern PATTERN]
                           [--api-key API_KEY] [--model MODEL] [--combined]
                           [--output OUTPUT] [directory]

positional arguments:
  directory            Directory to search for text files

optional arguments:
  -h, --help           Show this help message and exit
  --version            Show version and exit
  --pattern PATTERN    Glob pattern for files to process (default: *.txt)
  --api-key API_KEY    Anthropic API key (or set ANTHROPIC_API_KEY env variable)
  --model MODEL        Claude model to use (default: claude-sonnet-4-20250514)
  --combined           Create a single combined JSON file instead of one per document
  --output OUTPUT      Output path for combined JSON (used with --combined)
```

## Output Format

The tool generates JSON files with the following schema:

```json
{
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
  ],
  "documents": [
    {
      "id": "eph-1975-04",
      "title": "SAN JOSE AMATEUR ASTRONOMERS BULLETIN",
      "issueDate": "1975-04-01",
      "pdfPath": "pdfs/Eph75_04.pdf",
      "txtPath": "pdfs/Eph75_04.txt"
    }
  ]
}
```

### Field Descriptions

**Articles:**
- `id` - Unique identifier derived from document ID and article title
- `documentId` - Reference to parent document
- `title` - Article title (extracted from section headers)
- `summary` - AI-generated summary (2-3 sentences)
- `categories` - AI-assigned categories
- `keywords` - AI-extracted keywords
- `pageStart` / `pageEnd` - Page numbers (if page markers found)

**Documents:**
- `id` - Unique identifier derived from filename
- `title` - Document title (extracted from content)
- `issueDate` - Publication date (extracted from filename or content)
- `pdfPath` - Relative path to corresponding PDF file (if found)
- `txtPath` - Relative path to source text file

## Article Detection

The tool automatically detects article boundaries using these heuristics:

1. **All-caps headers** - Lines in all capital letters with 2+ words
2. **Numbered sections** - Lines starting with "1.", "I.", "(a)", etc.
3. **Custom patterns** - Can be extended in the `_is_section_header()` method

## Page Marker Detection

The tool recognizes various page marker formats:

- `[page 1]`
- `[Page 5]`
- `pg. 10`
- `p. 3`
- `Page 15`

## PDF Detection

The tool automatically looks for PDF files with the same base name as each text file. For example:
- Text file: `documents/Eph75_04.txt`
- PDF file: `documents/Eph75_04.pdf`

PDF paths are stored as relative paths from the input root directory.

## Working Without an API Key

If no API key is provided, the tool will still process documents but with basic functionality:
- Summaries will be simple excerpts (first 200 characters)
- Categories will be set to `["uncategorized"]`
- Keywords will be empty

## Examples

### Example 1: Process astronomy bulletins

```bash
document-summarizer ./astronomy-docs --pattern "Eph*.txt"
```

### Example 2: Process all markdown files with combined output

```bash
document-summarizer ./articles --pattern "*.md" --combined --output all_articles.json
```

### Example 3: Process with explicit API key

```bash
document-summarizer ./documents --api-key sk-ant-xxxxx
```

### Example 4: Use a different Claude model

```bash
document-summarizer ./documents --model claude-3-5-haiku-20241022
```

## Customization

### Adjusting Article Detection

Edit the `_is_section_header()` method in [src/document_summarizer/processor.py](src/document_summarizer/processor.py) to customize how section headers are detected based on your document structure.

### Customizing AI Processing

Edit the `process_with_ai()` method to adjust:
- The prompt sent to Claude
- The model used (`claude-sonnet-4-20250514` by default)
- Token limits
- Output format

### Page Marker Patterns

Modify the `page_marker_pattern` regex in the `DocumentProcessor.__init__()` method to recognize different page marker formats.

## Using as a Python Library

You can also use the tool programmatically in your Python code:

```python
from document_summarizer import DocumentProcessor
from pathlib import Path

# Initialize processor
processor = DocumentProcessor(api_key="your-api-key")

# Process documents
root_dir = Path("/path/to/documents")
txt_files = processor.find_files(root_dir, "*.txt")

documents = []
for txt_path in txt_files:
    doc = processor.process_document(txt_path, root_dir)
    documents.append(doc)

# Generate output
output = processor.generate_output(documents)
print(output)
```

## Project Structure

```
document-summarizer/
├── src/
│   └── document_summarizer/
│       ├── __init__.py          # Package initialization
│       ├── processor.py         # Core processing logic
│       └── cli.py              # Command-line interface
├── examples/
│   └── pdfs/                   # Example files
├── pyproject.toml              # Package configuration
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # This file
└── QUICKSTART.md             # Quick start guide
```

## Requirements

- Python 3.9+
- `anthropic` - Claude API client
- `python-dotenv` - Environment variable management

## License

MIT License - Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues or questions, please open an issue on the GitHub repository.

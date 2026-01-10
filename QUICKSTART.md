# Quick Start Guide

Get up and running with the Document Summarization Tool in 5 minutes.

## 1. Install the Package

```bash
pip install -e .
```

This installs the package in "editable" mode, making the `document-summarizer` command available globally.

## 2. Set Up API Key (Optional)

For AI-powered summaries, you'll need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file in your working directory:

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

Get your API key from: https://console.anthropic.com/

**Note:** The tool works without an API key but will generate basic excerpts instead of AI summaries.

## 3. Try It Out

Run the tool on the example files:

```bash
document-summarizer examples
```

This will process the example text files in the `examples/pdfs/` directory and create JSON files next to each text file.

## 4. View Results

Check the generated JSON files:

```bash
cat examples/pdfs/Eph75_04.json
```

You should see structured output with articles, summaries, categories, and keywords.

## 5. Process Your Own Files

Point the tool at your own directory:

```bash
document-summarizer /path/to/your/documents
```

## Common Use Cases

### Process specific file types
```bash
document-summarizer ./docs --pattern "*.md"
```

### Generate combined output
```bash
document-summarizer ./docs --combined --output all_docs.json
```

### Process without API key
```bash
# Just run normally - it will fall back to basic excerpts
document-summarizer ./docs
```

### Check version
```bash
document-summarizer --version
```

## Example Output Structure

```json
{
  "articles": [
    {
      "id": "eph-1975-04-recent-activities",
      "documentId": "eph-1975-04",
      "title": "RECENT ACTIVITIES",
      "summary": "...",
      "categories": ["events", "observing reports"],
      "keywords": ["telescope", "astronomy"],
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

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize article detection in [src/document_summarizer/processor.py](src/document_summarizer/processor.py)
- Adjust the AI prompt for domain-specific processing

## Troubleshooting

**"No files found"**
- Check that your directory path is correct
- Verify files match the pattern (default: `*.txt`)
- Try specifying a custom pattern with `--pattern`

**"API key error"**
- Verify your API key is set in `.env` or via `--api-key`
- Check that the key is valid at https://console.anthropic.com/

**"No articles extracted"**
- Check that your documents have clear section headers
- Headers should be in ALL CAPS or numbered (e.g., "1. Section")
- You can customize detection in the `_is_section_header()` method

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review the example files in `examples/pdfs/`
- Open an issue on GitHub

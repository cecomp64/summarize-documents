# Contributing to Document Summarizer

Thank you for your interest in contributing to the Document Summarizer project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/document-summarizer.git
cd document-summarizer
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Project Structure

```
document-summarizer/
├── src/document_summarizer/    # Main package code
│   ├── __init__.py             # Package initialization
│   ├── processor.py            # Core processing logic
│   └── cli.py                  # Command-line interface
├── examples/                   # Example files for testing
├── tests/                      # Test files (to be added)
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
└── README.md                  # Documentation
```

## Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes to the code

3. Test your changes:
```bash
# Run on example files
document-summarizer examples

# Test specific functionality
python test_basic.py
```

4. Format your code (if using dev dependencies):
```bash
black src/
```

## Adding New Features

### Customizing Article Detection

The article detection logic is in [src/document_summarizer/processor.py](src/document_summarizer/processor.py). Key methods:

- `_is_section_header()` - Determines if a line is a section header
- `_split_into_sections()` - Splits content into sections
- `_parse_section()` - Extracts title, body, and page numbers

### Customizing AI Processing

The AI processing is handled in the `process_with_ai()` method. You can:

- Modify the prompt
- Change the model (default: `claude-3-5-sonnet-20241022`)
- Adjust token limits
- Change the output format

### Adding New Output Formats

To add new output formats (e.g., CSV, XML):

1. Add a new method in `DocumentProcessor` class
2. Add a CLI argument in `cli.py`
3. Update the main processing loop

## Testing

Before submitting a pull request:

1. Test on the example files:
```bash
document-summarizer examples
```

2. Test with custom patterns:
```bash
document-summarizer examples --pattern "*.txt"
```

3. Test combined output:
```bash
document-summarizer examples --combined --output test.json
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and single-purpose

## Submitting Changes

1. Commit your changes:
```bash
git add .
git commit -m "Add feature: description of your changes"
```

2. Push to your fork:
```bash
git push origin feature/your-feature-name
```

3. Create a Pull Request on GitHub

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Full error message and stack trace
- Minimal example to reproduce the issue
- Expected vs actual behavior

## Feature Requests

Feature requests are welcome! Please:

- Check if the feature already exists
- Describe the use case clearly
- Explain how it would benefit users
- Consider submitting a PR if you can implement it

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

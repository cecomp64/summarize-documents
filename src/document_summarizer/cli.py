"""
Command-line interface for the document summarizer.
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from .processor import DocumentProcessor


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Process text files and extract articles with summaries",
        prog="document-summarizer"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "directory",
        nargs="?",
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
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--model-provider",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="Model provider to use: 'anthropic' (default) or 'ollama' for local models"
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2). Used only with --model-provider=ollama"
    )
    parser.add_argument(
        "--ollama-host",
        help="Ollama host URL (e.g., http://localhost:11434). Used only with --model-provider=ollama"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create a single combined JSON file instead of one per document"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory or file path. For --combined: path to combined JSON file. Without --combined: directory to save individual JSON files (default: next to source files)"
    )

    args = parser.parse_args()

    # Handle version
    if args.version:
        from . import __version__
        print(f"document-summarizer {__version__}")
        return 0

    # Validate directory argument
    if not args.directory:
        parser.error("the following arguments are required: directory")
        return 1

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    # Check for required configuration based on provider
    if args.model_provider == "anthropic":
        if not api_key:
            print("Warning: No Anthropic API key provided. Summaries will be basic excerpts.")
            print("Set ANTHROPIC_API_KEY environment variable or use --api-key option.")
    elif args.model_provider == "ollama":
        print(f"Using Ollama with model: {args.ollama_model}")
        if args.ollama_host:
            print(f"Ollama host: {args.ollama_host}")

    # Get Ollama configuration from environment if not provided
    ollama_host = args.ollama_host or os.getenv("OLLAMA_HOST")

    # Initialize processor
    processor = DocumentProcessor(
        api_key=api_key,
        model=args.model,
        model_provider=args.model_provider,
        ollama_model=args.ollama_model,
        ollama_host=ollama_host
    )

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

    # Determine output directory for distributed files
    output_dir = None
    if args.output and not args.combined:
        # If output specified without --combined, treat as output directory
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving JSON files to: {output_dir}\n")

    # Process documents
    documents = []
    for txt_path in txt_files:
        doc = processor.process_document(txt_path, directory)
        documents.append(doc)

        # Save individual JSON unless combined output requested
        if not args.combined:
            output = processor.generate_output([doc])
            processor.save_json(output, txt_path, output_dir)

        print()

    # Save combined output if requested
    if args.combined:
        output = processor.generate_output(documents)
        output_path = args.output or directory / "combined_output.json"
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved combined output: {output_path}")

    print(f"\nProcessed {len(documents)} document(s) successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

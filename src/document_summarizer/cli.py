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
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate embeddings for article summaries"
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only generate embeddings from existing JSON files (skips summary generation)"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "anthropic", "ollama", "gemini"],
        default="openai",
        help="Embedding provider: 'openai' (default), 'anthropic' (Voyage AI), 'ollama', or 'gemini'"
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model to use. Defaults: openai=text-embedding-3-small, anthropic=voyage-3, ollama=embeddinggemma, gemini=models/text-embedding-004"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for embeddings (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Google Gemini API key for embeddings (or set GEMINI_API_KEY env variable)"
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

    # Get API keys for embeddings
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")

    # Handle embeddings-only mode
    if args.embeddings_only:
        args.generate_embeddings = True  # Enable embeddings
        print("Embeddings-only mode: Will read existing JSON files and generate embeddings")

    # Check embedding configuration
    if args.generate_embeddings:
        print(f"Embeddings enabled with provider: {args.embedding_provider}")
        if args.embedding_provider == "openai" and not openai_api_key:
            print("Warning: No OpenAI API key provided for embeddings.")
            print("Set OPENAI_API_KEY environment variable or use --openai-api-key option.")
            args.generate_embeddings = False
        elif args.embedding_provider == "anthropic" and not api_key:
            print("Warning: No Anthropic API key provided for Voyage embeddings.")
            args.generate_embeddings = False
        elif args.embedding_provider == "gemini" and not gemini_api_key:
            print("Warning: No Gemini API key provided for embeddings.")
            print("Set GEMINI_API_KEY environment variable or use --gemini-api-key option.")
            args.generate_embeddings = False
        elif args.embedding_provider == "ollama":
            embedding_model = args.embedding_model or "embeddinggemma"
            print(f"Using Ollama embeddings with model: {embedding_model}")

    # Initialize processor
    processor = DocumentProcessor(
        api_key=api_key,
        model=args.model,
        model_provider=args.model_provider,
        ollama_model=args.ollama_model,
        ollama_host=ollama_host,
        generate_embeddings=args.generate_embeddings,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key
    )

    # Find files
    directory = args.directory.resolve()
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1

    # Handle embeddings-only mode differently
    if args.embeddings_only:
        # Find JSON files instead of text files
        json_pattern = args.pattern.replace('.txt', '.json') if args.pattern.endswith('.txt') else '*.json'
        json_files = list(directory.rglob(json_pattern))

        # Filter out embedding files
        json_files = [f for f in json_files if not f.stem.endswith('-embeddings')]

        if not json_files:
            print(f"No JSON files matching '{json_pattern}' found in {directory}")
            return 1

        print(f"Found {len(json_files)} JSON file(s) to process\n")

        # Process each JSON file to generate embeddings
        processed_count = 0
        for json_path in json_files:
            result = processor.generate_embeddings_from_json(json_path)
            if result:
                processed_count += 1

        print(f"\nGenerated embeddings for {processed_count} file(s)!")
        return 0

    # Normal mode: find text files
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
            output = processor.generate_output([doc], include_embeddings=args.generate_embeddings)
            processor.save_json(output, txt_path, output_dir)

        print()

    # Save combined output if requested
    if args.combined:
        output = processor.generate_output(documents, include_embeddings=args.generate_embeddings)
        output_path = args.output or directory / "combined_output.json"
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract embeddings if present
        embeddings_output = None
        if args.generate_embeddings and output.get("articles"):
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

        # Save main combined JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved combined output: {output_path}")

        # Save combined embeddings to separate file if they exist
        if embeddings_output:
            embeddings_path = output_path.parent / f"{output_path.stem}-embeddings.json"
            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_output, f, indent=2, ensure_ascii=False)
            print(f"Saved combined embeddings: {embeddings_path}")

    print(f"\nProcessed {len(documents)} document(s) successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

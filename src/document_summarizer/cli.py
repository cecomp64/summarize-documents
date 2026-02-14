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
        help="API key for model provider (ANTHROPIC_API_KEY or GEMINI_API_KEY env variable)"
    )
    parser.add_argument(
        "--model-provider",
        choices=["anthropic", "ollama", "gemini"],
        default="anthropic",
        help="Model provider: 'anthropic' (default), 'ollama' (local), or 'gemini'"
    )
    parser.add_argument(
        "--model",
        help="Model name. Defaults: anthropic=claude-sonnet-4-20250514, ollama=llama3.1, gemini=gemini-2.0-flash-exp"
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
        help="Embedding model to use. Defaults: openai=text-embedding-3-small, anthropic=voyage-3, ollama=embeddinggemma, gemini=gemini-embedding-001"
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=768,
        help="Output dimensionality for embeddings (default: 768). Currently supported by Gemini provider."
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key for embeddings (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Google Gemini API key for embeddings (or set GEMINI_API_KEY env variable)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing summary and embedding files (default: skip existing files)"
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

    # Get API key based on provider
    if args.model_provider == "anthropic":
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: No Anthropic API key provided. Summaries will be basic excerpts.")
            print("Set ANTHROPIC_API_KEY environment variable or use --api-key option.")
    elif args.model_provider == "gemini":
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: No Gemini API key provided. Summaries will be basic excerpts.")
            print("Set GEMINI_API_KEY environment variable or use --api-key option.")
    else:  # ollama
        api_key = None

    # Set default model based on provider if not specified
    if not args.model:
        if args.model_provider == "anthropic":
            args.model = "claude-sonnet-4-20250514"
        elif args.model_provider == "ollama":
            args.model = "llama3.1"
        elif args.model_provider == "gemini":
            args.model = "gemini-2.5-flash"  # Gemini 2.5 Flash

    print(f"Using {args.model_provider} with model: {args.model}")

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
        generate_embeddings=args.generate_embeddings,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        embedding_dimensions=args.embedding_dimensions
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
            # Check if embeddings already exist (unless forcing)
            embeddings_path = json_path.parent / f"{json_path.stem}-embeddings.json"
            if embeddings_path.exists() and not args.force:
                print(f"\nSkipping: {json_path} (embeddings already exist)")
                processed_count += 1
                continue

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
        # Determine output paths for checking existence
        if output_dir:
            json_path = output_dir / f"{txt_path.stem}.json"
            embeddings_path = output_dir / f"{txt_path.stem}-embeddings.json"
        else:
            json_path = txt_path.with_suffix('.json')
            embeddings_path = txt_path.parent / f"{txt_path.stem}-embeddings.json"

        # Check if files already exist (only if not forcing regeneration)
        summary_exists = json_path.exists() and not args.force
        embeddings_exist = embeddings_path.exists() and args.generate_embeddings and not args.force

        # Determine what needs to be processed
        skip_summary = summary_exists
        skip_embeddings = embeddings_exist

        # If both exist, skip entirely
        if skip_summary and (skip_embeddings or not args.generate_embeddings):
            print(f"\nSkipping: {txt_path} (summary and embeddings already exist)")
            # Still need to load the document for combined output
            if args.combined:
                doc = processor.load_document_from_json(json_path, txt_path, directory)
                if doc:
                    documents.append(doc)
            continue

        # Print status message
        if skip_summary:
            print(f"\nProcessing: {txt_path} (summary exists, generating embeddings only)")
        elif skip_embeddings:
            print(f"\nProcessing: {txt_path} (embeddings exist, generating summary only)")
        else:
            print(f"\nProcessing: {txt_path}")

        # Process the document
        doc = processor.process_document(txt_path, directory,
                                         skip_summary=skip_summary,
                                         skip_embeddings=skip_embeddings,
                                         existing_json_path=json_path if skip_summary else None)
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

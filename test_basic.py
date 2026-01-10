#!/usr/bin/env python3
"""
Basic test script to verify the tool works without requiring an API key.
"""

import sys
from pathlib import Path
from summarize_documents import DocumentProcessor, Document


def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("Testing Document Summarization Tool\n")
    print("=" * 50)

    # Test 1: File discovery
    print("\n1. Testing file discovery...")
    processor = DocumentProcessor(api_key=None)
    examples_dir = Path("examples")

    if not examples_dir.exists():
        print("   ❌ Examples directory not found")
        return False

    files = processor.find_files(examples_dir, "*.txt")
    print(f"   ✓ Found {len(files)} text file(s)")

    if len(files) == 0:
        print("   ❌ No files found")
        return False

    # Test 2: Document processing
    print("\n2. Testing document processing...")
    for txt_path in files:
        print(f"   Processing: {txt_path.name}")

        doc = processor.process_document(txt_path, examples_dir)

        print(f"     - Document ID: {doc.id}")
        print(f"     - Title: {doc.title[:50]}...")
        print(f"     - Articles found: {len(doc.articles)}")
        print(f"     - PDF found: {doc.pdf_path or 'No'}")

        if len(doc.articles) == 0:
            print("     ⚠ Warning: No articles extracted")
        else:
            print(f"     ✓ Extracted {len(doc.articles)} article(s)")

            # Show first article
            first_article = doc.articles[0]
            print(f"       - First article: {first_article.title}")
            print(f"       - Page range: {first_article.page_start}-{first_article.page_end}")

    # Test 3: JSON generation
    print("\n3. Testing JSON generation...")
    all_docs = []
    for txt_path in files:
        doc = processor.process_document(txt_path, examples_dir)
        all_docs.append(doc)

    output = processor.generate_output(all_docs)

    print(f"   ✓ Generated output with:")
    print(f"     - {len(output['documents'])} document(s)")
    print(f"     - {len(output['articles'])} article(s)")

    # Test 4: Validate JSON structure
    print("\n4. Validating JSON structure...")
    required_doc_fields = ["id", "title", "issueDate", "pdfPath", "txtPath"]
    required_article_fields = ["id", "documentId", "title", "summary", "categories", "keywords"]

    for doc in output['documents']:
        for field in required_doc_fields:
            if field not in doc:
                print(f"   ❌ Missing field '{field}' in document")
                return False

    for article in output['articles']:
        for field in required_article_fields:
            if field not in article:
                print(f"   ❌ Missing field '{field}' in article")
                return False

    print("   ✓ All required fields present")

    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("\nNote: This test runs without an API key, so summaries are basic excerpts.")
    print("To test with AI-powered summaries, set ANTHROPIC_API_KEY in .env")

    return True


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

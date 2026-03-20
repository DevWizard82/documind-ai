from rag_pipeline import *

pdf_path = "linear_programming.pdf"

print("Step 1: Extracting text...")
text = extract_text_from_pdf(pdf_path)
print(f"Extracted {len(text.split())} words")

print("\nStep 2: Chunking...")
chunks = chunk_text(text)
print(f"Created {len(chunks)} chunks")

print("\nStep 3: Building FAISS index...")
index, chunks = build_index(chunks)

print("\nStep 4: Saving index...")
save_index(index, chunks)

print("\nStep 5: Test retrieval + answer...")
question = "What is this document about?"
relevant_chunks = retrieve(question, index, chunks)
answer = answer_question(question, relevant_chunks)
print(f"\nQ: {question}")
print(f"A: {answer}")
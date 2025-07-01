# --- build_store.py ---
# This is your one-time data processing tool.

import os
from dotenv import load_dotenv
import time

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index" # This is where the final database will be saved
SEARCH_DIRECTORIES = [".", "essays"]

def create_vector_store():
    """
    The main function to build and save the vector store.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("FATAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
        return

    # 1. Load all documents from the specified directories
    print("--- Starting Knowledge Base Build ---")
    print("Step 1: Loading all .txt documents...")
    all_docs = []
    for directory in SEARCH_DIRECTORIES:
        if not os.path.isdir(directory):
            continue
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            doc = Document(page_content=text, metadata={"source": filename})
                            all_docs.append(doc)
                    except Exception as e:
                        print(f"    - Error reading {file_path}: {e}")
    
    if not all_docs:
        print("Error: No documents found. Halting process.")
        return
        
    print(f"Successfully loaded {len(all_docs)} documents.")

    # 2. Split the loaded documents into smaller chunks
    print("\nStep 2: Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # 3. Initialize the embeddings model
    print("\nStep 3: Initializing Google Embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Build the FAISS index (This is the slow, API-heavy part)
    print("\nStep 4: Creating FAISS index from chunks...")
    print("This will make many API calls and may take several minutes.")
    
    # We will process in batches to respect rate limits
    vector_store = FAISS.from_documents(documents=chunked_docs[:50], embedding=embeddings) # Start with the first 50
    print("Initial batch processed...")

    for i in range(50, len(chunked_docs), 50):
        batch = chunked_docs[i:i + 50]
        vector_store.add_documents(documents=batch, embedding=embeddings)
        print(f"Processed batch {i // 50 + 1}... Pausing for 60 seconds to respect rate limits.")
        time.sleep(60) # Wait 60 seconds between batches of 50

    # 5. Save the completed index to disk
    print("\nStep 5: Saving final index to disk...")
    vector_store.save_local(FAISS_INDEX_PATH)
    print("\n--- Build Complete! ---")
    print(f"Vector store has been saved to the '{FAISS_INDEX_PATH}' folder.")

if __name__ == "__main__":
    create_vector_store()

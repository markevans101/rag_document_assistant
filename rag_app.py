print("Starting RAG system...")
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize components
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings initialized!")

# Load document
print("\nLoading documents...")
loader = DirectoryLoader("documents", glob="**/*.txt")
documents = loader.load()
print(f"Loaded {len(documents)} documents!")

choice = input("\nPress Enter to see document content...")
if documents:
    print("\nDocument content:")
    print(documents[0].page_content)

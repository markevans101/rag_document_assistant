from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np

class EmbeddingsRAG:
    def __init__(self):
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        print("Initialization complete!")
    
    def add_document(self, name, content):
        """Add a document using embeddings"""
        print(f"Processing document: {name}")
        texts = [content]  # In a real system, we'd split long documents
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_texts(
                texts,
                self.embeddings,
                persist_directory="vectorstore"
            )
        else:
            self.vector_store.add_texts(texts)
            
        return f"Added document: {name}"
    
    def query(self, question, k=2):
        """Query using embeddings similarity"""
        if self.vector_store is None:
            return ["No documents added yet"]
            
        # Get similar documents
        docs = self.vector_store.similarity_search(question, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append(f"Found relevant text: {doc.page_content}")
            
        return results if results else ["No relevant documents found"]

def main():
    # Initialize our RAG system
    print("Starting up...")
    rag = EmbeddingsRAG()
    
    # Sample document
    test_content = """This is a sample document for testing RAG.
    The weather is sunny today.
    Machine learning is an interesting field.
    Artificial intelligence is transforming many industries.
    Neural networks can solve complex problems.
    The temperature is 75 degrees Fahrenheit."""
    
    print("\n=== Embeddings RAG System ===")
    
    while True:
        print("\nOptions:")
        print("1. Add document")
        print("2. Query documents")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            print(rag.add_document("test_doc", test_content))
            
        elif choice == "2":
            question = input("Enter your question: ")
            print("\nSearching...")
            results = rag.query(question)
            print("\nResults:")
            for result in results:
                print(result)
                
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

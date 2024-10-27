from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding

class LlamaRAG:
    def __init__(self):
        print("Initializing LlamaIndex RAG system...")
        # Set up embeddings
        self.embed_model = 
HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model)
        self.index = None
        print("Initialization complete!")

    def add_document(self, text_content):
        """Add a document to the index"""
        print("Processing document...")
        try:
            # Create documents
            documents = [Document(text=text_content)]
            
            # Create or update index
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context
            )
            print("Document processed successfully!")
            return "Document added successfully"
        except Exception as e:
            return f"Error adding document: {str(e)}"

    def query(self, question):
        """Query the index"""
        if self.index is None:
            return ["No documents added yet"]
        
        try:
            query_engine = self.index.as_query_engine()
            response = query_engine.query(question)
            return [str(response)]
        except Exception as e:
            return [f"Error during query: {str(e)}"]

def main():
    # Initialize RAG system
    rag = LlamaRAG()
    
    # Sample document
    test_content = """
    This is a sample document for testing RAG.
    The weather is sunny today.
    Machine learning is an interesting field.
    Artificial intelligence is transforming many industries.
    """
    
    print("\n=== LlamaIndex RAG System ===")
    
    while True:
        print("\nOptions:")
        print("1. Add document")
        print("2. Query documents")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            result = rag.add_document(test_content)
            print(result)
            
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

import os
from langchain_community.document_loaders import DirectoryLoader, 
TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

class SimpleRAG:
    def __init__(self, docs_dir="documents", db_dir="vectordb"):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = 
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = 
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        os.makedirs(self.docs_dir, exist_ok=True)
        
    def ingest_documents(self):
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", 
loader_cls=TextLoader)
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(documents=splits, 
embedding=self.embeddings, persist_directory=self.db_dir)
        self.vectorstore.persist()
        return len(splits)
    
    def query(self, question: str, k: int = 4) -> str:
        if not hasattr(self, 'vectorstore'):
            self.vectorstore = Chroma(persist_directory=self.db_dir, 
embedding_function=self.embeddings)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Retrieved context:\n{context}"

def main():
    rag = SimpleRAG()
    while True:
        print("\nSimple RAG System")
        print("1. Ingest documents")
        print("2. Query")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            try:
                num_chunks = rag.ingest_documents()
                print(f"Successfully processed documents into {num_chunks} 
chunks")
            except Exception as e:
                print(f"Error ingesting documents: {str(e)}")
        elif choice == "2":
            question = input("Enter your question: ")
            try:
                answer = rag.query(question)
                print("\nAnswer:", answer)
            except Exception as e:
                print(f"Error querying the system: {str(e)}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

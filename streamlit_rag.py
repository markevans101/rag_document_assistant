import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class SimpleRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len, separators=["\n\n", "\n", ".", "!"])
        self.vector_store = None

    def add_document(self, content):
        if not content.strip():
            return "Please add some text first"
        texts = self.text_splitter.split_text(content)
        if self.vector_store is None:
            self.vector_store = Chroma.from_texts(texts, self.embeddings, persist_directory="vectorstore")
        else:
            self.vector_store.add_texts(texts)
        return f"✅ Added document successfully! Created {len(texts)} chunks."

    def query(self, question, k=2):
        if self.vector_store is None:
            return ["Please add a document first."]
        if not question.strip():
            return ["Please enter a question."]
        docs = self.vector_store.similarity_search(question, k=k)
        return [doc.page_content for doc in docs]

@st.cache_resource
def get_rag_system():
    return SimpleRAG()

st.title("📚 RAG Document Assistant")

with st.sidebar:
    st.header("📄 Add New Document")
    doc_text = st.text_area("Paste your document here:", height=300, help="Paste any text document you want to query later.")
    if st.button("Add Document", type="primary"):
        if doc_text.strip():
            rag = get_rag_system()
            result = rag.add_document(doc_text)
            st.success(result)
        else:
            st.error("Please add some text first!")

st.header("❓ Ask Questions")

query = st.text_input("Enter your question:", help="Ask anything about the documents you added")
col1, col2 = st.columns(2)
with col1:
    k_value = st.slider("Number of results:", min_value=1, max_value=5, value=2, help="How many relevant passages to return")
with col2:
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                rag = get_rag_system()
                results = rag.query(query, k=k_value)
                st.header("📝 Results:")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i}"):
                        st.markdown(result)
        else:
            st.warning("Please enter a question!")

with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Add your document** in the sidebar
    2. **Ask questions** about the content
    3. **Adjust** the number of results
    4. **Explore** different parts of your document
    """)
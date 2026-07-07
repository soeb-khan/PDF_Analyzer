import streamlit as st
import os
from rag_pipeline import load_and_split_pdf, create_vectorstore, create_qa_chain

st.set_page_config(page_title="InsightPDF", page_icon="📄", layout="centered")

# --- Custom styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📄 InsightPDF</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about your PDF and get instant, source-backed answers.</p>', unsafe_allow_html=True)

# --- Sidebar for upload + settings ---
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    st.divider()
    st.caption("Built with LangChain, FAISS, and Groq (Llama 3.3)")

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Reading and indexing your document..."):
            chunks = load_and_split_pdf(temp_path, chunk_size=1000, chunk_overlap=100)
            vectorstore = create_vectorstore(chunks)
            st.session_state.qa_chain = create_qa_chain(vectorstore, num_chunks=len(chunks))
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # reset chat on new file
        st.toast(f"✅ {uploaded_file.name} indexed successfully!")

    # --- Display chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 View sources"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f'<div class="source-box"><b>Source {i+1} (Page {src["page"]})</b><br>{src["text"]}</div>',
                            unsafe_allow_html=True
                        )

    # --- Chat input ---
    query = st.chat_input("Ask something about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke({"query": query})
            st.write(response["result"])

            sources = [
                {"page": doc.metadata.get("page", "N/A"), "text": doc.page_content[:200] + "..."}
                for doc in response["source_documents"]
            ]
            with st.expander("📚 View sources"):
                for i, src in enumerate(sources):
                    st.markdown(
                        f'<div class="source-box"><b>Source {i+1} (Page {src["page"]})</b><br>{src["text"]}</div>',
                        unsafe_allow_html=True
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["result"],
            "sources": sources
        })

else:
    st.info("👈 Upload a PDF from the sidebar to get started.")

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # loads GROQ_API_KEY from .env


def load_and_split_pdf(pdf_path, chunk_size=2000, chunk_overlap=0):
    """
    Loads a PDF and splits it into chunks.

    Args:
        pdf_path: path to the PDF file
        chunk_size: max characters per chunk
        chunk_overlap: overlapping characters between chunks (helps preserve context across chunk boundaries)

    Returns:
        list of chunked Document objects
    """
    # Step 1: Load the PDF — this returns one Document object per page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from PDF")

    # Step 2: Split pages into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    return chunks


def create_vectorstore(chunks):
    """
    Converts chunks into embeddings and stores them in a FAISS vector store.

    Args:
        chunks: list of Document chunks (from load_and_split_pdf)

    Returns:
        FAISS vectorstore object
    """
    # Free, local embedding model — no API key needed for this part
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # This embeds every chunk and stores the vectors in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vectorstore created successfully")

    return vectorstore


def create_qa_chain(vectorstore, num_chunks):
    """
    Creates a RetrievalQA chain, adapting retrieval size based on document length.
    """
    # Small docs: pull in everything (better for "how many/list all" questions)
    # Large docs: cap retrieval so we don't blow context window / cost
    k = min(num_chunks, 10)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )

    return qa_chain


if __name__ == "__main__":
    test_pdf_path = "data/Prepisely.pdf"
    chunks = load_and_split_pdf(test_pdf_path)
    vectorstore = create_vectorstore(chunks)

    # NEW: create the QA chain
    qa_chain = create_qa_chain(vectorstore, num_chunks=len(chunks))

    query = "What programming languages does this person know?"
    response = qa_chain.invoke({"query": query})

    print("\n--- Answer ---")
    print(response["result"])

    print("\n--- Source chunks used ---")
    for doc in response["source_documents"]:
        print(f"- Page {doc.metadata['page']}: {doc.page_content[:100]}...")

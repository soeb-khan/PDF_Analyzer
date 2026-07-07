# 📄 PDF_Analyzer

A RAG-based document Q&A system that lets you upload one or more PDFs and ask natural-language questions, with source-cited answers.

## Features
- Upload multiple PDFs and query across all of them
- Semantic search using FAISS vector store + HuggingFace embeddings
- Answer generation using Groq (Llama 3.3 70B)
- Source citations showing which document/page an answer came from
- Chat-style interface built with Streamlit

## Tech Stack
- LangChain (document loading, text splitting, chains)
- FAISS (vector store for semantic search)
- HuggingFace sentence-transformers (embeddings)
- Groq API (LLM inference)
- Streamlit (UI)

## How it works
1. PDF(s) are loaded and split into overlapping chunks
2. Each chunk is embedded and stored in a FAISS vector store
3. User's question is embedded and matched against stored chunks (semantic similarity)
4. Top relevant chunks are passed to the LLM to generate a grounded answer
5. Answer is shown along with the source document/page it came from

## Setup
```bash
git clone <your-repo-url>
cd insightPDF
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

## Known limitations
- Works best with text-based PDFs; scanned/image-only PDFs need OCR (not yet implemented)
- Aggregation-style questions (e.g., "how many X") work best on smaller/larger documents fitted into a single context window

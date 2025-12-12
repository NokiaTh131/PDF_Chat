# PDF Chat

![License](https://img.shields.io/badge/license-MIT-blue.svg)

**PDF Chat** is an AI-powered application that allows you to have natural conversations with your PDF documents. Upload a file, ask questions, and get instant, accurate answers based on the document's content.

## Features

- **interactive Chat Interface**: Seamless chat experience similar to ChatGPT.
- **Document Ingestion**: Upload PDF files which are automatically parsed, chunked, and embedded.
- **Context-Aware Answers**: Uses Retrieval-Augmented Generation (RAG) to answer questions specifically from your document.

## Tech Stack

**Frontend:**
* [Streamlit]
* [Lucide React / React Icons]

**Backend & AI:**
* [Python]
* **LLM Framework:** [LangChain / LlamaIndex]
* **Model Provider:** [Groq]
* **Vector Database:** [FAISS]

## Architecture

The application follows a standard RAG pipeline:
1.  **Ingest**: The PDF text is extracted and split into manageable chunks.
2.  **Embed**: Chunks are converted into vector embeddings using an embedding model (e.g., `text-embedding-3-small`).
3.  **Store**: Vectors are stored in a vector database for efficient semantic search.
4.  **Retrieve**: User queries are embedded, and relevant chunks are fetched from the database.
5.  **Generate**: The LLM generates an answer using the retrieved chunks as context.

## 🏁 Getting Started

### Prerequisites
* Python (v3.10+)
* API Keys for [Groq]

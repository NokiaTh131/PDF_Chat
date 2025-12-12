# PDF Chat

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)

**PDF Chat** is an AI-powered application that allows you to have natural conversations with your PDF documents. Upload a file, ask questions, and get instant, accurate answers based on the document's content.

## Features

- **interactive Chat Interface**: Seamless chat experience similar to ChatGPT.
- **Document Ingestion**: Upload PDF files which are automatically parsed, chunked, and embedded.
- **Context-Aware Answers**: Uses Retrieval-Augmented Generation (RAG) to answer questions specifically from your document.
- **Source Citations**: Returns the specific page numbers or sections used to generate the answer.
- **Multi-File Support**: [Optional: Mention if it supports multiple PDFs at once].
- **Secure & Private**: [Optional: Mention if data is stored locally or processed securely].

## Tech Stack

**Frontend:**
* [React / Next.js / React Native]
* [Tailwind CSS / Styled Components]
* [Lucide React / React Icons]

**Backend & AI:**
* [Node.js / Python / Go]
* **LLM Framework:** [LangChain / LlamaIndex]
* **Model Provider:** [OpenAI GPT-4 / Anthropic Claude / Ollama (Local)]
* **Vector Database:** [Pinecone / Supabase pgvector / ChromaDB / FAISS]
* **PDF Processing:** [PDF.js / PyPDF2 / Unstructured]

## Architecture


The application follows a standard RAG pipeline:
1.  **Ingest**: The PDF text is extracted and split into manageable chunks.
2.  **Embed**: Chunks are converted into vector embeddings using an embedding model (e.g., `text-embedding-3-small`).
3.  **Store**: Vectors are stored in a vector database for efficient semantic search.
4.  **Retrieve**: User queries are embedded, and relevant chunks are fetched from the database.
5.  **Generate**: The LLM generates an answer using the retrieved chunks as context.

## 🏁 Getting Started

### Prerequisites
* Node.js (v18+) or Python (v3.10+)
* [Docker (if applicable)]
* API Keys for [OpenAI/Pinecone/etc.]

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/NokiaTh131/PDF_Chat.git](https://github.com/NokiaTh131/PDF_Chat.git)
    cd PDF_Chat
    ```

2.  **Install dependencies**
    ```bash
    # For Node/Next.js
    npm install
    # OR for Python
    # pip install -r requirements.txt
    ```

3.  **Environment Setup**
    Create a `.env` file in the root directory and add your keys:
    ```env
    # Example .env configuration
    OPENAI_API_KEY=sk-...
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=pdf-chat
    DATABASE_URL=...
    ```

4.  **Run the Application**
    ```bash
    # Development mode
    npm run dev
    # OR
    # streamlit run app.py
    ```

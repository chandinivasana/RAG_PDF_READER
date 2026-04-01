# RAG PDF Reader

A robust Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, index them into a vector database, and ask questions against the content using Large Language Models.

##  Features

- **Asynchronous Processing:** Uses Inngest to handle long-running PDF ingestion and LLM query workflows reliably.
- **Smart Chunking:** Leverages LlamaIndex's `SentenceSplitter` for context-aware document partitioning.
- **High-Performance Vector Search:** Utilizes Qdrant for efficient similarity search of document embeddings.
- **Modern UI:** Clean Streamlit interface for seamless document management and chat interactions.
- **Advanced Embeddings:** Powered by OpenAI's `text-embedding-3-large` for high-dimensional semantic understanding.

##  Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration:** [Inngest](https://www.inngest.com/)
- **Vector Database:** [Qdrant](https://qdrant.tech/)
- **LLM & Embeddings:** [OpenAI](https://openai.com/)
- **Data Framework:** [LlamaIndex](https://www.llamaindex.ai/)

##  Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10+
- [Docker](https://www.docker.com/) 
- [Inngest CLI](https://www.inngest.com/docs/local-development)
- [uv](https://github.com/astral-sh/uv) 

##  Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd RAG_PDF_READER
```

### 2. Install Dependencies
Using `uv`:
```bash
uv sync
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
INNGEST_EVENT_KEY=  # skip if local
INNGEST_SIGNING_KEY= # skip..!
```

### 4. Start Infrastructure
Start the Qdrant vector database:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Start the Inngest Dev Server:
```bash
inngest dev
```

##  Running the Application

Follow these steps in order to get the system up and running:

### 1. Run Qdrant
If using Docker, start the vector database:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Run the Backend
Start the FastAPI application:
```bash
uv run uvicorn main:app --reload --port 8000
```

### 3. Run Inngest Dev Server
In a separate terminal, start the Inngest development server pointing to your backend:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

### 4. Run the Streamlit App
In another terminal, launch the user interface:
```bash
uv run streamlit run streamlit_app.py
```

## SpoonFeeding?!-->

1. **Upload:** Open the Streamlit app (usually at `http://localhost:8501`) and upload a PDF. This triggers the ingestion workflow.
2. **Ingestion:** Inngest will asynchronously handle the chunking and embedding. You can monitor progress in the Inngest Dev Server UI.
3. **Query:** Once ingested, enter your question in the text input. The system will retrieve relevant context from Qdrant and generate an answer using the LLM.

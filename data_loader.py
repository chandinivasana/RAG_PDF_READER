from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client = OpenAI(timeout=60.0)
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str) -> list[str]:
    print(f"--- Loading PDF: {path} ---")
    # Convert string path to Path object to be safe
    pdf_file = Path(path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found at {path}")

    docs = PDFReader().load_data(file=pdf_file)
    texts = [d.text for d in docs if hasattr(d, 'text') and d.text]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    print(f"--- Extracted {len(chunks)} chunks ---")
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    print(f"--- Embedding {len(texts)} texts ---")
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [r.embedding for r in response.data]

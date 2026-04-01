import logging
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
import inngest
import inngest.fast_api

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

openai_client = OpenAI(timeout=60.0)

inngest_client = inngest.Inngest(
    app_id="rag-pdf-app",
    logger=logger,
    is_production=False,
)

@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    step = ctx.step

    def _load() -> dict:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        print(f"[{datetime.now()}] Step 1: Loading PDF from {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)
        return {"chunks": chunks, "source_id": source_id}

    def _upsert(chunks_and_src: dict) -> dict:
        chunks = chunks_and_src["chunks"]
        source_id = chunks_and_src["source_id"]
        print(f"[{datetime.now()}] Step 2: Embedding and Upserting {len(chunks)} chunks for {source_id}")
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        store = QdrantStorage()
        store.upsert(ids, vecs, payloads)
        print(f"[{datetime.now()}] Success: Ingested {len(chunks)} chunks.")
        return {"ingested": len(chunks)}

    chunks_and_src = await step.run("load-and-chunk", _load)
    ingested = await step.run("embed-and-upsert", lambda: _upsert(chunks_and_src))
    return ingested

@inngest_client.create_function(
    fn_id="rag-query-pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf(ctx: inngest.Context):
    step = ctx.step

    def _search() -> dict:
        question = ctx.event.data.get("question")
        top_k = ctx.event.data.get("top_k", 5)
        print(f"[{datetime.now()}] Step 1: Searching for context for: {question}")

        query_vecs = embed_texts([question])
        if not query_vecs:
            raise ValueError("Failed to embed question")

        store = QdrantStorage()
        found = store.search(query_vecs[0], top_k)
        print(f"[{datetime.now()}] Found {len(found['contexts'])} relevant contexts.")
        return found

    question = ctx.event.data.get("question")
    found = await step.run("embed-and-search", _search)

    def _generate_answer() -> str:
        print(f"[{datetime.now()}] Step 2: Generating LLM Answer...")
        context_block = "\n\n".join(f"- {c}" for c in found.get("contexts", []))

        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context provided."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the question using the context provided only."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    answer = await step.run("llm-answer", _generate_answer)
    print(f"[{datetime.now()}] Answer generation complete.")
    return {"answer": answer, "sources": found.get("sources", [])}

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "app": "rag-pdf-reader"}

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf],
)

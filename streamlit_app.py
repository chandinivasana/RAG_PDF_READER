from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag-pdf-app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        send_rag_ingest_event(path)
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


def send_rag_query_event(question: str, top_k: int) -> str | None:
    client = get_inngest_client()
    result = client.send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0] if result else None


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        st.error(f"Error fetching runs: {e}")
        return []


def wait_for_run_output(event_id: str, timeout_s: float = 300.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output for event {event_id} "
                f"(last status: {last_status})"
            )
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question_input = st.text_input("Your question")
    top_k_input = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question_input.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = send_rag_query_event(question_input.strip(), int(top_k_input))
            if event_id:
                # Poll the local Inngest API for the run's output
                try:
                    output = wait_for_run_output(event_id)
                    answer = output.get("answer", "")
                    sources = output.get("sources", [])

                    st.subheader("Answer")
                    st.write(answer or "(No answer)")
                    if sources:
                        st.caption("Sources")
                        for s in sources:
                            st.write(f"- {s}")
                except Exception as e:
                    st.error(f"Error getting answer: {e}")
            else:
                st.error("Failed to send event to Inngest.")

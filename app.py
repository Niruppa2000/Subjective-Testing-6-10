import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# ----------------------------
# CONFIG
# ----------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

# ----------------------------
# PDF & TEXT UTILITIES
# ----------------------------
def extract_text_from_pdf_filelike(file) -> str:
    # file is a BytesIO-like object from st.file_uploader
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def build_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------------------
# EMBEDDINGS + INDEX
# ----------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    return device, embedder, tokenizer, gen_model


def build_index_from_files(uploaded_files, chunk_size, overlap):
    docs = []     # list of {"doc_id", "chunk_id", "text"}
    for f in uploaded_files:
        raw_text = extract_text_from_pdf_filelike(f)
        chunks = build_chunks(raw_text, chunk_size, overlap)
        for idx, ch in enumerate(chunks):
            docs.append(
                {
                    "doc_id": f.name,
                    "chunk_id": idx,
                    "text": ch,
                }
            )
    return docs


def build_faiss_index(docs, embedder):
    emb_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(emb_dim)

    vectors = []
    for d in docs:
        vec = embedder.encode(d["text"], convert_to_numpy=True, show_progress_bar=False)
        vectors.append(vec)

    vectors = np.vstack(vectors).astype("float32")
    index.add(vectors)
    return index, vectors


def retrieve_context(query: str, index, embedder, docs, top_k: int = TOP_K):
    q_vec = embedder.encode(query, convert_to_numpy=True).astype("float32")
    q_vec = np.expand_dims(q_vec, axis=0)
    distances, indices = index.search(q_vec, top_k)
    indices = indices[0]
    retrieved = [docs[i] for i in indices]
    return retrieved


def build_prompt(retrieved_chunks, topic: str, target_class: int, num_questions: int = 5):
    context_text = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
You are an experienced NCERT school teacher for Class {target_class}.
Using ONLY the context from the NCERT textbook below, generate {num_questions} HIGH-QUALITY, long-answer subjective questions.

Requirements:
- Questions should match the difficulty and style of Class {target_class} NCERT board exam questions.
- Focus on understanding, explanation, and reasoning (no one-word answers).
- Do NOT provide answers, only questions.
- Number the questions clearly as 1., 2., 3., ...

Topic: {topic}

CONTEXT:
{context_text}
"""
    return prompt.strip()


def generate_subjective_questions(
    topic: str,
    target_class: int,
    num_questions: int,
    index,
    docs,
    embedder,
    tokenizer,
    gen_model,
    device,
    max_new_tokens: int = 420,
):
    retrieved = retrieve_context(topic, index, embedder, docs, top_k=TOP_K)
    prompt = build_prompt(retrieved, topic, target_class, num_questions)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text, retrieved


# ----------------------------
# STREAMLIT UI
# ----------------------------
def main():
    st.set_page_config(page_title="NCERT Subjective Question Generator", layout="wide")
    st.title("📚 NCERT Subjective Question Generator (Classes 6–10)")

    st.markdown(
        """
Upload NCERT **History / Civics / Geography / Science** PDFs for Classes 6–10 and generate
**high-quality long-answer questions** similar to board exam questions.
"""
    )

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs (you can select multiple files)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    target_class = st.selectbox("Select class", [6, 7, 8, 9, 10], index=2)
    num_questions = st.slider("Number of questions", 3, 10, 5)

    if not uploaded_files:
        st.info("👆 Please upload at least one NCERT PDF to begin.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Reading PDFs and building vector index..."):
        docs = build_index_from_files(uploaded_files, CHUNK_SIZE, CHUNK_OVERLAP)
        index, _ = build_faiss_index(docs, embedder)

    st.success(f"Indexed {len(docs)} text chunks from {len(uploaded_files)} PDF(s).")

    topic = st.text_input(
        "Enter chapter name / topic (e.g., 'Democracy', 'Mughal Empire', 'Electricity')"
    )

    if topic and st.button("Generate subjective questions"):
        with st.spinner("Generating questions..."):
            questions_text, retrieved = generate_subjective_questions(
                topic=topic,
                target_class=target_class,
                num_questions=num_questions,
                index=index,
                docs=docs,
                embedder=embedder,
                tokenizer=tokenizer,
                gen_model=gen_model,
                device=device,
            )

        st.subheader("Generated Long-Answer Questions")
        st.write(questions_text)

        with st.expander("Show context chunks used"):
            for r in retrieved:
                st.markdown(f"**{r['doc_id']} – chunk {r['chunk_id']}**")
                st.write(r["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()

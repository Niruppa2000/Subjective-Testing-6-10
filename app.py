import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# ============================
# CONFIG
# ============================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"   # use -large if you have enough RAM
TOP_K = 5


# ============================
# PDF & TEXT UTILITIES
# ============================
def extract_text_from_pdf_filelike(file) -> str:
    """Read text from a PDF uploaded via Streamlit (BytesIO-like object)."""
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def build_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split large text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================
# LOAD MODELS (CACHED)
# ============================
@st.cache_resource(show_spinner="Loading models (embedding + Flan-T5)...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    return device, embedder, tokenizer, gen_model


# ============================
# INDEX BUILDING
# ============================
def build_index_from_files(uploaded_files):
    """Convert uploaded science PDFs into chunked docs."""
    docs = []  # list of {"doc_id", "chunk_id", "text"}
    for f in uploaded_files:
        raw_text = extract_text_from_pdf_filelike(f)
        chunks = build_chunks(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
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
    """Build a FAISS index from chunk embeddings."""
    emb_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(emb_dim)

    vectors = []
    for d in docs:
        vec = embedder.encode(d["text"], convert_to_numpy=True, show_progress_bar=False)
        vectors.append(vec)

    vectors = np.vstack(vectors).astype("float32")
    index.add(vectors)
    return index


# ============================
# RETRIEVAL
# ============================
def retrieve_context(query: str, index, embedder, docs, top_k: int = TOP_K):
    """Retrieve top_k relevant chunks from the FAISS index."""
    q_vec = embedder.encode(query, convert_to_numpy=True).astype("float32")
    q_vec = np.expand_dims(q_vec, axis=0)
    distances, indices = index.search(q_vec, top_k)
    indices = indices[0]
    retrieved = [docs[i] for i in indices]
    return retrieved


# ============================
# HELPER: COSINE SIMILARITY
# ============================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================
# QUESTION GENERATION (UNIQUE)
# ============================
def generate_questions(
    topic: str,
    target_class: int,
    num_questions: int,
    index,
    docs,
    embedder,
    tokenizer,
    gen_model,
    device,
):
    """
    Generate EXACTLY num_questions questions.
    Uses semantic similarity to avoid repeating the same question.
    """
    # 1) Retrieve context once
    chunks = retrieve_context(topic, index, embedder, docs, top_k=TOP_K)
    context_text = "\n\n".join([c["text"] for c in chunks])

    questions = []
    question_vecs = []

    for i in range(num_questions):
        # Try multiple times to get a different question
        attempt = 0
        best_candidate = None

        while attempt < 5:
            already = ""
            if questions:
                already = "\nExisting questions (do NOT repeat these, create something new):\n" + \
                          "\n".join([f"- {q}" for q in questions])

            prompt = f"""
You are an experienced NCERT Science teacher for Class {target_class}.
Create ONE NEW, DIFFERENT, long-answer exam question for the topic: "{topic}".

Use ONLY the following NCERT Science textbook context:

{context_text}

{already}

Requirements:
- Output ONLY ONE question sentence.
- Start with words like: Explain, Describe, What do you mean by, How does, Why, etc.
- The question must require a detailed answer (4–8 lines).
- The new question must be semantically different from all existing questions.
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.9,
                    no_repeat_ngram_size=3,
                    num_beams=1,
                    early_stopping=True,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            text = text.replace("\n", " ").strip()

            # Clean possible numbering / bullets
            if text[:2].isdigit() and "." in text[:4]:
                text = text.split(".", 1)[1].strip()
            if text.startswith("- "):
                text = text[2:].strip()

            best_candidate = text

            # If it's the first question, accept directly
            if not questions:
                break

            # Check similarity with previous questions
            cand_vec = embedder.encode(text, convert_to_numpy=True)
            sims = [cosine_sim(cand_vec, prev) for prev in question_vecs]
            max_sim = max(sims) if sims else 0.0

            # If it's not too similar, accept it
            if max_sim < 0.85:
                question_vecs.append(cand_vec)
                break

            attempt += 1

        # After attempts, accept best_candidate anyway
        questions.append(best_candidate)
        if len(question_vecs) < len(questions):  # first question or fallback
            q_vec = embedder.encode(best_candidate, convert_to_numpy=True)
            question_vecs.append(q_vec)

    # Nicely numbered block
    questions_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return questions_block, chunks


# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="NCERT Science Subjective Question Generator", layout="wide")
    st.title("🔬 NCERT Science Subjective Question Generator (Classes 6–10)")

    st.markdown(
        """
Upload **NCERT Science PDFs for Classes 6–10**  
and generate **exam-style, long-answer subjective questions**.
"""
    )

    uploaded_files = st.file_uploader(
        "Upload NCERT Science PDFs (you can select multiple files)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        target_class = st.selectbox("Select Class", [6, 7, 8, 9, 10], index=1)
    with col2:
        num_questions = st.slider("How many questions?", 1, 10, 5)

    topic = st.text_input(
        "Enter Topic (Example: Nutrition in Plants, Acids Bases Salts, Motion, Electricity)"
    )

    if not uploaded_files:
        st.info("👆 Please upload at least one NCERT **Science** PDF to begin.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Reading PDFs and building Science knowledge base..."):
        docs = build_index_from_files(uploaded_files)
        if not docs:
            st.error("No text could be extracted from the uploaded PDFs.")
            return
        index = build_faiss_index(docs, embedder)

    st.success(f"Indexed {len(docs)} text chunks from {len(uploaded_files)} Science PDF(s).")

    if topic and st.button("Generate Questions"):
        with st.spinner(f"Generating {num_questions} unique science questions..."):
            questions_text, retrieved = generate_questions(
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

        st.subheader("📄 Generated Science Questions")
        st.write(questions_text)

        with st.expander("Show textbook chunks used"):
            for r in retrieved:
                st.markdown(f"**{r['doc_id']} – chunk {r['chunk_id']}**")
                st.write(r["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()

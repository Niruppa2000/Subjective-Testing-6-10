import re
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

# Use base Flan-T5 for deployment
# (if you later push your fine-tuned model to HF / repo, change this name)
GEN_MODEL_NAME = "google/flan-t5-base"

TOP_K = 4  # how many chunks to retrieve as context


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
@st.cache_resource(show_spinner="Loading models (embeddings + Flan-T5)...")
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
    """
    Convert uploaded NCERT PDFs into chunked docs.
    Returns list[dict]: {"doc_id", "chunk_id", "text"}
    """
    docs = []
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
# QUESTION POST-PROCESSING
# ============================
def clean_and_extract_questions(raw_text: str, topic: str, num_questions: int):
    """
    Turn raw model output (e.g. '1. Q1? 2. Q2? 3. Q3?') into a clean list of questions.

    - Splits using numbering patterns: 1., 2., 3.
    - Ensures each question ends with '?'
    - Filters out junk
    - Fills remaining slots with templates based on the topic
    """
    # Normalize spaces/newlines into a single string
    raw = " ".join(raw_text.split()).strip()

    # Capture segments like "1. ... 2. ... 3. ..."
    segments = []
    for match in re.finditer(r"\d+\.\s*(.+?)(?=\d+\.|$)", raw):
        seg = match.group(1).strip()
        segments.append(seg)

    # If no numbered pattern found, treat whole text as one segment
    if not segments and raw:
        segments = [raw]

    questions = []

    for seg in segments:
        text = seg.strip()

        # Remove leading bullet if present
        if text.startswith("- "):
            text = text[2:].strip()

        # Ignore very short junk
        if len(text.split()) < 4:
            continue

        # Ensure it ends with '?'
        if not text.endswith("?"):
            text = text.rstrip(". ") + "?"

        questions.append(text)

    # ---- Fallback templates based on topic ----
    templates = [
        f"What do you mean by {topic}?",
        f"Explain {topic} in detail with suitable examples.",
        f"Why is {topic} important? Explain.",
        f"Describe {topic} in your own words.",
        f"List and explain the main features of {topic}.",
        f"How does {topic} affect our daily life? Explain.",
        f"Write a short note on {topic}.",
    ]

    # Deduplicate while filling up to num_questions
    seen = set(q.lower() for q in questions)
    for t in templates:
        if len(questions) >= num_questions:
            break
        if t.lower() not in seen:
            questions.append(t)
            seen.add(t.lower())

    return questions[:num_questions]


# ============================
# QUESTION GENERATION
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
    Retrieve context from the uploaded PDFs and generate exam-style questions.
    """
    retrieved = retrieve_context(topic, index, embedder, docs, top_k=TOP_K)
    context_text = "\n\n".join([c["text"] for c in retrieved])

    prompt = f"""
You are an experienced NCERT Class {target_class} teacher.

Using ONLY the textbook extract given in CONTEXT, write {num_questions} clear, exam-style questions
on the topic "{topic}".

Rules:
- Questions must be simple and meaningful for Class {target_class}.
- Start with words like: What, Why, How, Explain, Describe, Define, List, etc.
- Each question must be complete and end with a question mark (?).
- Write them in this exact format:
1. Question 1?
2. Question 2?
3. Question 3?
(only the numbered list, nothing else).

CONTEXT:
{context_text}
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    questions = clean_and_extract_questions(raw_text, topic, num_questions)

    questions_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return questions_block, retrieved


# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="NCERT Subjective Question Generator", layout="wide")
    st.title("📚 NCERT Subjective Question Generator (Classes 6–10)")

    st.markdown(
        """
Upload **NCERT PDFs (Science) for Classes 6–10**  
and generate **exam-style subjective questions**, """
    )

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs (you can select multiple files)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        target_class = st.selectbox("Select Class", [6, 7, 8, 9, 10], index=0)
    with col2:
        num_questions = st.slider("How many questions?", 1, 10, 5)

    topic = st.text_input(
        "Enter Topic (Example: Balanced diet, Motion, Acids Bases Salts, Ashoka, Harappan civilisation)"
    )

    if not uploaded_files:
        st.info("👆 Please upload at least one NCERT PDF to begin.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Reading PDFs and building knowledge base..."):
        docs = build_index_from_files(uploaded_files)
        if not docs:
            st.error("No text could be extracted from the uploaded PDFs.")
            return
        index = build_faiss_index(docs, embedder)

    st.success(f"Indexed {len(docs)} text chunks from {len(uploaded_files)} PDF(s).")

    if topic and st.button("Generate Questions"):
        with st.spinner(f"Generating {num_questions} questions..."):
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

        st.subheader("📄 Generated Questions")
        st.write(questions_text)

        with st.expander("Show textbook chunks used"):
            for r in retrieved:
                st.markdown(f"**{r['doc_id']} – chunk {r['chunk_id']}**")
                st.write(r["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()

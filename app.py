import os
import re
import io
import json
import gc
import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="NCERT — Similar Question Generator", layout="wide")
st.title("NCERT — Generate Teacher-Style Questions (CSV → Retrieval → Generation)")

CSV_PATH_IN_REPO = "ncert_teacher_style_questions_full.csv"  # ensure this exists in repo root
LORA_FOLDER = "./lora_ncert_finetuned"  # adjust if your adapter folder has different name
BASE_MODEL = "google/flan-t5-base"     # change if your LoRA used different base
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Helpers: load CSV
# -----------------------
@st.cache_data(show_spinner=False)
def load_questions_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # empty df template
        df = pd.DataFrame(columns=["Class", "Chapter", "Question"])
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_questions_csv(CSV_PATH_IN_REPO)
if df.empty:
    st.warning(f"No CSV found at {CSV_PATH_IN_REPO}. Upload one or place the downloaded CSV in the repo root.")
    uploaded = st.file_uploader("Upload CSV (columns: Class, Chapter, Question)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("CSV uploaded.")
    else:
        st.stop()

# Display sample
st.write("Dataset preview:")
st.dataframe(df.head(8))

# -----------------------
# UI controls
# -----------------------
col1, col2 = st.columns([2, 1])
with col2:
    class_options = sorted(df["Class"].unique().tolist())
    class_choice = st.selectbox("Choose Class", class_options, index=0 if class_options else 0)
    chapters_for_class = sorted(df[df["Class"] == class_choice]["Chapter"].unique().tolist())
    chapter_choice = st.selectbox("Choose Chapter", chapters_for_class, index=0 if chapters_for_class else 0)
    num_questions = st.slider("How many new questions to generate?", 1, 10, 5)
    retrieve_k = st.slider("Retrieve K example questions (style)", 1, 8, 4)
    num_samples = st.slider("Model samples to aggregate (diversity)", 1, 5, 2)
    generate_btn = st.button("Generate Similar Questions")

# -----------------------
# Load embedder & FAISS (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def build_embedder_and_index(df: pd.DataFrame, column: str = "Question"):
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")  # keep on CPU for stability
    texts = df[column].fillna("").astype(str).tolist()
    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return embedder, index, emb

embedder, index, embeddings = build_embedder_and_index(df, "Question")

# -----------------------
# Load generation model & tokenizer (cached)
# -----------------------
@st.cache_resource(show_spinner=True)
def load_generation_model(base_model: str = BASE_MODEL, adapter_dir: str = LORA_FOLDER):
    """Return tokenizer, model, bool(adapter_attached)."""
    # Try tokenizer from adapter first (if present); else from base
    tokenizer = None
    if os.path.isdir(adapter_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
            except Exception:
                tokenizer = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Load base model (float) and attach adapter if possible
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    base.to(DEVICE)
    adapter_attached = False
    if os.path.isdir(adapter_dir):
        try:
            model = PeftModel.from_pretrained(base, adapter_dir)
            model.to(DEVICE)
            adapter_attached = True
        except Exception as e:
            # fallback to base
            model = base
    else:
        model = base

    return tokenizer, model, adapter_attached

with st.spinner("Loading generation model (may take 20-60s)..."):
    tokenizer, model, adapter_attached = load_generation_model()

if adapter_attached:
    st.success("LoRA adapter attached to base model.")
else:
    st.info("Using base model (adapter not attached or not found).")

# -----------------------
# Utilities: retrieval & prompt building
# -----------------------
def retrieve_examples(query: str, k: int = 4) -> List[str]:
    qv = embedder.encode(query, convert_to_numpy=True).astype("float32")
    D, I = index.search(np.expand_dims(qv, 0), k)
    idxs = I[0].tolist()
    examples = [df.iloc[i]["Question"] for i in idxs]
    return examples

def build_prompt(class_no, chapter, examples: List[str], num_q: int):
    examples_block = "\n".join([f"{i+1}. {ex.strip()}" for i, ex in enumerate(examples)])
    prompt = (
        f"You are an experienced NCERT Class {class_no} teacher.\n"
        f"CHAPTER: {chapter}\n\n"
        f"Below are example teacher-style long-answer questions that show the required style and depth:\n\n"
        f"{examples_block}\n\n"
        f"Using the same style, write EXACTLY {num_q} numbered teacher-style long-answer questions for the chapter '{chapter}'. "
        "Number them 1., 2., 3., ... Each question must be a single sentence ending with a question mark. Prefer questions that refer to chapter activities or examples where suitable."
    )
    return prompt

# Parsing function
def parse_numbered_questions(text: str) -> List[str]:
    matches = re.findall(r"(?:^|\n)\s*(\d{1,2})[.)\-]?\s*(.+?)(?=(?:\n\s*\d{1,2}[.)\-])|\Z)", text, flags=re.S)
    qs = []
    if matches:
        for _, txt in matches:
            q = " ".join(txt.split())
            if not q.endswith("?"):
                q = q.rstrip(". ") + "?"
            qs.append(q)
    else:
        sents = re.findall(r"[A-Z][^?!.]{10,}\?", text)
        for s in sents:
            qs.append(s.strip())
    return qs

# -----------------------
# Generation routine
# -----------------------
def generate_from_prompt(prompt: str, num_return_sequences: int = 1, max_new_tokens: int = 300) -> List[str]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.92,
        repetition_penalty=1.6,
        no_repeat_ngram_size=3,
        num_return_sequences=num_return_sequences,
        early_stopping=True
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]

# -----------------------
# Generation button action
# -----------------------
if generate_btn:
    with st.spinner("Retrieving examples & generating..."):
        # Use chapter text and some example questions for retrieval
        seed_text = chapter_choice
        retrieved = retrieve_examples(seed_text, k=retrieve_k)
        st.markdown("**Retrieved examples (style used in prompt):**")
        for i, ex in enumerate(retrieved, 1):
            st.write(f"{i}. {ex}")

        # Build prompt and sample multiple times to aggregate diversity
        prompt = build_prompt(class_choice, chapter_choice, retrieved, num_questions)
        sampled_texts = []
        for _ in range(num_samples):
            sampled_texts += generate_from_prompt(prompt, num_return_sequences=1, max_new_tokens=350)

        # Parse & deduplicate
        candidates = []
        seen = set()
        for raw in sampled_texts:
            parsed = parse_numbered_questions(raw)
            for q in parsed:
                key = q.lower()
                if key not in seen and len(q.split()) > 4:
                    candidates.append(q)
                    seen.add(key)
        # Fill templates if insufficient
        templates = [
            f"Explain the main ideas of the chapter '{chapter_choice}' with suitable examples?",
            f"What are the key points discussed in '{chapter_choice}' and why are they important?",
            f"Describe any important activity from '{chapter_choice}' and explain its outcome?",
            f"How does the chapter '{chapter_choice}' relate to daily life? Give examples?"
        ]
        for t in templates:
            if len(candidates) >= num_questions:
                break
            if t not in seen:
                candidates.append(t)
                seen.add(t.lower())

        final = candidates[:num_questions]

        # Display results
        st.subheader("Generated Questions")
        for i, q in enumerate(final, 1):
            st.markdown(f"**{i}.** {q}")

        # Show raw outputs in expanders
        st.subheader("Raw model outputs")
        for i, raw in enumerate(sampled_texts, 1):
            with st.expander(f"Raw sample {i} (truncated)"):
                st.code(raw[:4000])

        # Download CSV
        out_df = pd.DataFrame({
            "Class": [class_choice]*len(final),
            "Chapter": [chapter_choice]*len(final),
            "Question_No": list(range(1, len(final)+1)),
            "Question": final
        })
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download generated questions CSV", data=csv_bytes, file_name="generated_ncert_questions.csv", mime="text/csv")

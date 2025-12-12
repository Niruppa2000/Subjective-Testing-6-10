# app.py
import streamlit as st
import io
import csv

st.set_page_config(page_title="NCERT Teacher Question Generator", layout="wide")
st.title("📘 NCERT Teacher-Style Question Generator (CSV-only, lightweight)")
st.write("Upload a CSV file with columns: Class, Chapter, Question")

uploaded = st.file_uploader("Upload CSV (CSV must contain headers: Class, Chapter, Question)", type=["csv"])
if not uploaded:
    st.info("Please upload your CSV file to proceed.")
    st.stop()

# read CSV using stdlib csv (works without pandas)
try:
    # uploaded is a BytesIO; decode to text
    text_io = io.TextIOWrapper(uploaded, encoding="utf-8")
    reader = csv.DictReader(text_io)
    rows = [r for r in reader]
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# minimal validation
required_cols = {"Class", "Chapter", "Question"}
if not required_cols.issubset(set(reader.fieldnames or [])):
    st.error(f"CSV must have these headers: {required_cols}. Found: {reader.fieldnames}")
    st.stop()

# Build index structures
classes = sorted({r["Class"].strip() for r in rows if r.get("Class") is not None})
if not classes:
    st.error("No 'Class' values found in CSV.")
    st.stop()

selected_class = st.selectbox("Select Class", classes)

# collect chapters for chosen class
chapters_set = sorted({r["Chapter"].strip() for r in rows if r.get("Class") and r["Class"].strip() == str(selected_class)})
if not chapters_set:
    st.warning("No chapters found for selected class.")
    chapter_choice = st.text_input("Chapter name (or upload CSV with chapters)", "")
else:
    chapter_choice = st.selectbox("Select Chapter", chapters_set)

# filter rows
filtered = [r for r in rows if r.get("Class") and r.get("Chapter") and r["Class"].strip() == str(selected_class) and r["Chapter"].strip() == str(chapter_choice)]

st.subheader(f"📖 Questions for Class {selected_class} — {chapter_choice}")
if not filtered:
    st.info("No questions found for this Class & Chapter. Make sure CSV rows exactly match Class and Chapter text.")
else:
    for idx, r in enumerate(filtered, start=1):
        q = (r.get("Question") or "").strip()
        if q:
            st.markdown(f"**{idx}. {q}**")

# allow download of filtered questions as CSV
import csv, io
if filtered:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Class", "Chapter", "Question"])
    writer.writeheader()
    for r in filtered:
        writer.writerow({"Class": r.get("Class",""), "Chapter": r.get("Chapter",""), "Question": r.get("Question","")})
    st.download_button("Download filtered questions as CSV", data=output.getvalue().encode("utf-8"), file_name="filtered_ncert_questions.csv")

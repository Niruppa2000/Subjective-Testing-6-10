import streamlit as st
import pandas as pd
import random

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NCERT Question Generator",
    layout="wide"
)

st.title("📘 NCERT Chapter-wise Question Generator")
st.write("Generate questions **directly from CSV** based on chapter name")

# ----------------------------
# Load CSV
# ----------------------------
@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)

uploaded_file = st.file_uploader("📂 Upload Questions CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    required_columns = {
        "class", "chapter", "question",
        "option_a", "option_b", "option_c", "option_d", "answer"
    }

    if not required_columns.issubset(set(df.columns)):
        st.error("❌ CSV does not contain required columns")
        st.stop()

    # ----------------------------
    # User Inputs
    # ----------------------------
    chapter_name = st.text_input("📖 Enter Chapter Name", placeholder="e.g. Nutrition in Plants")
    num_questions = st.number_input(
        "🔢 Number of Questions",
        min_value=1,
        max_value=50,
        value=5
    )

    if st.button("🚀 Generate Questions"):
        filtered_df = df[df["chapter"].str.lower() == chapter_name.lower()]

        if filtered_df.empty:
            st.warning("⚠ No questions found for this chapter")
        else:
            questions = filtered_df.sample(
                min(num_questions, len(filtered_df)),
                random_state=random.randint(1, 10000)
            )

            st.success(f"✅ Showing {len(questions)} questions from **{chapter_name}**")

            # ----------------------------
            # Display Questions
            # ----------------------------
            for idx, row in enumerate(questions.itertuples(), start=1):
                st.markdown(f"### Q{idx}. {row.question}")

                st.write(f"A. {row.option_a}")
                st.write(f"B. {row.option_b}")
                st.write(f"C. {row.option_c}")
                st.write(f"D. {row.option_d}")

                with st.expander("✅ Show Answer"):
                    st.write(f"**Correct Answer:** {row.answer}")

else:
    st.info("⬆ Upload your questions CSV to get started")

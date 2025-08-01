import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Title
st.title("📚 StudyMate - Ask Questions from PDFs")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# LLM model
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"

headers = {
    "Authorization": "Bearer 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"  # ✅ your token
}

def query_llm(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# PDF text extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Text splitter
def split_text(text, max_len=500):
    words = text.split()
    return [' '.join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

# Vectorize chunks
def vectorize_chunks(chunks):
    return embed_model.encode(chunks)

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)
    embeddings = vectorize_chunks(chunks)

    # Build FAISS index
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Ask question
    question = st.text_input("Ask a question based on the PDF:")

    if question:
        question_embedding = embed_model.encode([question])
        D, I = index.search(np.array(question_embedding), k=3)
        matched_chunks = [chunks[i] for i in I[0]]

        with st.expander("🔍 Top matching content"):
            for chunk in matched_chunks:
                st.write("- " + chunk[:300] + "...")

        # Prepare prompt
        context = "\n\n".join(matched_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        st.subheader("🧠 Answer")
        answer = query_llm(prompt)
        st.success(answer)

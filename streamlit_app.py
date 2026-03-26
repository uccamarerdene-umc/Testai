# -*- coding: utf-8 -*-
import streamlit as st
import os
import glob
import time
import unicodedata
import random

from docx import Document as DocxReader
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Central Test AI", page_icon="🤖", layout="wide")

def get_key(name):
    val = st.secrets.get(name)
    return str(val).strip() if val else None

google_api_key = get_key("GOOGLE_API_KEY")
pinecone_api_key = get_key("PINECONE_API_KEY")
index_name = "testai"

if not google_api_key or not pinecone_api_key:
    st.error("API key missing!")
    st.stop()

genai.configure(api_key=google_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# ---------------- CSS ----------------
st.markdown("""
<style>

body {
    background-color: #F5F7FB;
    font-family: 'Inter', sans-serif;
}

/* LEFT PANEL */
.left-panel {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    height: 100%;
}

/* CHAT CARD */
.chat-card {
    background-color: white;
    padding: 24px;
    border-radius: 16px;
    border-left: 4px solid #6D28D9;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

/* BUTTON */
.stButton>button {
    background-color: #F97316;
    color: white;
    border-radius: 12px;
}

/* INPUT */
input {
    border-radius: 12px !important;
}

/* HISTORY ITEM */
.history-item {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    cursor: pointer;
}

.history-item:hover {
    background-color: #F1F5F9;
}

</style>
""", unsafe_allow_html=True)

# ---------------- EMBEDDING ----------------
def embed_text(text):
    text = text[:8000]
    last_error = None

    for _ in range(5):
        try:
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text
            )
            if "embedding" in res:
                return res["embedding"]

        except Exception as e:
            last_error = e
            if "429" in str(e) or "ResourceExhausted" in str(e):
                time.sleep(10 + random.random()*5)
            else:
                time.sleep(2)

    raise Exception(f"Embedding failed: {last_error}")

# ---------------- CLEAN ----------------
def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ---------------- CHUNK ----------------
def split_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------- LOAD DOCX ----------------
def load_docx():
    docs = []
    for file in glob.glob("data/*.docx"):
        doc = DocxReader(file)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        text = clean_text(text)

        for chunk in split_text(text):
            docs.append({"text": chunk, "source": file})

    return docs

# ---------------- SYNC ----------------
with st.sidebar:
    st.header("⚙️ Sync")
    pwd = st.text_input("Password", type="password")

    if pwd == "admin123":
        if st.button("🔄 Sync Data"):

            docs = load_docx()

            if not docs:
                st.error("No documents found!")
            else:
                with st.spinner("Uploading..."):

                    existing = [i["name"] for i in pc.list_indexes()]
                    if index_name in existing:
                        pc.delete_index(index_name)
                        time.sleep(5)

                    pc.create_index(
                        name=index_name,
                        dimension=3072,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )

                    while not pc.describe_index(index_name).status["ready"]:
                        time.sleep(2)

                    index = pc.Index(index_name)

                    batch_size = 2
                    vectors = []

                    for i, doc in enumerate(docs):

                        emb = embed_text(doc["text"])

                        vectors.append({
                            "id": f"id-{i}",
                            "values": emb,
                            "metadata": {
                                "text": doc["text"][:500],
                                "source": doc["source"]
                            }
                        })

                        if len(vectors) == batch_size:
                            index.upsert(vectors=vectors)
                            vectors = []
                            time.sleep(5)

                    if vectors:
                        index.upsert(vectors=vectors)

                    st.success(f"✅ Sync done: {len(docs)} chunks")

# ---------------- MAIN LAYOUT ----------------
col1, col2, col3 = st.columns([1, 3, 1])

# -------- LEFT PANEL --------
with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    st.markdown("## Central Test AI туслах")

    st.button("➕ Central Test Хиймэл оюун ухаант туслах гэж юу вэ?")

    st.markdown("### Research History")

    st.markdown('<div class="history-item">Шинжлэх ухаанаар баталгаажсан сэтгэл зүйн тест<br><small>Today</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="history-item">Олон улсын стандартаар баталгаажсан сэтгэл зүйн тест<br><small>Today</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="history-item">Технологид суурилсан ухаалаг шийдэл<br><small>Yesterday</small></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- CENTER --------
with col2:

    query = st.text_input("Ask Central Test AI...")

    if query:
        with st.spinner("AI бичиж байна..."):
            try:
                index = pc.Index(index_name)

                query_vector = embed_text(query)

                results = index.query(
                    vector=query_vector,
                    top_k=5,
                    include_metadata=True
                )

                matches = results.get("matches", [])

                if not matches:
                    st.warning("Мэдээлэл олдсонгүй")
                else:
                    context = "\n\n".join([m["metadata"]["text"] for m in matches])

                    model = genai.GenerativeModel("gemini-2.5-flash")

                    prompt = f"""
Чи бол Central Test компанийн AI туслах.

Мэдээлэл:
{context}

Асуулт:
{query}
"""

                    st.markdown('<div class="chat-card">', unsafe_allow_html=True)

                    placeholder = st.empty()
                    full_text = ""

                    response = model.generate_content(prompt)

                    for char in response.text:
                        full_text += char
                        placeholder.markdown(full_text)
                        time.sleep(0.003)

                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                import traceback
                st.error(str(e))
                st.code(traceback.format_exc())

# -------- RIGHT PANEL --------
with col3:
    st.markdown("### Central Test хиймэл оюун ухаант туслахыг хэрхэн ашиглах вэ?")
    st.info("Тестийн үр дүнг хэрэглээнд нэвтрүүлэх боломж")

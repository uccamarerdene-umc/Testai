# -*- coding: utf-8 -*-
import streamlit as st
import os
import glob
import time
import unicodedata

from docx import Document as DocxReader
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Central Test AI Assistant",
    page_icon="🤖",
    layout="wide"
)

def get_key(name):
    val = st.secrets.get(name)
    if val:
        return str(val).replace('—', '-').strip()
    return None

google_api_key = get_key("GOOGLE_API_KEY")
pinecone_api_key = get_key("PINECONE_API_KEY")
index_name = "testai"

if not google_api_key or not pinecone_api_key:
    st.error("API key missing!")
    st.stop()

genai.configure(api_key=google_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# ---------------- EMBEDDING ----------------
def embed_text(text):
    return genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )["embedding"]

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ---------------- LOAD DOCX ----------------
def load_docx():
    docs = []
    for file in glob.glob("data/*.docx"):
        doc = DocxReader(file)

        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        docs.append({"text": clean_text(text), "source": file})

    return docs

# ---------------- SIDEBAR SYNC ----------------
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

                    # recreate index
                    if index_name in [i["name"] for i in pc.list_indexes()]:
                        pc.delete_index(index_name)
                        time.sleep(5)

                    pc.create_index(
                        name=index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )

                    index = pc.Index(index_name)

                    batch_size = 20
                    vectors = []

                    for i, doc in enumerate(docs):
                        emb = embed_text(doc["text"])

                        vectors.append({
                            "id": f"id-{i}",
                            "values": emb,
                            "metadata": {
                                "text": doc["text"],
                                "source": doc["source"]
                            }
                        })

                        if len(vectors) == batch_size:
                            index.upsert(vectors=vectors)
                            vectors = []
                            time.sleep(2)

                    if vectors:
                        index.upsert(vectors=vectors)

                    st.success("Sync completed!")

# ---------------- UI ----------------
st.title("🤖 Central Test AI")

query = st.text_input("Асуултаа бичнэ үү")

if query:
    with st.spinner("Хариулт бэлдэж байна..."):
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

                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = f"""
Чи бол Central Test компанийн AI туслах.

Доорх мэдээлэлд тулгуурлан зөвхөн монгол хэлээр хариул.

Мэдээлэл:
{context}

Асуулт:
{query}

Хэрэв мэдээлэл байхгүй бол:
"Мэдээлэл алга" гэж хариул.
"""

                response = model.generate_content(prompt)

                st.markdown("### 🤖 Хариулт")
                st.write(response.text)

                with st.expander("Эх сурвалж"):
                    for m in matches:
                        st.write(m["metadata"]["source"])

        except Exception as e:
            st.error(str(e))

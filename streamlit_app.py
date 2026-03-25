import streamlit as st
import os
import glob
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Тохиргоо болон Нууцлал
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖", layout="wide")

# --- CUSTOM CSS (Figma Design Concept) ---
st.markdown("""
    <style>
    /* Үндсэн арын дэвсгэр */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Гарчиг болон Текст */
    h1 {
        color: #1E293B;
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
    }
    
    /* Sidebar-ийн загвар */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Товчлуурын загвар */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background-color: #4F46E5;
        color: white;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    
    /* Input box-ийг Figma шиг болгох */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 12px;
    }
    
    /* Хариултын хэсгийн загвар */
    .answer-box {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant")
st.markdown("<p style='color: #64748B;'>Байгууллагын дотоод мэдээллийн сангаас хайлт хийх ухаалаг туслах</p>", unsafe_allow_html=True)
st.markdown("---")

index_name = "testai" 

@st.cache_resource
def load_models():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

if not google_api_key or not pinecone_api_key:
    st.error("API keys missing! Streamlit Secrets-ээ шалгана уу.")
    st.stop()

embeddings, pc = load_models()

# 3. Sidebar - Өгөгдөл удирдах (Sync)
with st.sidebar:
    st.header("⚙️ Тохиргоо")
    if st.button("🔄 Өгөгдөл Синхрончлох"):
        if not os.path.exists("data"):
            os.makedirs("data")
            st.warning("'data' хавтас олдсонгүй.")
        else:
            docx_files = glob.glob("data/*.docx")
            if not docx_files:
                st.error(".docx файл олдсонгүй!")
            else:
                with st.spinner("Мэдээллийг шинэчилж байна..."):
                    try:
                        loader = DirectoryLoader("data", glob="./*.docx", loader_cls=Docx2txtLoader)
                        docs = loader.load()
                        
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800, 
                            chunk_overlap=150,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        texts = splitter.split_documents(docs)
                        
                        # --- BATCH PROCESSING (Safe Mode) ---
                        batch_size = 5  
                        vectorstore = None
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i + batch_size]
                            status_text.text(f"Боловсруулж байна: {i}/{len(texts)}")
                            
                            if i == 0:
                                vectorstore = PineconeVectorStore.from_documents(
                                    batch, embeddings, index_name=index_name, pinecone_api_key=pinecone_api_key
                                )
                            else:
                                vs = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
                                vs.add_documents(batch)
                            
                            progress = (i + len(batch)) / len(texts)
                            progress_bar.progress(min(progress, 1.0))
                            time.sleep(15) # Google Quota protection
                        
                        status_text.text("Синхрончлол дууслаа!")
                        st.success("Амжилттай хадгаллаа.")
                    except Exception as e:
                        st.error(f"Алдаа: {e}")

# 4. Chat Interface
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Жишээ нь: Компанийн амралтын журам ямар байдаг вэ?")

if query:
    with st.spinner("Мэдээлэл дуудаж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
            
            search_results = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=google_api_key,
                temperature=0.1
            )
            
            prompt = f"""
            Та бол Central Test компанийн AI туслах байна. 
            Доорх мэдээлэл дээр тулгуурлан асуултанд маш дэлгэрэнгүй хариул.
            
            Мэдээлэл:
            {context}
            
            Асуулт: {query}
            
            ЗААВАР:
            1. Зөвхөн өгөгдсөн мэдээллийг ашигла.
            2. Хариултыг монгол хэлээр, маш эелдэг, цэгцтэй бичээрэй.
            """
            
            response = llm.invoke(prompt)
            
            st.markdown("### 🤖 Хариулт:")
            st.markdown(f"""
                <div class="answer-box">
                    {response.content}
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Ашигласан эх сурвалж"):
                st.info(context)
                
        except Exception as e:
            st.error(f"Алдаа: {e}")

import streamlit as st
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Тохиргоо болон Нууцлал
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant (BGE-M3)")
st.markdown("---")

# Pinecone индекс (1024 dimension-той байх ёстой)
index_name = "centralai" 

# 2. Моделиудыг ачаалах (Cache ашиглана)
@st.cache_resource
def load_models():
    # Монгол хэлэнд зориулсан 1024 dimensions-той BGE-M3 модель
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

# Түлхүүрүүд байхгүй бол анхааруулга өгөх
if not google_api_key or not pinecone_api_key:
    st.error("API keys missing! Streamlit Secrets-ээ шалгана уу.")
    st.stop()

embeddings, pc = load_models()

# 3. Sidebar - Өгөгдөл удирдах (Sync)
with st.sidebar:
    st.header("⚙️ Тохиргоо")
    if st.button("🔄 Өгөгдөл Синхрончлох"):
        # Жижиг 'data' хавтас байгаа эсэхийг шалгах, байхгүй бол үүсгэх
        if not os.path.exists("data"):
            os.makedirs("data")
            st.warning("'data' хавтас олдсонгүй, шинээр үүсгэлээ. Дотор нь .docx файлуудаа хийнэ үү.")
        else:
            docx_files = glob.glob("data/*.docx")
            if not docx_files:
                st.error("'data' хавтас дотор .docx файл олдсонгүй!")
            else:
                with st.spinner("Баримтуудыг боловсруулж байна (BGE-M3)..."):
                    try:
                        # DirectoryLoader-г жижиг 'data' хавтастай холбох
                        loader = DirectoryLoader("data", glob="./*.docx", loader_cls=Docx2txtLoader)
                        docs = loader.load()
                        
                        # Текстийг монгол хэлний онцлогт тохируулж хуваах
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800, 
                            chunk_overlap=150,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        texts = splitter.split_documents(docs)
                        
                        # Pinecone руу илгээх
                        PineconeVectorStore.from_documents(
                            texts, 
                            embeddings, 
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )
                        st.success(f"Амжилттай! {len(texts)} хэсэг текстийг хадгаллаа.")
                    except Exception as e:
                        st.error(f"Алдаа гарлаа: {e}")

# 4. Chat Interface

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

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

# Pinecone индекс (MiniLM ашиглах тул 384 dimension-той байх ёстой)
index_name = "testai" 

# 2. Моделиудыг ачаалах (Оновчтой хувилбар)
@st.cache_resource
def load_models():
    # Streamlit Cloud-д зориулсан хөнгөн, хурдан модель (384 dimensions)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
        if not os.path.exists("data"):
            os.makedirs("data")
            st.warning("'data' хавтас олдсонгүй, шинээр үүсгэлээ. Дотор нь .docx файлуудаа хийнэ үү.")
        else:
            docx_files = glob.glob("data/*.docx")
            if not docx_files:
                st.error("'data' хавтас дотор .docx файл олдсонгүй!")
            else:
                with st.spinner("Баримтуудыг боловсруулж байна..."):
                    try:
                        loader = DirectoryLoader("data", glob="./*.docx", loader_cls=Docx2txtLoader)
                        docs = loader.load()
                        
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=700, 
                            chunk_overlap=100,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        texts = splitter.split_documents(docs)
                        
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
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Мэдээллийн сангаас хайх...")

if query:
    with st.spinner("Мэдээллийг шинжилж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
            
            search_results = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
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
            2. Хариултыг монгол хэлээр, эелдэг бич.
            """
            
            response = llm.invoke(prompt)
            st.markdown("### 🤖 Хариулт:")
            st.write(response.content)
            
            with st.expander("Эх сурвалж"):
                st.info(context)
                
        except Exception as e:
            st.error(f"Алдаа: {e}")

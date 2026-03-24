import streamlit as st
import os
import glob
import time  # Хугацаа хэмжих, амраахад хэрэгтэй
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Тохиргоо болон Нууцлал
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

# Pinecone индекс (Gemini Embedding 001 нь 768 dimension-той)
index_name = "testai" 

# 2. Моделиудыг ачаалах (API-аар шууд холбогдоно)
@st.cache_resource
def load_models():
    # Google Gemini Embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
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
                with st.spinner("Google Gemini-ээр өгөгдлийг хэсэгчлэн илгээж байна..."):
                    try:
                        loader = DirectoryLoader("data", glob="./*.docx", loader_cls=Docx2txtLoader)
                        docs = loader.load()
                        
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        texts = splitter.split_documents(docs)
                        
                        # --- BATCH PROCESSING (Quota алдаанаас зайлсхийх) ---
                        batch_size = 10  # Нэг удаад 10 текст илгээнэ
                        vectorstore = None
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i:i + batch_size]
                            status_text.text(f"Боловсруулж байна: {i}/{len(texts)} хэсэг...")
                            
                            if i == 0:
                                # Эхний хэсэгт шинээр вектор сан үүсгэнэ
                                vectorstore = PineconeVectorStore.from_documents(
                                    batch, 
                                    embeddings, 
                                    index_name=index_name,
                                    pinecone_api_key=pinecone_api_key
                                )
                            else:
                                # Дараагийн хэсгүүдийг нэмж хадгална
                                vectorstore.add_documents(batch)
                            
                            # Явцыг шинэчлэх
                            progress = (i + len(batch)) / len(texts)
                            progress_bar.progress(progress)
                            
                            # Google-ийн Rate Limit-д тусахгүйн тулд 10 секунд амраана
                            if i + batch_size < len(texts):
                                time.sleep(10)
                        
                        status_text.text("Синхрончлол дууслаа!")
                        st.success(f"Амжилттай! Нийт {len(texts)} хэсэг текстийг хадгаллаа.")
                        
                    except Exception as e:
                        st.error(f"Алдаа гарлаа: {e}")

# 4. Chat Interface
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Мэдээллийн сангаас хайх...")

if query:
    with st.spinner("Хайж байна..."):
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
            2. Мэдээлэлд байхгүй зүйлийг өөрөөсөө бүү зохио.
            3. Хариултыг монгол хэлээр, эелдэг бичээрэй.
            """
            
            response = llm.invoke(prompt)
            st.markdown("### 🤖 Хариулт:")
            st.write(response.content)
            
            with st.expander("Эх сурвалж харах"):
                st.info(context)
                
        except Exception as e:
            st.error(f"Алдаа: {e}")

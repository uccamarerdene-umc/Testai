import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Configuration and Secrets
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant (BGE-M3)")
st.markdown("---")

# АНХААР: Pinecone дээр 1024 dimension-той шинэ индекс үүсгээрэй!
index_name = "central-test-bge" 

# 2. Model Loading (BGE-M3 ашиглан шинэчилсэн)
@st.cache_resource
def load_models():
    # Монгол хэлэнд зориулсан 1024 dimensions-той BGE-M3 модель
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'} # Хэрэв GPU байгаа бол 'cuda'
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# 3. Sidebar - Data Management
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        if not os.path.exists("Data"):
            st.error("'Data' folder not found!")
        else:
            with st.spinner("Processing documents (BGE-M3)..."):
                try:
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    
                    # Монгол хэлний утга санааг хадгалахын тулд chunk-ийг арай томсгов
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800, 
                        chunk_overlap=150,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    texts = splitter.split_documents(docs)
                    
                    # Pinecone-д хадгалах
                    PineconeVectorStore.from_documents(
                        texts, 
                        embeddings, 
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Successfully synced {len(texts)} blocks with BGE-M3!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# 4. Chat Interface
query = st.text_input("Ask a question:", placeholder="Central Test-ийн мэдээллийн сангаас хайх...")

if query:
    if not google_api_key or not pinecone_api_key:
        st.warning("API keys are missing.")
    else:
        with st.spinner("Analyzing with BGE-M3 and generating response..."):
            try:
                vectorstore = PineconeVectorStore(
                    index_name=index_name, 
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Ижил төстэй 5-7 хэсгийг шүүж гаргах
                search_results = vectorstore.similarity_search(query, k=6)
                context = "\n\n".join([doc.page_content for doc in search_results])
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", # Хамгийн тогтвортой хувилбар
                    google_api_key=google_api_key,
                    temperature=0.1
                )
                
                prompt = f"""
                Та бол Central Test компанийн албан ёсны AI туслах байна. 
                Доорх 'Мэдээлэл' хэсэгт байгаа текст дээр тулгуурлан хэрэглэгчийн асуултанд маш дэлгэрэнгүй хариулна уу.
                
                Мэдээлэл:
                {context}
                
                Асуулт: {query}
                
                ХАРИУЛАХ ЗААВАР:
                1. Зөвхөн өгөгдсөн 'Мэдээлэл' доторх текстийг ашигла.
                2. Мэдээлэл дутуу байвал өөрийн мэдлэгээр бүү нөхөж хариул.
                3. Хариултыг монгол хэлээр, маш ойлгомжтой, эелдэг бич.
                """
                
                response = llm.invoke(prompt)
                st.markdown("### 🤖 AI Response:")
                st.write(response.content)
                
                with st.expander("Эх сурвалж (Шүүж авсан өгөгдөл)"):
                    st.info(context)
                    
            except Exception as e:
                st.error(f"System error: {e}")

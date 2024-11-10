from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from dotenv import load_dotenv
import os
import time
import streamlit as st
import nest_asyncio

load_dotenv()

nest_asyncio.apply()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if not nvidia_api_key:
    st.error("NVIDIA API key is missing from environment variables.")
else:
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.llm = NVIDIA(model="meta/llama-3.1-405b-instruct")

# Styling for the title and other text elements
st.markdown("""
    <style>
    .title {
        font-size: 35px;
        font-weight: bold;
        color: #1e90ff;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    .button:hover {
        background-color: #45a049;
    }
    .footer {
        font-size: 12px;
        color: #777;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Health Symptoms & Medical Advice Demo</div>', unsafe_allow_html=True)

prompt_template = """
Based on the provided medical information, please answer the following health-related question.
Provide an accurate diagnosis or medical advice based only on the context of the symptoms mentioned.
<context>
{context}
</context>
User's Symptoms: {input}
Disease Diagnosis and Medical Advice:
"""

def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.loader = SimpleDirectoryReader("./medical_data")
            documents = st.session_state.loader.load_data()

            if len(documents) == 0:
                st.error("No documents found")
                return

            documents_to_process = documents[:30]
            if len(documents_to_process) == 0:
                st.error("No document")
                return

            st.write("Embedding started...")

            st.session_state.index = VectorStoreIndex.from_documents(
                documents_to_process, embeddings=NVIDIAEmbedding(model="NV-Embed-QA")
            )
            st.write("Medical Vector Store DB is Ready!")
        except Exception as e:
            st.error(f"Error during embedding process: {e}")

prompt1 = st.text_input("Enter Your Health Symptoms (e.g., fever, cough, fatigue, etc.)", placeholder="Enter symptoms here...", max_chars=200)


if st.button("Start Document Embedding", key="embedding_button", help="Click to process the documents and generate embeddings."):
    vector_embedding()

st.markdown("---")  

if prompt1:
    if "index" in st.session_state:
        try:
            query_engine = st.session_state.index.as_query_engine(similarity_top_k=20)

            start = time.process_time()
            response = query_engine.query(f"User's Symptoms: {prompt1}")
            st.write("Response time:", time.process_time() - start)

            st.write("Medical Advice & Disease Diagnosis: ", response.response)

            with st.expander("Related Medical Information", expanded=False):
                for doc in response.source_nodes:
                    st.write(f"File: {doc.metadata['file_name']} (Page {doc.metadata['page_label']})")
                    st.write(f"Path: {doc.metadata['file_path']}")
                    st.write(doc.node.text)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error during retrieval process: {e}")
    else:
        st.write("Please run the 'Start Document Embedding' process first.")


st.markdown('<div class="footer">Powered by LlamaIndex & NVIDIA AI</div>', unsafe_allow_html=True)

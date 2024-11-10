# Health Symptoms & Medical Advice Demo

This project is a **Retrieval-Augmented Generation (RAG)** system designed to provide medical advice based on user symptoms. It uses **NVIDIA Embeddings** and **LlamaIndex** to retrieve and synthesize relevant medical information from a document database.

## Features
- **NVIDIA Embeddings** for accurate contextual understanding.
- **LlamaIndex** for efficient query processing on large medical datasets.
- **Streamlit Interface** for user-friendly interaction.
- Deployed using **Streamlit Cloud (Snowflake Deployment)**.

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/saiMahadasa/llamaindex_nvidia_medical_advice_rag.git
cd healthcare_rag

###Create a .env file in the project directory and add your NVIDIA API key:
NVIDIA_API_KEY=your_nvidia_api_key

###Make sure you have Python 3.8+ installed, then run:
pip install -r requirements.txt

###Prepare Medical Data
Place your medical documents in a folder named medical_data in the project root directory.

###Start the Streamlit app locally:
streamlit run app.py

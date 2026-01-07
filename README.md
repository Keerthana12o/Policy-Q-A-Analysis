# 🛡️ PolicyPal: AI-Powered Insurance Policy Analyzer

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3-orange?style=for-the-badge)

## 📌 Project Overview
**PolicyPal** is an RAG (Retrieval-Augmented Generation) application designed to solve the complexity of reading insurance documents. 

Instead of searching through 50+ pages of legal jargon, users can simply upload a PDF policy and ask questions like *"Is flood damage covered?"* or *"What is my deductible?"*. The AI retrieves the exact section and explains it in plain English.

## 🚀 Key Features
- **Hybrid RAG Architecture:** 
  - **Memory:** Uses **Local CPU Embeddings** (HuggingFace) for unlimited, free document indexing (No Rate Limits).
  - **Brain:** Uses **Groq (Llama 3.3)** for lightning-fast, high-intelligence reasoning.
- **Hallucination Prevention:** The AI is strictly constrained to answer *only* based on the provided document.
- **Source Transparency:** (Optional) Can cite specific sections used for the answer.
- **Privacy First:** Document processing happens locally; only relevant text chunks are sent to the LLM.

## 🛠️ Tech Stack
| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Rapid UI development for data apps. |
| **LLM** | Llama 3.3 (via Groq) | Faster than GPT-4, completely free API. |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Runs locally on CPU. No API costs or latency. |
| **Vector DB** | FAISS (CPU) | Efficient similarity search for text chunks. |
| **Orchestrator** | LangChain | Manages the RAG pipeline and context retrieval. |

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- A free API Key from [Groq Console](https://console.groq.com/)


python -m venv venv
.\venv\Scripts\activate


pip install -r requirements.txt

GROQ_API_KEY=gsk_your_actual_api_key_here

streamlit run app.py

Upload: Drag and drop your Insurance Policy PDF into the sidebar.

Process: Click the "Analyze Policy" button.

Note: The first run downloads a small model (50MB) locally, so it may take 30 seconds.

Ask: Type questions like:

"Does this policy cover theft?"

"What are the exclusions for water damage?"

"How much is the premium increase after an accident?"

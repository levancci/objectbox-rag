# ObjectBox RAG with Local Llama3

A Streamlit application that implements RAG (Retrieval Augmented Generation) using:
- Local Llama3 model via Ollama
- ObjectBox for vector storage
- Local PDF document processing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and pull required models:
```bash
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

3. Place your PDF documents in the `us_census` directory

4. Run the application:
```bash
streamlit run app.py
```

## Features

- Fully offline RAG implementation
- PDF document processing
- Vector similarity search
- Local LLM integration

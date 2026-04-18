# Codebasics FAQ Assistant

A retrieval-augmented question-answering system that grounds answers in a domain-specific FAQ corpus using vector retrieval and prompt-constrained generation.

This project is designed to answer Codebasics questions with source-grounded responses instead of relying on a general-purpose chatbot alone. It combines semantic retrieval over a curated FAQ dataset with an LLM that is instructed to stay within the retrieved context and explicitly acknowledge when the knowledge base is insufficient.

## Why This Matters

Most chatbot demos look convincing until they face domain-specific questions. This project focuses on a narrower but more realistic problem: improving answer reliability for a known knowledge base.

The system is built to:

- retrieve semantically relevant FAQ entries from a local vector index
- constrain generation to retrieved context
- surface retrieved source chunks in the UI for transparency
- keep the retrieval layer local and reproducible with FAISS

## System Architecture

1. `codebasics_faqs.csv` is loaded as the source corpus.
2. `hkunlp/instructor-large` converts each FAQ entry into embeddings.
3. FAISS stores those embeddings in a local vector index.
4. A retriever selects the most relevant chunks for each user question.
5. `gpt-4o` generates an answer using a prompt that discourages unsupported claims.
6. Streamlit displays the answer and the retrieved context used to support it.

## Tradeoffs

- `Instructor-large` improves retrieval quality, but it is heavier than smaller embedding models.
- FAISS keeps the app fast and local, but it is better suited to small or medium corpora than production-scale search.
- The app optimizes for grounded answers over conversational flexibility, so it may decline to answer when context is weak.
- The current corpus is a CSV FAQ file, which is simple to manage but limits coverage and document structure.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ketaki99/RAG-Question-Answering.git
cd RAG-Question-Answering
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-openai langchain-community faiss-cpu InstructorEmbedding sentence-transformers python-dotenv
```

### 4. Add your OpenAI API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 5. Start the app

```bash
streamlit run main.py
```

The FAISS index is generated locally if it does not already exist. You can also rebuild it from the Streamlit UI using the `Rebuild knowledge base` button.

## Limitations

- Answers are only as strong as the FAQ corpus.
- The retriever uses a simple similarity threshold and does not rerank results.
- There is no evaluation suite yet for retrieval precision, groundedness, or answer quality.
- The UI is optimized for demo clarity, not multi-user deployment or observability.

## What I Would Improve Next

- add an evaluation harness for retrieval accuracy and answer groundedness
- separate ingestion, indexing, and serving into clearer workflows
- add metadata-aware retrieval and reranking
- persist citations in a cleaner, user-facing format
- containerize the app and add dependency pinning for reproducible setup

## Demo

### Home Screen
![Home Screen](screenshots/ss-1.png)

### Question Answering Example (Python)
![Python Question](screenshots/ss-2.png)

### Question Answering Example (TypeScript)
![TypeScript Question](screenshots/ss-3.png)

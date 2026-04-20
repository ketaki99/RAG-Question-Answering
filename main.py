import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.set_page_config(page_title="RAG FAQ Assistant", page_icon="📚")

st.title("RAG FAQ Assistant")
st.caption(
    "Retrieval-augmented QA system using FAISS, LangChain, Instructor embeddings, and OpenAI."
)

btn = st.button("Rebuild knowledge base")
if btn:
    create_vector_db()
    st.success("Knowledge base rebuilt from codebasics_faqs.csv")

question = st.text_input("Ask a question about the Codebasics FAQ knowledge base")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])

    sources = response.get("source_documents", [])
    if sources:
        st.subheader("Retrieved context")
        for index, doc in enumerate(sources[:3], start=1):
            st.markdown(f"**Chunk {index}**")
            st.write(doc.page_content)

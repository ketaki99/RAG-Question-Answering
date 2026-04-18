from pathlib import Path

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()  # take environment variables from .env (especially openai api key)

llm = ChatOpenAI(
    model="gpt-4o",  # model name from OpenAI
    temperature=0.2,  # lower = more deterministic
)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)
vectordb_file_path = "faiss_index"
prompt_template = """
You are answering questions about Codebasics using only the provided context.

Rules:
- If the answer is supported by the context, answer clearly and concisely.
- If the context is insufficient, say that the answer is not available in the knowledge base.
- Do not invent policies, prices, timelines, or guarantees.

Context:
{context}

Question:
{question}

Answer:
"""


def create_vector_db():
    loader = CSVLoader(
        file_path="codebasics_faqs.csv", source_column="prompt", encoding="latin-1"
    )
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    if not Path(vectordb_file_path).exists():
        create_vector_db()

    vectordb = FAISS.load_local(
        vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do i need to know Python?"))

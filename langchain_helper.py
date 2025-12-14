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


def create_vector_db():
    loader = CSVLoader(
        file_path="codebasics_faqs.csv", source_column="prompt", encoding="latin-1"
    )
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
    )

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do i need to know Python?"))

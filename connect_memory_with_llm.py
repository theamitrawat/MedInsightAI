import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from embeddings import ONNXEmbeddings, find_embedding_model_dir, get_project_root


BASE_DIR = get_project_root()
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
GROQ_MODEL = "openai/gpt-oss-120b"

PROMPT_TEMPLATE = """
Use the context below to answer the user's question.
If the answer is not in the context, say that it is not available in the
knowledge base. Do not make up medical information.

Context:
{context}

Question:
{question}

Answer:
"""


load_dotenv(BASE_DIR / ".env", override=False)


def load_llm() -> ChatGroq:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to your .env file.")

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0,
        groq_api_key=groq_api_key,
    )


def load_vectorstore() -> FAISS:
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found at vectorstore/db_faiss. "
            "Run `python create_memory_for_llm.py` first."
        )

    print(f"Loading vector store from {DB_FAISS_PATH}...")
    embedding_model = ONNXEmbeddings(find_embedding_model_dir())
    db = FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print("Vector store loaded.")
    return db


def format_context(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def answer_question(db: FAISS, llm: ChatGroq, question: str) -> dict:
    docs = db.as_retriever(search_kwargs={"k": 3}).invoke(question)
    chain = PromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(docs), "question": question})
    return {"answer": answer, "source_documents": docs}


def main() -> None:
    print("=" * 50)
    print("  MedInsight AI - CLI Mode")
    print("=" * 50)

    try:
        db = load_vectorstore()
        llm = load_llm()
    except (EnvironmentError, FileNotFoundError) as error:
        print(f"\nSetup Error: {error}")
        raise SystemExit(1)

    print("\nType a medical question below, or type 'quit' to exit.\n")

    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            response = answer_question(db, llm, user_query)
            print(f"\nMedInsight AI: {response['answer']}")

            source_docs = response.get("source_documents", [])
            if source_docs:
                print("\nSources:")
                for index, doc in enumerate(source_docs, 1):
                    page = doc.metadata.get("page", "N/A")
                    source = os.path.basename(doc.metadata.get("source", "Unknown"))
                    print(f"  [{index}] {source} - Page {page}")
            print()
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()

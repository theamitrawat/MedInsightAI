import datetime
import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from embeddings import ONNXEmbeddings, find_embedding_model_dir, get_project_root


BASE_DIR = get_project_root()
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
LOG_DIR = BASE_DIR / "logs"
QUERY_LOG_FILE = LOG_DIR / "queries.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

SUGGESTED_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed and treated?",
    "What causes anemia and how is it managed?",
    "Explain the difference between Type 1 and Type 2 diabetes.",
    "What are common side effects of antibiotics?",
    "How does the immune system fight infections?",
]

SYSTEM_PROMPT = """
You are MedInsight AI, a medical question-answering assistant.
Use only the provided context. If the context does not contain enough
information, say that it is not available in the knowledge base.

When the user asks about an illness or medical condition, use this format:

1. Overview - Brief description of the condition
2. Symptoms - Key symptoms as bullet points
3. Precautions - Prevention and safety measures
4. Medicines / Treatment - Common treatments mentioned in the context

Do not invent medical facts. Always end with a reminder to consult a
qualified healthcare professional.

Context:
{context}
"""


load_dotenv(BASE_DIR / ".env", override=False)

for key in ("GROQ_API_KEY", "HF_TOKEN"):
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

os.environ.setdefault("HF_HOME", str(BASE_DIR / ".model_cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(BASE_DIR / ".model_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / ".model_cache"))


def setup_logging() -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=LOG_DIR / "app.log",
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    except OSError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )


setup_logging()


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found at vectorstore/db_faiss. "
            "Run `python create_memory_for_llm.py` first."
        )

    embedding_model = ONNXEmbeddings(find_embedding_model_dir())
    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner="Initializing AI model...")
def get_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to .env locally or Streamlit secrets."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        groq_api_key=groq_api_key,
        streaming=True,
    )


def retrieve_with_scores(vectorstore, query: str, k: int = TOP_K):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )
    docs = retriever.invoke(query)
    scored_docs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    score_map = {doc.page_content[:120]: score for doc, score in scored_docs}
    return docs, score_map


def format_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Source {index + 1}]\n{doc.page_content}"
        for index, doc in enumerate(docs)
    )


def build_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def log_query(query: str, answer: str, latency_ms: int, num_sources: int) -> None:
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "answer_preview": answer[:200],
        "latency_ms": latency_ms,
        "num_sources": num_sources,
        "model": GROQ_MODEL,
    }

    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    except OSError:
        pass

    logging.info("Query completed | latency=%sms | sources=%s", latency_ms, num_sources)


def export_chat(messages: list[dict]) -> str:
    lines = [
        "MedInsight AI - Chat Export",
        f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]

    for message in messages:
        role = "You" if message["role"] == "user" else "MedInsight AI"
        lines.extend([f"[{role}]", message["content"], ""])

    return "\n".join(lines)


def render_sources(docs, score_map: dict) -> None:
    with st.expander("View sources and relevance scores", expanded=False):
        for index, doc in enumerate(docs):
            score = score_map.get(doc.page_content[:120])
            score_text = f"{score * 100:.1f}%" if score is not None else "N/A"
            page = doc.metadata.get("page", "N/A")
            source = os.path.basename(doc.metadata.get("source", "Unknown"))

            st.markdown(
                f"**Source {index + 1}** - `{source}` | Page {page} | "
                f"Relevance: **{score_text}**"
            )
            st.caption(doc.page_content[:300] + "...")
            if index < len(docs) - 1:
                st.divider()


def render_feedback(message_index: int) -> None:
    col_up, col_down, _ = st.columns([1, 1, 8])
    key = f"feedback_{message_index}"
    current_value = st.session_state.get(key)

    with col_up:
        if st.button(
            "Helpful",
            key=f"up_{message_index}",
            type="primary" if current_value == "up" else "secondary",
        ):
            st.session_state[key] = "up"

    with col_down:
        if st.button(
            "Not helpful",
            key=f"down_{message_index}",
            type="primary" if current_value == "down" else "secondary",
        ):
            st.session_state[key] = "down"


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("response_meta", {})


def render_sidebar() -> None:
    with st.sidebar:
        st.title("MedInsight AI")
        st.caption("Medical RAG chatbot")
        st.divider()

        st.markdown("#### System Info")
        st.markdown(f"**Model:** `{GROQ_MODEL}`")
        st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**Retrieval:** MMR, top {TOP_K}")
        st.markdown("**Vector DB:** FAISS")
        st.divider()

        messages = st.session_state.get("messages", [])
        total_queries = sum(1 for item in messages if item["role"] == "user")
        helpful = sum(
            1
            for key, value in st.session_state.items()
            if key.startswith("feedback_") and value == "up"
        )
        not_helpful = sum(
            1
            for key, value in st.session_state.items()
            if key.startswith("feedback_") and value == "down"
        )

        st.markdown("#### Session Stats")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Queries", total_queries)
        col_b.metric("Good", helpful)
        col_c.metric("Bad", not_helpful)
        st.divider()

        if messages:
            st.download_button(
                "Export Chat",
                data=export_chat(messages),
                file_name=f"medinsight_{datetime.date.today()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("Clear Chat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.caption(
            "For informational and educational use only. "
            "Not a substitute for professional medical advice."
        )


def render_chat_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] != "assistant":
                continue

            meta = st.session_state.response_meta.get(index, {})
            if meta:
                col_a, col_b, col_c = st.columns(3)
                col_a.caption(f"{meta.get('latency_ms', '?')} ms")
                col_b.caption(f"{meta.get('num_sources', '?')} sources")
                col_c.caption(GROQ_MODEL.split("/")[-1])

            if "docs" in meta:
                render_sources(meta["docs"], meta.get("score_map", {}))

            render_feedback(index)


def render_suggested_questions() -> None:
    if st.session_state.messages:
        return

    st.markdown("#### Try asking")
    columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if columns[index % 2].button(
            question,
            key=f"suggest_{index}",
            use_container_width=True,
        ):
            st.session_state["pending_input"] = question
            st.rerun()


def answer_question(vectorstore, llm, user_input: str) -> None:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Searching knowledge base..."):
        docs, score_map = retrieve_with_scores(vectorstore, user_input)
        context = format_context(docs)

    chain = build_chat_prompt() | llm | StrOutputParser()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()

        try:
            for chunk in chain.stream(
                {
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": user_input,
                }
            ):
                full_response += chunk
                placeholder.markdown(full_response + "...")

            placeholder.markdown(full_response)
            latency_ms = int((time.time() - start_time) * 1000)

            col_a, col_b, col_c = st.columns(3)
            col_a.caption(f"{latency_ms} ms")
            col_b.caption(f"{len(docs)} sources")
            col_c.caption(GROQ_MODEL.split("/")[-1])
            render_sources(docs, score_map)

            assistant_index = len(st.session_state.messages)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            st.session_state.response_meta[assistant_index] = {
                "latency_ms": latency_ms,
                "num_sources": len(docs),
                "docs": docs,
                "score_map": score_map,
            }

            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=full_response))
            st.session_state.chat_history = st.session_state.chat_history[-12:]

            render_feedback(assistant_index)
            log_query(user_input, full_response, latency_ms, len(docs))

        except Exception as error:
            st.error(f"Error while generating answer: {error}")
            logging.exception("Query failed")


def render_about_tab() -> None:
    st.markdown("## About MedInsight AI")
    st.markdown(
        """
MedInsight AI is a student-friendly medical RAG chatbot. It retrieves relevant
medical text from a local FAISS knowledge base and sends that context to a Groq
LLM for a structured answer.

### Workflow
User question -> ONNX embeddings -> FAISS retrieval -> Groq LLM -> Streamlit chat

### Important note
This app is for education and portfolio demonstration only. It should not be
used for diagnosis, treatment decisions, or emergency medical advice.
        """
    )


def main() -> None:
    st.set_page_config(page_title="MedInsight AI", page_icon="MI", layout="wide")
    initialize_session_state()
    render_sidebar()

    try:
        vectorstore = get_vectorstore()
        llm = get_llm()
    except FileNotFoundError as error:
        st.error("Required local files are missing.")
        st.info(str(error))
        st.stop()
    except EnvironmentError as error:
        st.error("Configuration error.")
        st.info(str(error))
        st.stop()
    except Exception as error:
        st.error(f"Failed to load app resources: {error}")
        st.stop()

    tab_chat, tab_about = st.tabs(["Chat", "About"])

    with tab_chat:
        st.markdown("## MedInsight AI")
        st.caption("Ask a medical question and get an answer grounded in the knowledge base.")
        render_suggested_questions()
        render_chat_history()

        pending_input = st.session_state.pop("pending_input", None)
        user_input = st.chat_input("Ask a medical question...") or pending_input
        if user_input:
            answer_question(vectorstore, llm, user_input)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

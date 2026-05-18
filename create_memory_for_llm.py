import csv
import glob
import os
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import ONNXEmbeddings, find_embedding_model_dir, get_project_root


BASE_DIR = get_project_root()
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 256


def load_pdfs(data_path: Path) -> list[Document]:
    pdf_files = sorted(glob.glob(str(data_path / "*.pdf")))
    if not pdf_files:
        print("  No PDF files found.")
        return []

    documents = []
    for index, pdf_path in enumerate(pdf_files, 1):
        name = os.path.basename(pdf_path)
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  [{index}/{len(pdf_files)}] Loading {name} ({size_mb:.1f} MB)...", end=" ")
        start_time = time.time()

        try:
            docs = PyPDFLoader(pdf_path).load()
            documents.extend(docs)
            print(f"ok, {len(docs)} pages ({time.time() - start_time:.1f}s)")
        except Exception as error:
            print(f"skipped: {error}")

    print(f"  PDFs total: {len(documents)} pages from {len(pdf_files)} files.")
    return documents


def load_xml_qa(data_path: Path) -> list[Document]:
    xml_files = glob.glob(str(data_path / "**" / "*.xml"), recursive=True)
    if not xml_files:
        print("  No XML files found.")
        return []

    documents = []
    skipped = 0

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
            focus = root.findtext("Focus", default="General Medicine").strip()
            source_name = root.get("source", "MedicalQA")

            for qa_pair in root.findall(".//QAPair"):
                question = (qa_pair.findtext("Question") or "").strip()
                answer = (qa_pair.findtext("Answer") or "").strip()
                if not question or not answer:
                    continue

                answer = re.sub(r"\s+", " ", answer)
                documents.append(
                    Document(
                        page_content=(
                            f"Topic: {focus}\n"
                            f"Question: {question}\n"
                            f"Answer: {answer}"
                        ),
                        metadata={
                            "source": xml_path,
                            "source_name": source_name,
                            "focus": focus,
                            "type": "QA",
                        },
                    )
                )
        except Exception:
            skipped += 1

    print(
        f"  XML QA: {len(documents)} Q&A pairs "
        f"({len(xml_files)} files, {skipped} skipped)."
    )
    return documents


def load_csvs(data_path: Path) -> list[Document]:
    documents = []

    descriptions = {}
    desc_path = data_path / "symptom_Description.csv"
    if desc_path.exists():
        with open(desc_path, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                disease = row.get("Disease", "").strip()
                description = row.get("Description", "").strip()
                if disease and description:
                    descriptions[disease.lower()] = (disease, description)

    precautions = {}
    prec_path = data_path / "symptom_precaution.csv"
    if prec_path.exists():
        with open(prec_path, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                disease = row.get("Disease", "").strip()
                values = [
                    row.get(f"Precaution_{index}", "").strip()
                    for index in range(1, 5)
                    if row.get(f"Precaution_{index}", "").strip()
                ]
                if disease and values:
                    precautions[disease.lower()] = (disease, values)

    for key in sorted(set(descriptions) | set(precautions)):
        disease_name, description = descriptions.get(key, (key.title(), ""))
        _, precaution_values = precautions.get(key, (key.title(), []))

        parts = [f"Disease: {disease_name}"]
        if description:
            parts.append(f"Description: {description}")
        if precaution_values:
            parts.append(
                "Precautions:\n"
                + "\n".join(f"- {precaution}" for precaution in precaution_values)
            )

        documents.append(
            Document(
                page_content="\n".join(parts),
                metadata={
                    "source": "symptom_csv",
                    "disease": disease_name,
                    "type": "structured_disease_info",
                },
            )
        )

    severity_path = data_path / "Symptom-severity.csv"
    if severity_path.exists():
        lines = []
        with open(severity_path, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                symptom = row.get("Symptom", "").strip()
                weight = row.get("weight", "").strip()
                if symptom and weight:
                    lines.append(f"{symptom}: severity weight {weight}")

        for start in range(0, len(lines), 50):
            documents.append(
                Document(
                    page_content=(
                        "Symptom Severity Reference:\n"
                        + "\n".join(lines[start : start + 50])
                    ),
                    metadata={
                        "source": "Symptom-severity.csv",
                        "type": "severity_data",
                    },
                )
            )

    dataset_path = data_path / "dataset.csv"
    if dataset_path.exists():
        disease_symptoms: dict[str, list[str]] = {}
        with open(dataset_path, encoding="utf-8") as file:
            reader = csv.DictReader(file)
            disease_col = next(
                (
                    column
                    for column in reader.fieldnames or []
                    if "disease" in column.lower() or "prognosis" in column.lower()
                ),
                None,
            )

            if disease_col:
                for row in reader:
                    disease = row.get(disease_col, "").strip()
                    if not disease:
                        continue

                    symptoms = [
                        value.strip()
                        for key, value in row.items()
                        if key != disease_col and value and value.strip() not in ("0", "")
                    ]
                    disease_symptoms.setdefault(disease, []).extend(symptoms)

        for disease, symptoms in disease_symptoms.items():
            unique_symptoms = list(dict.fromkeys(symptoms))
            if unique_symptoms:
                documents.append(
                    Document(
                        page_content=(
                            f"Disease: {disease}\nAssociated Symptoms:\n"
                            + "\n".join(f"- {symptom}" for symptom in unique_symptoms[:30])
                        ),
                        metadata={
                            "source": "dataset.csv",
                            "disease": disease,
                            "type": "symptom_mapping",
                        },
                    )
                )

    print(f"  CSV: {len(documents)} structured documents.")
    return documents


def clean_documents(docs: list[Document]) -> list[Document]:
    cleaned = []
    for doc in docs:
        text = re.sub(r"\n{3,}", "\n\n", doc.page_content)
        text = re.sub(r" {2,}", " ", text).strip()
        if len(text) < 30:
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned


def create_chunks(docs: list[Document]) -> list[Document]:
    structured_types = {
        "QA",
        "structured_disease_info",
        "severity_data",
        "symptom_mapping",
    }
    pdf_docs = [doc for doc in docs if doc.metadata.get("type") not in structured_types]
    small_docs = [doc for doc in docs if doc.metadata.get("type") in structured_types]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pdf_chunks = splitter.split_documents(pdf_docs)
    chunks = pdf_chunks + small_docs
    avg_chars = sum(len(chunk.page_content) for chunk in chunks) // max(len(chunks), 1)

    print(
        f"  Chunks: {len(chunks)} total "
        f"({len(pdf_chunks)} PDF + {len(small_docs)} structured), "
        f"average {avg_chars} chars"
    )
    return chunks


def get_embedding_model() -> ONNXEmbeddings:
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    model = ONNXEmbeddings(find_embedding_model_dir())
    print("  Embedding model ready.")
    return model


def build_vectorstore(chunks: list[Document], embedding_model: ONNXEmbeddings) -> FAISS:
    if not chunks:
        raise ValueError("No documents were loaded. Add files to the data folder first.")

    DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)
    db = None
    start_time = time.time()

    print(f"  Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        if db is None:
            db = FAISS.from_documents(batch, embedding_model)
        else:
            db.add_documents(batch)

        done = min(start + BATCH_SIZE, len(chunks))
        percent = int(done / len(chunks) * 100)
        print(f"  {percent:3d}% ({done}/{len(chunks)})", end="\r", flush=True)

    db.save_local(str(DB_FAISS_PATH))
    print(f"\n  Saved vector store to {DB_FAISS_PATH} ({time.time() - start_time:.1f}s)")
    return db


def main() -> None:
    print("=" * 62)
    print("  MedInsight AI - Knowledge Base Builder")
    print("=" * 62)
    total_start = time.time()

    print("\n[1/6] Loading PDFs...")
    pdf_docs = load_pdfs(DATA_PATH)

    print("\n[2/6] Loading XML QA datasets...")
    xml_docs = load_xml_qa(DATA_PATH)

    print("\n[3/6] Loading CSV structured data...")
    csv_docs = load_csvs(DATA_PATH)

    all_docs = pdf_docs + xml_docs + csv_docs
    print(f"\n      Raw total: {len(all_docs)} documents")

    print("\n[4/6] Cleaning...")
    all_docs = clean_documents(all_docs)
    print(f"  After cleaning: {len(all_docs)} documents")

    print("\n[5/6] Chunking...")
    chunks = create_chunks(all_docs)

    print("\n[6/6] Building FAISS vector store...")
    embedding_model = get_embedding_model()
    build_vectorstore(chunks, embedding_model)

    elapsed = time.time() - total_start
    print("\n" + "=" * 62)
    print(f"  Done in {elapsed:.1f}s")
    print(f"  PDFs        : {len(pdf_docs):>6} pages")
    print(f"  XML QA      : {len(xml_docs):>6} Q&A pairs")
    print(f"  CSV records : {len(csv_docs):>6} documents")
    print(f"  Total chunks: {len(chunks):>6}")
    print("\n  Next: streamlit run medibot.py")
    print("=" * 62)


if __name__ == "__main__":
    main()

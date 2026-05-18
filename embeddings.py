from pathlib import Path

from langchain_core.embeddings import Embeddings


MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = ".model_cache"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_model_cache_path() -> Path:
    return get_project_root() / MODEL_CACHE_DIR


def find_embedding_model_dir() -> Path:
    """Find the downloaded Hugging Face snapshot that contains the ONNX model."""
    cache_dir = get_model_cache_path()
    model_root = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"

    candidates = [model_root]
    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        candidates.extend(sorted(snapshots_dir.glob("*")))

    for path in candidates:
        if (path / "onnx" / "model.onnx").exists() and (path / "tokenizer.json").exists():
            return path

    raise FileNotFoundError(
        "Embedding model files were not found. Run `python download_model.py` first."
    )


class ONNXEmbeddings(Embeddings):
    """Small ONNX embedder for all-MiniLM-L6-v2 without importing torch."""

    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        model_path = model_dir / "onnx" / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing ONNX model files in {model_dir}")

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self._tokenizer.enable_truncation(max_length=128)

    def _mean_pool(self, token_embeddings, attention_mask):
        import numpy as np

        mask = attention_mask[..., np.newaxis].astype(float)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).tolist()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.array(
            [item.attention_mask for item in encoded],
            dtype=np.int64,
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        output = self._session.run(None, inputs)
        return self._mean_pool(output[0], attention_mask)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for start in range(0, len(texts), 32):
            embeddings.extend(self._embed(texts[start : start + 32]))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

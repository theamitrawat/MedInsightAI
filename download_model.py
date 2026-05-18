from pathlib import Path

from huggingface_hub import snapshot_download

from embeddings import MODEL_CACHE_DIR, MODEL_REPO, find_embedding_model_dir


def main() -> None:
    print(f"Downloading {MODEL_REPO} into {MODEL_CACHE_DIR}/ ...")
    Path(MODEL_CACHE_DIR).mkdir(exist_ok=True)

    local_path = snapshot_download(
        repo_id=MODEL_REPO,
        cache_dir=MODEL_CACHE_DIR,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "onnx/model.onnx",
        ],
    )

    print(f"\nModel snapshot downloaded to: {local_path}")
    print(f"ONNX model found at: {find_embedding_model_dir()}")
    print("\nNext steps:")
    print("  1. Run: python create_memory_for_llm.py")
    print("  2. Run: streamlit run medibot.py")


if __name__ == "__main__":
    main()

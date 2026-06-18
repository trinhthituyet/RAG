"""Loading raw markdown documents from the knowledge-base directory."""

from pathlib import Path

from config import KNOWLEDGE_BASE_PATH


class DocumentLoader:
    """Walk a knowledge-base directory and yield {type, source, text} dicts."""

    def __init__(self, base_path: Path = KNOWLEDGE_BASE_PATH):
        self.base_path = base_path

    def load(self) -> list[dict]:
        documents = []
        for folder in self.base_path.iterdir():
            doc_type = folder.name
            for file in folder.rglob("*.md"):
                with open(file, "r", encoding="utf-8") as f:
                    documents.append({
                        "type": doc_type,
                        "source": file.as_posix(),
                        "text": f.read(),
                    })
        print(f"Loaded {len(documents)} documents")
        return documents

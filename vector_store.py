"""Persistent vector store wrapper around chromadb."""

from chromadb import PersistentClient

from config import COLLECTION_NAME, DB_NAME
from models import Result


class VectorStore:
    """Owns the chroma client + collection and exposes rebuild/query."""

    def __init__(self, embeddings, db_name: str = DB_NAME, collection_name: str = COLLECTION_NAME):
        self.embeddings = embeddings
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = PersistentClient(path=db_name)
        self._collection = self._client.get_or_create_collection(collection_name)

    def rebuild(self, chunks: list[Result]) -> None:
        if self.collection_name in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(self.collection_name)

        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        self._collection = self._client.get_or_create_collection(self.collection_name)
        ids = [str(i) for i in range(len(chunks))]
        metas = [chunk.metadata for chunk in chunks]
        self._collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        print(f"Vectorstore created with {self._collection.count()} documents")

    def query(self, question: str, k: int) -> list[Result]:
        query_vec = self.embeddings.embed_documents([question])[0]
        results = self._collection.query(query_embeddings=[query_vec], n_results=k)
        return [
            Result(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

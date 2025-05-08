import numpy as np
from src.language_model_interface import LanguageModelInterface
from src.typed_dicts import LanguageModelConfig, VectorDBConfig
#from google.genai.types import ContentEmbedding
import chromadb

class VectorDB:
    def __init__(self, collection_name: str, config:VectorDBConfig, lm_config:LanguageModelConfig):
        lm = LanguageModelInterface(lm_config) 
        self.config = config
        class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                return lm.create_embedding(text=input, task_type=config.task_type) #type:ignore

        # persist to reduce api calls
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())

    def add_document(self, document: str, id: str):
        """
        Add a document to the vector database.
        :param document: Document to add.
        """
        self.collection.add(
            documents=[document],
            ids=[id]
        )
    
    def get_n_closest(self, text: str, n: int) -> list[str]:
        """
        Get the n closest documents to the given text.
        :param text: Text to compare against.
        :param n: Number of closest documents to retrieve.
        :return: List of closest documents.
        """
        results = self.collection.query(
            query_texts=[text],
            n_results=n
        )

        return [str(item) for item in results['ids'][0]]
    
    def remove_document(self, id: str):
        """
        Remove a document from the vector database.
        :param id: ID of the document to remove.
        """
        self.collection.delete(ids=[id])
    
    def embed(self, text: str) -> list[float]:
        return self.collection._embedding_function([text])[0]
    
    def get_cosine_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.db.embed(text_a)
        emb_b = self.db.embed(text_b)

        if emb_a is None or emb_b is None:
            return 0.0

        a, b = np.array(emb_a), np.array(emb_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

"""
Service d'embedding — Convertit du texte en vecteurs
Utilise sentence-transformers pour générer des embeddings réels
"""
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingService:
    """Service pour générer des embeddings à partir de texte."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise le service d'embedding.
        
        Args:
            model_name: Nom du modèle sentence-transformers
                       'all-MiniLM-L6-v2' = 384 dimensions, rapide, bon compromis
                       'all-mpnet-base-v2' = 768 dimensions, plus précis mais plus lent
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"  [Embedding] Modèle '{model_name}' chargé (dimension: {self.dimension})")
    
    def encode(self, text: str) -> List[float]:
        """
        Convertit un texte en vecteur d'embedding.
        
        Args:
            text: Texte à encoder
            
        Returns:
            Liste de floats représentant le vecteur
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convertit plusieurs textes en vecteurs (plus efficace que encode() en boucle).
        
        Args:
            texts: Liste de textes
            
        Returns:
            Liste de vecteurs
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
"""
Générateurs de requêtes — Un par type de base
Utilisent le schéma réel de la base pour générer des prompts dynamiques
"""
from .llm_service import LLMService
from .schema_inspectors import (
    MongoSchemaInspector, 
    QdrantSchemaInspector, 
    Neo4jSchemaInspector
)
from .prompt_generators import (
    generate_mongo_prompt,
    generate_qdrant_prompt,
    generate_neo4j_prompt
)
from .models import MongoConfig, QdrantConfig, Neo4jConfig
# from .embeding_service import EmbeddingService
import json
import re


class QueryGenerator:
    """Classe de base pour tous les générateurs."""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.schema = None
    
    def inspect_schema(self):
        """Récupère le schéma de la base (à implémenter par les sous-classes)."""
        raise NotImplementedError
    
    def generate_prompt(self) -> str:
        """Génère le prompt avec le schéma (à implémenter par les sous-classes)."""
        raise NotImplementedError
    
    def generate(self, user_query: str) -> str:
        """
        Génère une requête à partir du texte naturel.
        Inspecte le schéma au premier appel.
        """
        # Inspecter le schéma si pas encore fait
        if self.schema is None:
            self.schema = self.inspect_schema()
        
        # Générer le prompt avec le schéma
        prompt = self.generate_prompt()
        
        # Générer la requête via LLM
        return self.llm.generate(prompt, user_query)

    def extract_json(self, text: str) -> dict:
        
        """
        Extrait le premier JSON valide trouvé dans un texte.
        Lève ValueError si impossible.
        """

        # Tentative simple : entre premier { et dernier }
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("Aucun JSON détecté")

        candidate = text[start:end+1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Fallback regex (multiline, greedy)
        matches = re.findall(r"\{.*\}", text, re.DOTALL)

        for m in matches:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError("JSON invalide ou non trouvable")

class MongoQueryGenerator(QueryGenerator):
    """Génère des filtres JSON MongoDB."""
    
    def __init__(self, llm_service: LLMService, config: MongoConfig):
        super().__init__(llm_service)
        self.config = config
    
    def inspect_schema(self):
        inspector = MongoSchemaInspector(self.config.uri, self.config.database)
        return inspector.inspect()
    
    def generate_prompt(self) -> str:
        return generate_mongo_prompt(self.schema)

    def generate(self, user_query: str) -> str:
        response = super().generate(user_query)
        return super().extract_json(response)

class QdrantQueryGenerator(QueryGenerator):
    """Extrait les termes clés pour Qdrant."""
    
    def __init__(self, llm_service: LLMService, config: QdrantConfig):#, embedding_service: EmbeddingService):
        super().__init__(llm_service)
        self.config = config
        # self.embedding_service = embedding_service
    
    def inspect_schema(self):
        inspector = QdrantSchemaInspector(self.config.url)
        return inspector.inspect()
    
    def generate_prompt(self) -> str:
        return generate_qdrant_prompt(self.schema)
    
    def generate(self, user_query: str) -> dict:
        """
        Génère les termes de recherche et le vecteur d'embedding.
        Retourne un dict directement utilisable avec Qdrant.
        """
        # Inspecter le schéma si pas encore fait
        if self.schema is None:
            self.schema = self.inspect_schema()
        
        # Générer le prompt avec le schéma
        prompt = self.generate_prompt()
        
        # Extraire les termes clés via LLM
        search_terms = self.llm.generate(prompt, user_query)
        
        # Générer l'embedding à partir des termes
        # query_vector = self.embedding_service.encode(search_terms)
        
        return {"search_terms": search_terms}
        
        # return {
        #     "query_vector": query_vector,
        #     "search_terms": search_terms,
        #     "vector_dimension": len(query_vector)
        # }


class Neo4jQueryGenerator(QueryGenerator):
    """Génère des requêtes Cypher Neo4j."""
    
    def __init__(self, llm_service: LLMService, config: Neo4jConfig):
        super().__init__(llm_service)
        self.config = config
    
    def inspect_schema(self):
        inspector = Neo4jSchemaInspector(
            self.config.uri, 
            self.config.user, 
            self.config.password
        )
        return inspector.inspect()
    
    def generate_prompt(self) -> str:
        return generate_neo4j_prompt(self.schema)

    def generate(self, user_query: str) -> str:
        response = super().generate(user_query)
        return super().extract_json(response)["cypher"]
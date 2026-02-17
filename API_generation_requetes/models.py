"""
Modèles de configuration pour les bases de données
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class DatabaseConfig(BaseModel):
    """Configuration de base pour toutes les bases."""
    db_type: Literal["mongodb", "qdrant", "neo4j"]
    name: str = Field(description="Nom identifiant unique de la config")


class MongoConfig(DatabaseConfig):
    """Configuration MongoDB."""
    db_type: Literal["mongodb"] = "mongodb"
    uri: str = Field(description="URI de connexion MongoDB")
    database: str = Field(description="Nom de la base de données")
    
    class Config:
        json_schema_extra = {
            "example": {
                "db_type": "mongodb",
                "name": "pharma_db",
                "uri": "mongodb://admin:password@localhost:27017",
                "database": "pharma"
            }
        }


class QdrantConfig(DatabaseConfig):
    """Configuration Qdrant."""
    db_type: Literal["qdrant"] = "qdrant"
    url: str = Field(description="URL du serveur Qdrant")
    
    class Config:
        json_schema_extra = {
            "example": {
                "db_type": "qdrant",
                "name": "vectors_db",
                "url": "http://localhost:6333"
            }
        }


class Neo4jConfig(DatabaseConfig):
    """Configuration Neo4j."""
    db_type: Literal["neo4j"] = "neo4j"
    uri: str = Field(description="URI Bolt de connexion")
    user: str = Field(description="Nom d'utilisateur")
    password: str = Field(description="Mot de passe")
    
    class Config:
        json_schema_extra = {
            "example": {
                "db_type": "neo4j",
                "name": "graph_db",
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
        }
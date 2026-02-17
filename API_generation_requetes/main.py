"""
API FastAPI — Génération de requêtes en langage naturel
────────────────────────────────────────────────────────

Architecture modulaire avec configuration dynamique des bases.

Endpoints:
  POST /config/add          → Ajoute une config de base de données
  POST /config/batch        → Ajoute plusieurs configs en une seule requête
  GET  /config/list         → Liste toutes les configs
  DELETE /config/{name}     → Supprime une config
  POST /generate/{db_name}  → Génère une requête pour une base configurée
"""
# import sys
# from pathlib import Path

# # Ajouter le dossier courant au PYTHONPATH pour les imports
# sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Union, List

from .llm_service import LLMService
from .query_generators import (
    MongoQueryGenerator,
    QdrantQueryGenerator,
    Neo4jQueryGenerator
)
from .models import MongoConfig, QdrantConfig, Neo4jConfig
# from .embeding_service import EmbeddingService


# ════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════

load_dotenv()

app = FastAPI(
    title="Dynamic Query Generator API",
    description="Génère des requêtes depuis du langage naturel avec introspection automatique du schéma",
    version="2.0.0"
)

# Service LLM global
llm = LLMService()
# embedding = EmbeddingService()  # Charge le modèle d'embedding au démarrage

# Stockage des configs et générateurs (en mémoire)
# En production, utiliser une vraie base de données
configs: Dict[str, Union[MongoConfig, QdrantConfig, Neo4jConfig]] = {}
generators: Dict[str, Union[MongoQueryGenerator, QdrantQueryGenerator, Neo4jQueryGenerator]] = {}


# ════════════════════════════════════════════════════════
#  MODÈLES
# ════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Trouve les antibiotiques qui coûtent moins de 10 euros"
            }
        }


class QueryResponse(BaseModel):
    database_name: str
    database_type: str
    user_query: str
    generated_query: Union[dict, str]  # dict pour MongoDB/Qdrant, str pour Neo4j
    schema_used: dict


class BatchConfigRequest(BaseModel):
    """Modèle pour l'ajout de plusieurs configs en une fois."""
    configs: List[Union[MongoConfig, QdrantConfig, Neo4jConfig]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "configs": [
                    {
                        "db_type": "mongodb",
                        "name": "pharma_db",
                        "uri": "mongodb://admin:password@localhost:27018",
                        "database": "pharma"
                    },
                    {
                        "db_type": "qdrant",
                        "name": "vectors_db",
                        "url": "http://localhost:6333"
                    },
                    {
                        "db_type": "neo4j",
                        "name": "graph_db",
                        "uri": "bolt://localhost:7687",
                        "user": "neo4j",
                        "password": "password"
                    }
                ]
            }
        }


class BatchConfigResponse(BaseModel):
    """Réponse de l'ajout batch."""
    total: int
    successful: List[str]
    failed: List[Dict[str, str]]


# ════════════════════════════════════════════════════════
#  ENDPOINTS — CONFIGURATION
# ════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "Dynamic Query Generator API",
        "endpoints": {
            "config": {
                "add": "POST /config/add",
                "batch": "POST /config/batch",
                "list": "GET /config/list",
                "delete": "DELETE /config/{name}"
            },
            "generate": "POST /generate/{db_name}"
        }
    }


def _create_generator(config: Union[MongoConfig, QdrantConfig, Neo4jConfig]):
    """Crée le générateur approprié selon le type de config."""
    if config.db_type == "mongodb":
        return MongoQueryGenerator(llm, config)
    elif config.db_type == "qdrant":
        return QdrantQueryGenerator(llm, config)#, embedding)
    elif config.db_type == "neo4j":
        return Neo4jQueryGenerator(llm, config)
    else:
        raise ValueError(f"Type de base inconnu: {config.db_type}")


@app.post("/config/add")
def add_config(config: Union[MongoConfig, QdrantConfig, Neo4jConfig]):
    """
    Ajoute une configuration de base de données.
    
    Le schéma sera inspecté automatiquement lors de la première génération.
    """
    if config.name in configs:
        raise HTTPException(
            status_code=400, 
            detail=f"Une config nommée '{config.name}' existe déjà"
        )
    
    # Stocker la config
    configs[config.name] = config
    
    # Créer le générateur approprié
    try:
        generators[config.name] = _create_generator(config)
    except Exception as e:
        # Rollback si erreur
        del configs[config.name]
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création du générateur: {str(e)}"
        )
    
    return {
        "message": f"Config '{config.name}' ajoutée avec succès",
        "db_type": config.db_type,
        "name": config.name
    }


@app.post("/config/batch", response_model=BatchConfigResponse)
def add_configs_batch(request: BatchConfigRequest):
    """
    Ajoute plusieurs configurations en une seule requête.
    
    Traite toutes les configs et retourne un rapport détaillé.
    Si une config échoue, les autres continuent d'être traitées.
    """
    successful = []
    failed = []
    
    for config in request.configs:
        try:
            # Vérifier si existe déjà
            if config.name in configs:
                failed.append({
                    "name": config.name,
                    "error": f"Une config nommée '{config.name}' existe déjà"
                })
                continue
            
            # Stocker la config
            configs[config.name] = config
            
            # Créer le générateur
            try:
                generators[config.name] = _create_generator(config)
                successful.append(config.name)
            except Exception as e:
                # Rollback
                del configs[config.name]
                failed.append({
                    "name": config.name,
                    "error": f"Erreur lors de la création du générateur: {str(e)}"
                })
        
        except Exception as e:
            failed.append({
                "name": getattr(config, "name", "unknown"),
                "error": str(e)
            })
    
    return BatchConfigResponse(
        total=len(request.configs),
        successful=successful,
        failed=failed
    )


@app.get("/config/list")
def list_configs():
    """Liste toutes les configurations enregistrées."""
    return {
        "count": len(configs),
        "databases": [
            {
                "name": name,
                "type": cfg.db_type,
                "schema_inspected": generators[name].schema is not None
            }
            for name, cfg in configs.items()
        ]
    }


@app.delete("/config/{name}")
def delete_config(name: str):
    """Supprime une configuration."""
    if name not in configs:
        raise HTTPException(status_code=404, detail=f"Config '{name}' introuvable")
    
    del configs[name]
    del generators[name]
    
    return {"message": f"Config '{name}' supprimée"}


# ════════════════════════════════════════════════════════
#  ENDPOINTS — GÉNÉRATION
# ════════════════════════════════════════════════════════

@app.post("/generate/{db_name}", response_model=QueryResponse)
def generate_query(db_name: str, request: QueryRequest):
    """
    Génère une requête pour la base de données spécifiée.
    
    Le schéma est automatiquement inspecté au premier appel.
    Les appels suivants réutilisent le schéma en cache.
    """
    # Vérifier que la config existe
    if db_name not in configs:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune config nommée '{db_name}'. Utilisez POST /config/add d'abord."
        )
    
    try:
        generator = generators[db_name]
        config = configs[db_name]
        
        # Générer la requête (inspecte le schéma si besoin)
        generated = generator.generate(request.query)
        
        return QueryResponse(
            database_name=db_name,
            database_type=config.db_type,
            user_query=request.query,
            generated_query=generated,
            schema_used=generator.schema
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )


@app.post("/generate/{db_name}/refresh-schema")
def refresh_schema(db_name: str):
    """
    Force la réinspection du schéma pour une base.
    Utile si la structure de la base a changé.
    """
    if db_name not in configs:
        raise HTTPException(status_code=404, detail=f"Config '{db_name}' introuvable")
    
    try:
        generator = generators[db_name]
        generator.schema = None  # Reset le cache
        generator.schema = generator.inspect_schema()  # Réinspecter
        
        return {
            "message": f"Schéma de '{db_name}' réinspecté",
            "schema": generator.schema
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la réinspection: {str(e)}"
        )


# ════════════════════════════════════════════════════════
#  LANCEMENT
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
"""
Schema Inspectors — Récupèrent la structure des bases de données
"""
from pymongo import MongoClient
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from typing import Dict, Any


class SchemaInspector:
    """Classe de base pour l'inspection de schéma."""
    
    def inspect(self) -> Dict[str, Any]:
        """Retourne la structure de la base."""
        raise NotImplementedError


class MongoSchemaInspector(SchemaInspector):
    """Inspecte le schéma MongoDB."""
    
    def __init__(self, uri: str, database: str):
        self.client = MongoClient(uri, authSource="admin", serverSelectionTimeoutMS=5000)
        self.db = self.client[database]
    
    def inspect(self) -> Dict[str, Any]:
        """
        Analyse les collections et extrait un schéma représentatif.
        Pour chaque collection, prend un échantillon de documents.
        """
        try:
            collections = self.db.list_collection_names()
            schema = {"database": self.db.name, "collections": {}}
            
            for col_name in collections:
                # Prendre un échantillon de 5 documents
                sample = list(self.db[col_name].find().limit(5))
                
                if not sample:
                    schema["collections"][col_name] = {"fields": {}, "sample_count": 0}
                    continue
                
                # Extraire les champs et leurs types
                fields = {}
                for doc in sample:
                    for key, value in doc.items():
                        if key == "_id":
                            continue
                        
                        # Détecter le type
                        type_name = type(value).__name__
                        if isinstance(value, list) and value:
                            type_name = f"array<{type(value[0]).__name__}>"
                        
                        if key not in fields:
                            fields[key] = {"type": type_name, "examples": []}
                        
                        # Ajouter un exemple
                        if len(fields[key]["examples"]) < 2:
                            fields[key]["examples"].append(value)
                
                schema["collections"][col_name] = {
                    "fields": fields,
                    "sample_count": len(sample)
                }
            
            return schema
        
        finally:
            self.client.close()


class QdrantSchemaInspector(SchemaInspector):
    """Inspecte le schéma Qdrant."""
    
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)
    
    def inspect(self) -> Dict[str, Any]:
        """
        Récupère les collections et leurs configurations.
        """
        try:
            collections_info = self.client.get_collections()
            schema = {"collections": {}}
            
            for col in collections_info.collections:
                col_name = col.name
                col_info = self.client.get_collection(col_name)
                
                # Récupérer quelques points pour voir les payloads
                points = self.client.scroll(
                    collection_name=col_name,
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                # Extraire les champs des payloads
                payload_fields = {}
                for point in points:
                    if point.payload:
                        for key, value in point.payload.items():
                            if key not in payload_fields:
                                payload_fields[key] = {
                                    "type": type(value).__name__,
                                    "examples": []
                                }
                            if len(payload_fields[key]["examples"]) < 2:
                                payload_fields[key]["examples"].append(value)
                
                schema["collections"][col_name] = {
                    "vector_size": col_info.config.params.vectors.size,
                    "distance": col_info.config.params.vectors.distance.name,
                    "points_count": col_info.points_count,
                    "payload_fields": payload_fields
                }
            
            return schema
        
        finally:
            self.client.close()


class Neo4jSchemaInspector(SchemaInspector):
    """Inspecte le schéma Neo4j."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def inspect(self) -> Dict[str, Any]:
        """
        Récupère les labels, relations et propriétés du graphe.
        """
        try:
            with self.driver.session() as session:
                # Récupérer les labels de nœuds
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]
                
                # Récupérer les types de relations
                rels_result = session.run("CALL db.relationshipTypes()")
                relationships = [record["relationshipType"] for record in rels_result]
                
                # Pour chaque label, récupérer les propriétés
                nodes_schema = {}
                for label in labels:
                    props_result = session.run(
                        f"MATCH (n:{label}) RETURN properties(n) AS props LIMIT 3"
                    )
                    
                    properties = {}
                    for record in props_result:
                        for key, value in record["props"].items():
                            if key not in properties:
                                properties[key] = {
                                    "type": type(value).__name__,
                                    "examples": []
                                }
                            if len(properties[key]["examples"]) < 2:
                                properties[key]["examples"].append(value)
                    
                    # Compter les nœuds
                    count_result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                    count = count_result.single()["count"]
                    
                    nodes_schema[label] = {
                        "properties": properties,
                        "count": count
                    }
                
                # Pour chaque relation, trouver les patterns
                relationships_schema = {}
                for rel_type in relationships:
                    pattern_result = session.run(
                        f"""
                        MATCH (a)-[r:{rel_type}]->(b)
                        RETURN labels(a)[0] AS from_label, 
                               labels(b)[0] AS to_label
                        LIMIT 1
                        """
                    )
                    pattern = pattern_result.single()
                    if pattern:
                        relationships_schema[rel_type] = {
                            "from": pattern["from_label"],
                            "to": pattern["to_label"]
                        }
                
                return {
                    "nodes": nodes_schema,
                    "relationships": relationships_schema
                }
        
        finally:
            self.driver.close()
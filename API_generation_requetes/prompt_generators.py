"""
Générateurs de prompts dynamiques — Injectent le schéma réel de la base
"""
import json


def generate_mongo_prompt(schema: dict) -> str:
    """
    Génère un prompt MongoDB avec le schéma réel de la base.
    
    Args:
        schema: Structure retournée par MongoSchemaInspector
    """
    collections_desc = []
    
    for col_name, col_data in schema["collections"].items():
        fields_desc = []
        for field_name, field_info in col_data["fields"].items():
            examples = field_info.get("examples", [])
            examples_str = ", ".join(f'"{ex}"' if isinstance(ex, str) else str(ex) 
                                    for ex in examples[:2])
            fields_desc.append(
                f'  "{field_name}": {field_info["type"]}  // Ex: {examples_str}'
            )
        
        collections_desc.append(f"""
Collection "{col_name}":
{{
{chr(10).join(fields_desc)}
}}""")
    
    prompt = f"""Tu es un expert MongoDB. Convertis la question en langage naturel en un filtre JSON MongoDB valide.

BASE DE DONNÉES: {schema["database"]}

SCHÉMA RÉEL:
{chr(10).join(collections_desc)}

RÈGLES:
- Réponds UNIQUEMENT avec le JSON du filtre, rien d'autre
- Utilise les opérateurs: $gt, $lt, $gte, $lte, $in, $regex (avec "i"), $elemMatch
- Pour les arrays, utilise $in pour chercher un élément
- Pas de texte explicatif, pas de markdown, pas de ```json

EXEMPLES DE SYNTAXE:
{{"field": "value"}}
{{"field": {{"$lt": 10}}}}
{{"array_field": {{"$in": ["value"]}}}}
{{"text_field": {{"$regex": "pattern", "$options": "i"}}}}

Question suivante:"""
    
    return prompt


def generate_qdrant_prompt(schema: dict) -> str:
    """
    Génère un prompt Qdrant avec les collections et payloads réels.
    
    Args:
        schema: Structure retournée par QdrantSchemaInspector
    """
    collections_desc = []
    
    for col_name, col_data in schema["collections"].items():
        payload_fields = []
        for field_name, field_info in col_data.get("payload_fields", {}).items():
            examples = field_info.get("examples", [])
            examples_str = ", ".join(f'"{ex}"' if isinstance(ex, str) else str(ex) 
                                    for ex in examples[:2])
            payload_fields.append(f"  - {field_name} ({field_info['type']}): {examples_str}")
        
        collections_desc.append(f"""
Collection "{col_name}":
  - Dimension vecteurs: {col_data["vector_size"]}
  - Nombre de points: {col_data["points_count"]}
  - Champs payload:
{chr(10).join(payload_fields) if payload_fields else "    (aucun)"}""")
    
    prompt = f"""Tu es un assistant qui extrait les termes clés pour une recherche vectorielle.

COLLECTIONS DISPONIBLES:
{chr(10).join(collections_desc)}

RÈGLES:
- Réponds avec une phrase courte (5-10 mots max)
- Concentre-toi sur les termes qui apparaissent dans les payloads
- Pas de mots de liaison inutiles
- Pas de JSON, pas de code, pas de markdown

EXEMPLES:
Question: "Je cherche un médicament pour la fièvre"
Réponse: fièvre médicament

Question: "Quels sont les anti-inflammatoires ?"
Réponse: anti-inflammatoire

Question suivante:"""
    
    return prompt


def generate_neo4j_prompt(schema: dict) -> str:
    """
    Génère un prompt Neo4j avec le schéma réel du graphe.
    
    Args:
        schema: Structure retournée par Neo4jSchemaInspector
    """
    # Décrire les nœuds
    nodes_desc = []
    for label, node_data in schema["nodes"].items():
        props = []
        for prop_name, prop_info in node_data["properties"].items():
            examples = prop_info.get("examples", [])
            examples_str = ", ".join(f'"{ex}"' if isinstance(ex, str) else str(ex) 
                                    for ex in examples[:2])
            props.append(f"  - {prop_name}: {prop_info['type']}  // Ex: {examples_str}")
        
        nodes_desc.append(f"""
Nœud "{label}" ({node_data["count"]} nœuds):
{chr(10).join(props) if props else "  (aucune propriété)"}""")
    
    # Décrire les relations
    rels_desc = []
    for rel_type, rel_data in schema["relationships"].items():
        rels_desc.append(
            f'  ({rel_data["from"]})-[:{rel_type}]->({rel_data["to"]})'
        )
    
    prompt = f"""Tu es un expert Neo4j/Cypher. Convertis la question en une requête Cypher valide.

SCHÉMA DU GRAPHE:

NŒUDS:
{chr(10).join(nodes_desc)}

RELATIONS:
{chr(10).join(rels_desc) if rels_desc else "  (aucune relation)"}

RÈGLES:
- Réponds UNIQUEMENT avec un objet JSON contenant la requête Cypher
- Format de réponse OBLIGATOIRE: {{"cypher": "ta requête ici"}}
- Dans la requête Cypher, utilise des alias dans le RETURN (ex: "RETURN m.name AS nom")
- Limite avec "LIMIT 10" sauf demande spécifique
- Pour les recherches textuelles, utilise WHERE avec comparaisons
- Pas de texte explicatif avant ou après le JSON
- Pas de markdown, pas de ```json ou ```cypher

EXEMPLES DE RÉPONSE:

Question: "Liste les catégories"
{{"cypher": "MATCH (c:Categorie) RETURN c.name AS categorie"}}

Question: "Médicaments moins de 5 euros"
{{"cypher": "MATCH (m:Medicament) WHERE m.price < 5 RETURN m.name AS medicament, m.price AS prix LIMIT 10"}}

Question: "Quels médicaments traitent la fièvre ?"
{{"cypher": "MATCH (m:Medicament)-[:TRAITE]->(ml:Maladie {{name: \\"fièvre\\"}}) RETURN m.name AS medicament, m.brand AS marque LIMIT 10"}}

Question suivante:"""
    
    return prompt
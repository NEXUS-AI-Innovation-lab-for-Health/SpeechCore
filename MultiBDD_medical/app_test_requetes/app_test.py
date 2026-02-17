import os
from pymongo import MongoClient
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# -----------------------------
# MongoDB
# -----------------------------
mongo_uri = "mongodb://admin:password@localhost:27018"
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client["pharma"]
medications = list(mongo_db.medications.find())
print("MongoDB - medications:")
for m in medications:
    print(f"  - {m.get('name', 'unknown')}")

# -----------------------------
# Neo4j
# -----------------------------
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_pass = "password"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

def get_medications_neo4j(tx):
    # Adapter aux labels et relations réels de ton seed
    query = """
    MATCH (m:Medicament)-[:TRAITE]->(d:Maladie)
    RETURN m.name AS med, d.name AS disease
    LIMIT 10
    """
    return list(tx.run(query))

with driver.session() as session:
    results = session.execute_read(get_medications_neo4j)
    print("\nNeo4j - Medicament → Maladie:")
    for r in results:
        print(f"  - {r['med']} → {r['disease']}")

# -----------------------------
# Qdrant
# -----------------------------
qdrant = QdrantClient(url="http://localhost:6333")

# Lister les collections
collections = qdrant.get_collections()
print("\nQdrant collections:")
for c in collections.collections:
    print(f"  - {c.name}")

# Lire quelques points dans la collection "medications"
try:
    points = qdrant.scroll(collection_name="medications", limit=5)
    print("\nQdrant - points in 'medications':")
    # points = qdrant.scroll(...) renvoie un tuple : (list_of_records, total_count)
    points_list, total_count = qdrant.scroll(collection_name="medications", limit=5)
    print(f"\nQdrant - points in 'medications' (total {total_count}):")
    for p in points_list:
        print(f"  - id: {p.id}, name: {p.payload.get('name')}, brand: {p.payload.get('brand')}, category: {p.payload.get('category')}")
        if 'indications' in p.payload:
            print(f"      indications: {p.payload['indications']}")

except Exception as e:
    print("Qdrant 'medications' not found or empty:", e)

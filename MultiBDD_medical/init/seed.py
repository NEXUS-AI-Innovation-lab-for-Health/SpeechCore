"""
seed.py — Initialisation des données factices (médicaments)
────────────────────────────────────────────────────────────
  MongoDB  → documents détaillés sur chaque médicament
  Qdrant   → vecteurs simulés pour recherche par similarité
  Neo4j    → graphe : Médicament ─TRAITE→ Maladie, Médicament ─APPARTIENT_A→ Catégorie
"""
import os
import time
import random

from pymongo   import MongoClient
from qdrant_client        import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j     import GraphDatabase


# ════════════════════════════════════════════════════════
#  DONNÉES FACTICES
# ════════════════════════════════════════════════════════

MEDICATIONS = [
    {
        "name":                 "Ibuprofen",
        "brand":                "Advil",
        "dosage":               "200mg",
        "category":             "Anti-inflammatoire",
        "indications":          ["douleur", "fièvre", "inflammation"],
        "side_effects":         ["maux d'estomac", "vertige"],
        "price":                4.50,
        "prescription_required": False,
    },
    {
        "name":                 "Paracétamol",
        "brand":                "Doliprane",
        "dosage":               "500mg",
        "category":             "Antalgique",
        "indications":          ["douleur", "fièvre"],
        "side_effects":         ["allergie rare"],
        "price":                2.80,
        "prescription_required": False,
    },
    {
        "name":                 "Amoxicilline",
        "brand":                "Amoxil",
        "dosage":               "500mg",
        "category":             "Antibiotique",
        "indications":          ["infection respiratoire", "infection urinaire", "infection dentaire"],
        "side_effects":         ["diarrhée", "allergie"],
        "price":                8.20,
        "prescription_required": True,
    },
    {
        "name":                 "Metformine",
        "brand":                "Glucophage",
        "dosage":               "850mg",
        "category":             "Antidiabétique",
        "indications":          ["diabète type 2"],
        "side_effects":         ["nausées", "diarrhée"],
        "price":                5.10,
        "prescription_required": True,
    },
    {
        "name":                 "Omeprazole",
        "brand":                "Losec",
        "dosage":               "20mg",
        "category":             "Inhibiteur pompe protons",
        "indications":          ["ulcère gastrique", "reflux gastro-œsophagien"],
        "side_effects":         ["mal de tête", "diarrhée"],
        "price":                6.00,
        "prescription_required": True,
    },
    {
        "name":                 "Atorvastatine",
        "brand":                "Lipitor",
        "dosage":               "20mg",
        "category":             "Statine",
        "indications":          ["hypercholestérolémie", "prévention cardiovasculaire"],
        "side_effects":         ["douleur musculaire", "fatigue"],
        "price":                12.50,
        "prescription_required": True,
    },
    {
        "name":                 "Lisinopril",
        "brand":                "Prinivil",
        "dosage":               "10mg",
        "category":             "IEC",
        "indications":          ["hypertension", "insuffisance cardiaque", "prévention cardiovasculaire"],
        "side_effects":         ["toux", "vertige"],
        "price":                7.30,
        "prescription_required": True,
    },
    {
        "name":                 "Cétirizine",
        "brand":                "Zyrtec",
        "dosage":               "10mg",
        "category":             "Antihistaminique",
        "indications":          ["rhinitis allergique", "urticaire"],
        "side_effects":         ["somnolence"],
        "price":                3.90,
        "prescription_required": False,
    },
]

# Dimension des vecteurs simulés (en réalité ce serait
# la dimension de ton modèle d'embedding, ex: 384 ou 768)
VECTOR_DIM = 8

# Graine fixed pour que les vecteurs soient reproductibles entre les runs
random.seed(42)


def _random_vector() -> list[float]:
    return [round(random.uniform(-1, 1), 4) for _ in range(VECTOR_DIM)]


# ════════════════════════════════════════════════════════
#  ATTENTE — les bases peuvent mettre du temps à démarrer
# ════════════════════════════════════════════════════════

def wait_for(name: str, check_fn, timeout: int = 90, interval: int = 3):
    """Boucle retry jusqu'à ce que check_fn() ne throw plus."""
    start = time.time()
    while True:
        try:
            check_fn()
            print(f"  ✓ {name} est prête")
            return
        except Exception:
            if time.time() - start > timeout:
                raise TimeoutError(f"⛔ {name} non accessible après {timeout}s")
            print(f"  ⏳ En attente de {name}...")
            time.sleep(interval)


# ════════════════════════════════════════════════════════
#  SEED MONGODB
# ════════════════════════════════════════════════════════

def seed_mongo():
    uri    = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017")
    client = MongoClient(uri, authSource="admin")
    db     = client["pharma"]
    col    = db["medications"]

    col.delete_many({})              # on clean avant d'insérer
    col.insert_many(MEDICATIONS)

    print(f"  [MongoDB] {len(MEDICATIONS)} médicaments → 'pharma.medications'")
    client.close()


# ════════════════════════════════════════════════════════
#  SEED QDRANT
# ════════════════════════════════════════════════════════

def seed_qdrant():
    url    = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)

    # Supprimer la collection si elle existe déjà
    existing = [c.name for c in client.get_collections().collections]
    if "medications" in existing:
        client.delete_collection("medications")

    # Créer avec la bonne dimension
    client.create_collection(
        collection_name="medications",
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )

    # Insérer les points (id, vecteur, payload)
    points = [
        PointStruct(
            id=i + 1,                          # id entier positif
            vector=_random_vector(),
            payload={
                "name":       med["name"],
                "brand":      med["brand"],
                "category":   med["category"],
                "indications": med["indications"],
            }
        )
        for i, med in enumerate(MEDICATIONS)
    ]
    client.upsert(collection_name="medications", points=points)

    print(f"  [Qdrant] {len(points)} points → collection 'medications' (dim={VECTOR_DIM})")
    client.close()


# ════════════════════════════════════════════════════════
#  SEED NEO4J
# ════════════════════════════════════════════════════════

# Carte : maladie → liste de médicaments qui la traitent
TREATMENT_MAP = {
    "douleur":                      ["Ibuprofen", "Paracétamol"],
    "fièvre":                       ["Ibuprofen", "Paracétamol"],
    "inflammation":                 ["Ibuprofen"],
    "infection respiratoire":       ["Amoxicilline"],
    "infection urinaire":           ["Amoxicilline"],
    "infection dentaire":           ["Amoxicilline"],
    "diabète type 2":               ["Metformine"],
    "ulcère gastrique":             ["Omeprazole"],
    "reflux gastro-œsophagien":    ["Omeprazole"],
    "hypercholestérolémie":         ["Atorvastatine"],
    "prévention cardiovasculaire":  ["Atorvastatine", "Lisinopril"],
    "hypertension":                 ["Lisinopril"],
    "insuffisance cardiaque":       ["Lisinopril"],
    "rhinitis allergique":          ["Cétirizine"],
    "urticaire":                    ["Cétirizine"],
}


def seed_neo4j():
    uri      = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASS", "password")
    driver   = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as s:
        # ── Clean total ─────────────────────────────
        s.run("MATCH (n) DETACH DELETE n")

        # ── Créer les nœuds Medicament ───────────────
        for med in MEDICATIONS:
            s.run(
                """
                CREATE (:Medicament {
                    name:     $name,
                    brand:    $brand,
                    dosage:   $dosage,
                    category: $category,
                    price:    $price
                })
                """,
                name=med["name"],
                brand=med["brand"],
                dosage=med["dosage"],
                category=med["category"],
                price=med["price"],
            )

        # ── Créer les nœuds Maladie + relation TRAITE ──
        for maladie, med_names in TREATMENT_MAP.items():
            # Créer ou retrouver le nœud Maladie
            s.run("MERGE (:Maladie {name: $name})", name=maladie)

            # Relier chaque médicament concerné
            for med_name in med_names:
                s.run(
                    """
                    MATCH (m  :Medicament {name: $med_name})
                    MATCH (ml :Maladie    {name: $maladie})
                    CREATE (m)-[:TRAITE]->(ml)
                    """,
                    med_name=med_name,
                    maladie=maladie,
                )

        # ── Créer les nœuds Categorie + relation APPARTIENT_A ──
        for med in MEDICATIONS:
            s.run(
                """
                MATCH  (m :Medicament {name: $name})
                MERGE  (c :Categorie  {name: $category})
                CREATE (m)-[:APPARTIENT_A]->(c)
                """,
                name=med["name"],
                category=med["category"],
            )

    print(
        f"  [Neo4j]  Graphe créé → "
        f"{len(MEDICATIONS)} médicaments, "
        f"{len(TREATMENT_MAP)} maladies, "
        f"{len(set(m['category'] for m in MEDICATIONS))} catégories"
    )
    driver.close()


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Démarrage de l'initialisation...\n")

    # ── 1) Attendre que les trois bases soient up ───────
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017")
    wait_for("MongoDB", lambda: MongoClient(mongo_uri, serverSelectionTimeoutMS=2000).server_info())

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    wait_for("Qdrant",  lambda: QdrantClient(url=qdrant_url).get_collections())

    neo4j_uri  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS", "password")
    def _check_neo4j():
        d = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        d.verify_connectivity()
        d.close()
    wait_for("Neo4j",   _check_neo4j)

    # ── 2) Insérer les données ───────────────────────────
    print()
    seed_mongo()
    seed_qdrant()
    seed_neo4j()

    print("\n✅ Données initialisées dans les trois bases !")
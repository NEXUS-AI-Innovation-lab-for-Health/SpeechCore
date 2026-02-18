#!/usr/bin/env python3
"""
Serveur FastAPI unifié pour SpeechCore
Combine :
- Transcription audio avec Whisper
- Extraction automatique de données avec Ollama/Mistral
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import tempfile
import os
import whisper
import httpx
import json
import re

# ══════════════════════════════════════════════════════════════
# MODÈLES DE DONNÉES
# ══════════════════════════════════════════════════════════════

class FormField(BaseModel):
    """Définition d'un champ de formulaire"""
    name: str                           # Nom technique (ex: "patient_name")
    label: str                          # Label affiché (ex: "Nom du patient")
    type: str                           # Type de champ (text, number, date, etc.)
    required: bool = False              # Obligatoire ou non
    semantic_hint: Optional[str] = None # Indice sémantique pour l'IA

class FormSchema(BaseModel):
    """Structure complète d'un formulaire"""
    fields: List[FormField]

class ExtractionRequest(BaseModel):
    """Requête pour extraire des données d'un texte"""
    form: FormSchema  # Structure du formulaire
    text: str         # Texte transcrit à analyser

class ExtractionResponse(BaseModel):
    """Résultat de l'extraction"""
    data: Dict[str, Optional[str]]  # Valeurs extraites pour chaque champ
    success: bool = True

# ══════════════════════════════════════════════════════════════
# CLIENT OLLAMA
# ══════════════════════════════════════════════════════════════

class LLMClient:
    """
    Client pour communiquer avec Ollama (serveur local d'IA)
    Ollama doit être lancé avec : ollama run mistral
    """
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    async def generate(self, prompt: str) -> str:
        """
        Envoie un prompt à Ollama et récupère la réponse
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # Pas de streaming, on veut tout d'un coup
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                return response.json()["response"].strip()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur Ollama : {str(e)}. Vérifiez que 'ollama run mistral' est lancé."
            )

# ══════════════════════════════════════════════════════════════
# EXTRACTEUR DE FORMULAIRE
# ══════════════════════════════════════════════════════════════

class FormExtractor:
    """
    Utilise l'IA (Mistral via Ollama) pour extraire automatiquement
    des informations d'un texte et remplir un formulaire
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def build_prompt(self, form: FormSchema, text: str) -> str:
        """
        Construit le prompt qui sera envoyé à l'IA
        
        Le prompt explique à l'IA :
        - Quels champs chercher
        - Le texte à analyser
        - Le format de réponse attendu (JSON)
        """
        fields_description = []
        for field in form.fields:
            hint = field.semantic_hint or field.label
            fields_description.append(f'- "{field.name}": {hint}')
        
        fields_str = '\n'.join(fields_description)
        
        prompt = f"""Tu es un assistant médical spécialisé dans l'extraction d'informations.

Voici un texte d'une consultation médicale :
"{text}"

Extrait les informations suivantes et RÉPONDS UNIQUEMENT avec un objet JSON valide :
{fields_str}

Si une information n'est pas présente dans le texte, utilise null.

Format de réponse (JSON uniquement, sans autre texte) :
{{
  "field_name": "valeur extraite ou null"
}}

Exemple :
Texte : "Le patient s'appelle Jean Dupont, il a 45 ans"
Réponse : {{"nom": "Dupont", "prenom": "Jean", "age": "45"}}

IMPORTANT : Réponds UNIQUEMENT avec le JSON, sans explication."""

        return prompt

    def parse_llm_output(self, raw_output: str) -> dict:
        """
        Parse la réponse de l'IA pour extraire le JSON
        
        Parfois l'IA rajoute du texte avant/après le JSON,
        donc on cherche le JSON avec une regex
        """
        if not raw_output:
            return {}

        # Chercher le JSON dans la réponse
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            return {}

        json_str = match.group(0)

        # Parser le JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {}

        # Normaliser : tout en string ou None
        normalized = {}
        for key, value in data.items():
            if value is None or str(value).lower() in ["null", "none", "n/a"]:
                normalized[key] = None
            else:
                normalized[key] = str(value)

        return normalized

    async def extract(self, form: FormSchema, text: str) -> dict:
        """
        Fonction principale d'extraction
        
        1. Construit le prompt
        2. Envoie à l'IA
        3. Parse la réponse
        4. Garantit que tous les champs sont présents
        """
        # Construire le prompt
        prompt = self.build_prompt(form, text)
        
        print(f"📤 Envoi à Ollama...")
        
        # Appeler l'IA
        raw_output = await self.llm.generate(prompt)
        
        print(f"📥 Réponse brute : {raw_output[:200]}...")
        
        # Parser la réponse
        data = self.parse_llm_output(raw_output)
        
        # Garantir que TOUS les champs du formulaire existent dans la réponse
        result = {}
        for field in form.fields:
            result[field.name] = data.get(field.name)

        return result

# ══════════════════════════════════════════════════════════════
# INITIALISATION FASTAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="API SpeechCore",
    description="API pour transcription audio (Whisper) et extraction de données (Ollama)",
    version="2.0.0"
)

# CORS : permettre les requêtes depuis le navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger Whisper au démarrage
print("🔄 Chargement du modèle Whisper (base)...")
whisper_model = whisper.load_model("base")
print("✅ Whisper chargé !")

# Initialiser le système d'extraction
llm_client = LLMClient()
extractor = FormExtractor(llm_client)

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API SpeechCore - Transcription + Extraction",
        "status": "ok",
        "endpoints": {
            "/health": "Vérifier que le serveur fonctionne",
            "/transcribe": "Transcrire un fichier audio avec Whisper",
            "/extract": "Extraire des données d'un texte avec Ollama"
        }
    }

@app.get("/health")
async def health():
    """Route de santé"""
    return {
        "status": "ok",
        "whisper": "loaded",
        "message": "Serveur opérationnel"
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcrit un fichier audio avec Whisper
    
    Identique à la version simple
    """
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"📂 Fichier reçu : {file.filename}")
        
        # Transcrire
        result = whisper_model.transcribe(tmp_path, language='fr', fp16=False)
        text = result["text"].strip()
        
        print(f"✅ Transcription : '{text[:50]}...'")
        
        # Nettoyer
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return {
            "text": text,
            "language": result.get("language", "fr"),
            "success": True
        }
    
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

@app.post("/extract", response_model=ExtractionResponse)
async def extract_form(req: ExtractionRequest):
    """
    Extrait automatiquement des données d'un texte pour remplir un formulaire
    
    Exemple de requête :
    {
      "form": {
        "fields": [
          {"name": "nom", "label": "Nom du patient", "type": "text"},
          {"name": "age", "label": "Âge", "type": "number"}
        ]
      },
      "text": "Le patient s'appelle Jean Dupont et il a 45 ans"
    }
    
    Réponse :
    {
      "data": {
        "nom": "Dupont",
        "age": "45"
      },
      "success": true
    }
    """
    try:
        print(f"📝 Extraction demandée pour {len(req.form.fields)} champs")
        print(f"📄 Texte : {req.text[:100]}...")
        
        # Appeler l'extracteur
        data = await extractor.extract(req.form, req.text)
        
        print(f"✅ Extraction terminée : {data}")
        
        return {"data": data, "success": True}
    
    except Exception as e:
        print(f"❌ Erreur extraction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   SERVEUR SPEECHCORE COMPLET                    ║
    ║                                                   ║
    ║   http://localhost:8000                          ║
    ║                                                   ║
    ║   Routes :                                        ║
    ║   - GET  /health                                  ║
    ║   - POST /transcribe (Whisper)                    ║
    ║   - POST /extract (Ollama/Mistral)                ║
    ║                                                   ║
    ║   ⚠️  Avant de lancer, installer Ollama :        ║
    ║   https://ollama.com/download                     ║
    ║   puis : ollama pull mistral                      ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
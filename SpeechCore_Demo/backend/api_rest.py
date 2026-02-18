#!/usr/bin/env python3
"""
API REST FastAPI pour transcription audio
+ Extraction avec Ollama
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import tempfile
import shutil
import httpx
import json
import re

from transcription_engines import transcrire_vosk, transcrire_whisper, transcrire_gladia
from utils import generer_json

# ══════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC POUR EXTRACTION
# ══════════════════════════════════════════════════════════════

class FormField(BaseModel):
    """Définition d'un champ de formulaire"""
    name: str
    label: str
    type: str
    required: bool = False
    semantic_hint: Optional[str] = None

class FormSchema(BaseModel):
    """Structure complète d'un formulaire"""
    fields: List[FormField]

class ExtractionRequest(BaseModel):
    """Requête pour extraire des données d'un texte"""
    form: FormSchema
    text: str

class ExtractionResponse(BaseModel):
    """Résultat de l'extraction"""
    data: Dict[str, Optional[str]]
    success: bool = True

# ══════════════════════════════════════════════════════════════
# CLIENT OLLAMA
# ══════════════════════════════════════════════════════════════

class LLMClient:
    """Client pour communiquer avec Ollama (serveur local d'IA)"""
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    async def generate(self, prompt: str) -> str:
        """Envoie un prompt à Ollama et récupère la réponse"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                return response.json()["response"].strip()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur Ollama : {str(e)}. Vérifiez que 'ollama serve' est lancé."
            )

# ══════════════════════════════════════════════════════════════
# EXTRACTEUR DE FORMULAIRE
# ══════════════════════════════════════════════════════════════

class FormExtractor:
    """Utilise l'IA (Mistral via Ollama) pour extraire des informations"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def build_prompt(self, form: FormSchema, text: str) -> str:
        """Construit le prompt pour l'IA"""
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
        """Parse la réponse de l'IA pour extraire le JSON"""
        if not raw_output:
            return {}

        # Chercher le JSON dans la réponse
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            return {}

        json_str = match.group(0)

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
        """Fonction principale d'extraction"""
        prompt = self.build_prompt(form, text)
        
        print(f"📤 Envoi à Ollama...")
        
        raw_output = await self.llm.generate(prompt)
        
        print(f"📥 Réponse brute : {raw_output[:200]}...")
        
        data = self.parse_llm_output(raw_output)
        
        # Garantir que TOUS les champs existent
        result = {}
        for field in form.fields:
            result[field.name] = data.get(field.name)

        return result

# ══════════════════════════════════════════════════════════════
# INITIALISATION FASTAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="API de Transcription Audio + Extraction",
    description="Vosk, Whisper, Gladia + Ollama",
    version="3.0.0"
)

# CORS : permettre les requêtes depuis React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser Ollama pour l'extraction
llm_client = LLMClient()
extractor = FormExtractor(llm_client)

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "API de Transcription Audio v3",
        "endpoints": {
            "/vosk": "Transcription Vosk (local, rapide)",
            "/whisper": "Transcription Whisper (local, GPU)",
            "/gladia": "Transcription Gladia (cloud, 10h/mois)",
            "/extract": "Extraction de données avec Ollama"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health():
    """Route de santé"""
    return {
        "status": "ok",
        "message": "Serveur opérationnel",
        "endpoints": ["vosk", "whisper", "gladia", "extract"]
    }


@app.post("/vosk")
async def transcription_vosk(
    file: UploadFile = File(...),
    modele: str = "grand",
    nb_locuteurs: int = 2,
    reduction_bruit: bool = True,
    type_environnement: str = "2",
    methode_bruit: str = "noisereduce"
):
    """
    Transcription avec Vosk
    
    - **modele**: petit ou grand
    - **nb_locuteurs**: 1-10
    - **reduction_bruit**: true/false
    - **type_environnement**: 1=Silencieux, 2=Normal, 3=Bruyant, 4=Constant
    - **methode_bruit**: noisereduce ou silero
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Fichier WAV requis")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        resultats = transcrire_vosk(
            tmp_path,
            modele=modele,
            nb_locuteurs=nb_locuteurs,
            reduction_bruit=reduction_bruit,
            type_environnement=type_environnement,
            methode_bruit=methode_bruit
        )
        
        json_response = generer_json(file.filename, resultats, "Vosk")
        return JSONResponse(content=json_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/whisper")
async def transcription_whisper(
    file: UploadFile = File(...),
    config: str = "cpu_rapide",
    nb_locuteurs: int = 2,
    reduction_bruit: bool = True,
    type_environnement: str = "2",
    methode_bruit: str = "noisereduce"
):
    """
    Transcription avec Whisper
    
    - **config**: cpu_rapide, cpu_qualite, gpu_equilibre, gpu_max, ultra_rapide
    - **nb_locuteurs**: 1-10
    - **reduction_bruit**: true/false
    - **type_environnement**: 1=Silencieux, 2=Normal, 3=Bruyant, 4=Constant
    - **methode_bruit**: noisereduce ou silero
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Fichier WAV requis")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        resultats = transcrire_whisper(
            tmp_path,
            config=config,
            nb_locuteurs=nb_locuteurs,
            reduction_bruit=reduction_bruit,
            type_environnement=type_environnement,
            methode_bruit=methode_bruit
        )
        
        json_response = generer_json(file.filename, resultats, "Whisper")
        return JSONResponse(content=json_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/gladia")
async def transcription_gladia(
    file: UploadFile = File(...),
    nb_locuteurs: int = 0
):
    """
    Transcription avec Gladia API
    
    - **nb_locuteurs**: 0 pour auto, 1-10 pour fixe
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Fichier WAV requis")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        resultats = transcrire_gladia(tmp_path, nb_locuteurs=nb_locuteurs)
        json_response = generer_json(file.filename, resultats, "Gladia")
        return JSONResponse(content=json_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/extract", response_model=ExtractionResponse)
async def extract_form(req: ExtractionRequest):
    """
    Extrait automatiquement des données d'un texte pour remplir un formulaire
    
    Utilise Ollama (Mistral) pour analyser le texte et extraire les valeurs.
    
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
    ╔═══════════════════════════════════════════════════════╗
    ║   API SPEECHCORE COMPLÈTE                           ║
    ║                                                       ║
    ║   http://localhost:8000                              ║
    ║                                                       ║
    ║   ROUTES DISPONIBLES :                                ║
    ║   • POST /vosk      - Transcription Vosk             ║
    ║   • POST /whisper   - Transcription Whisper          ║
    ║   • POST /gladia    - Transcription Gladia           ║
    ║   • POST /extract   - Extraction Ollama              ║
    ║                                                       ║
    ║   📚 Documentation : http://localhost:8000/docs      ║
    ║                                                       ║
    ║   ⚠️  PRÉREQUIS :                                     ║
    ║   • Ollama : ollama serve                            ║
    ║   • Modèle : ollama pull mistral                     ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
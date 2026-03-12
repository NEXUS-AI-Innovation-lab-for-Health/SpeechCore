# 🐳 Docker — SpeechCore

## 📁 Fichiers

```
├── Dockerfile.api_rest        # Image API REST
├── Dockerfile.api_websocket   # Image API WebSocket
├── entrypoint_rest.sh         # Script de démarrage REST
├── entrypoint_websocket.sh    # Script de démarrage WebSocket
└── docker-compose.yml         # Orchestration des deux services
```

## 🚀 Démarrage

```bash
# Premier lancement (build + téléchargement des modèles Vosk ~1.5 GB)
docker compose up --build

# En arrière-plan
docker compose up --build -d

# Arrêter
docker compose down
```

## 🌐 Services

| Service   | URL                              | Description          |
|-----------|----------------------------------|----------------------|
| API REST  | http://localhost:8000            | Endpoints REST       |
| Docs      | http://localhost:8000/docs       | Swagger UI           |
| WebSocket | ws://localhost:8001/ws/transcribe| Transcription live   |
| Web UI    | http://localhost:8001            | Interface de test    |

## 📦 Volumes

Les modèles et caches sont persistés dans des volumes Docker partagés entre les deux conteneurs :

| Volume            | Contenu                        | Taille   |
|-------------------|--------------------------------|----------|
| `vosk_models`     | vosk-model-small-fr-0.22       | ~41 MB   |
| `vosk_models_grand` | vosk-model-fr-0.22           | ~1.5 GB  |
| `whisper_cache`   | Cache modèles Whisper          | variable |

> ⚠️ Au **premier démarrage**, les deux conteneurs téléchargent les modèles en parallèle.
> Pour éviter un double téléchargement, lance d'abord le REST seul, puis les deux :
> ```bash
> docker compose up api_rest      # attendre la fin du téléchargement
> # Ctrl+C
> docker compose up -d            # relancer les deux
> ```

## 🔧 Commandes utiles

```bash
# Voir les logs en temps réel
docker compose logs -f

# Logs d'un seul service
docker compose logs -f api_rest
docker compose logs -f api_websocket

# Rebuilder un seul service
docker compose build api_rest

# Redémarrer un service
docker compose restart api_rest

# Supprimer les conteneurs ET les volumes (repart de zéro)
docker compose down -v
```

## Lancer les deux services
docker compose up --build

# #Ou en arrière-plan
docker compose up --build -d

## Un seul service
docker compose up api_rest
docker compose up api_websocket

---

# 🎙️ Système de Transcription Audio Modulaire

Architecture modulaire pour la transcription audio avec 3 moteurs (Vosk, Whisper, Gladia) et 3 interfaces (CLI, REST, WebSocket).

## 📁 Structure du projet

```
├── audio_processing.py      # 🎚️  Traitement audio (analyse, réduction bruit)
├── transcription_engines.py # 🤖 Moteurs de transcription (Vosk, Whisper, Gladia)
├── utils.py                 # 🛠️  Utilitaires (JSON, fichiers)
├── cli.py                   # 💻 Interface console interactive
├── api_rest.py              # 📡 API REST FastAPI
└── api_websocket.py         # 🌐 API WebSocket temps réel
```

## 🚀 Installation

```bash
# Dépendances de base
pip3 install vosk faster-whisper resemblyzer noisereduce soundfile scikit-learn requests

# Pour Silero VAD (optionnel)
pip3 install torch torchaudio scipy

# Pour les APIs
pip3 install fastapi uvicorn python-multipart

# Modèles Vosk
wget https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip
unzip vosk-model-fr-0.22.zip
```

## 💻 Utilisation CLI

```bash
python3 cli.py
```

Interface interactive qui guide l'utilisateur :
1. Choix du moteur (Vosk, Whisper, Gladia)
2. Configuration spécifique
3. Sélection du fichier
4. Options de traitement
5. Type de sortie

## 📡 API REST

### Démarrage

```bash
python3 api_rest.py
```

Accès : http://localhost:8000
Documentation : http://localhost:8000/docs

### Endpoints

#### POST /vosk
```bash
curl -X POST "http://localhost:8000/vosk" \
  -F "file=@audio.wav" \
  -F "modele=grand" \
  -F "nb_locuteurs=2" \
  -F "reduction_bruit=true" \
  -F "type_environnement=2" \
  -F "methode_bruit=noisereduce"
```

#### POST /whisper
```bash
curl -X POST "http://localhost:8000/whisper" \
  -F "file=@audio.wav" \
  -F "config=cpu_rapide" \
  -F "nb_locuteurs=2" \
  -F "reduction_bruit=true" \
  -F "methode_bruit=silero"
```

#### POST /gladia
```bash
curl -X POST "http://localhost:8000/gladia" \
  -F "file=@audio.wav" \
  -F "nb_locuteurs=0"
```

### Réponse JSON

```json
{
  "fichier_source": "audio.wav",
  "date_traitement": "2026-02-04 12:00:00",
  "moteur": "Vosk",
  "analyse_audio": {
    "duree": 50.0,
    "sample_rate": 16000,
    "canaux": 1,
    "niveau_db": -32.0,
    "activite_vocale": 33.8
  },
  "statistiques": {
    "nombre_mots": 245,
    "nombre_segments": 18,
    "nombre_locuteurs": 2,
    "langue_detectee": "fr",
    "confiance_langue": 1.0
  },
  "locution_separee": [
    {"locuteur": "Locuteur 0", "texte": "Bonjour..."},
    {"locuteur": "Locuteur 1", "texte": "Merci..."}
  ],
  "transcription_complete": "Bonjour et bienvenue...",
  "transcription_avec_locuteurs": "[Locuteur 0] Bonjour..."
}
```

## 🌐 API WebSocket

### Démarrage

```bash
python3 api_websocket.py
```

Accès : http://localhost:8000 (interface web intégrée)

### Exemple Python Client

```python
import asyncio
import websockets
import json
import base64

async def transcrire():
    with open('audio.wav', 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    config = {
        "engine": "vosk",
        "modele_vosk": "grand",
        "nb_locuteurs": 2,
        "methode_bruit": "silero",
        "type_environnement": "2",
        "audio": audio_base64
    }
    
    async with websockets.connect('ws://localhost:8000/ws/transcribe') as ws:
        await ws.send(json.dumps(config))
        
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if data['type'] == 'status':
                print(f"Status: {data['message']}")
            elif data['type'] == 'result':
                print(f"Transcription: {data['transcription_complete']}")
                break
            elif data['type'] == 'error':
                print(f"Erreur: {data['message']}")
                break

asyncio.run(transcrire())
```

## 🎚️ Module: audio_processing.py

Fonctions disponibles :

```python
from audio_processing import analyser_audio, reduire_bruit

# Analyse audio
stats = analyser_audio(Path("audio.wav"))
# {'duree': 50.0, 'sample_rate': 16000, 'canaux': 1, 'niveau_db': -32.0, 'activite_vocale': 33.8}

# Réduction de bruit
fichier_clean = reduire_bruit(
    Path("audio.wav"),
    type_environnement="2",  # 1-4
    methode="silero"  # "noisereduce" ou "silero"
)
```

## 🤖 Module: transcription_engines.py

Fonctions disponibles :

```python
from transcription_engines import transcrire_vosk, transcrire_whisper, transcrire_gladia

# Vosk
resultats = transcrire_vosk(
    Path("audio.wav"),
    modele="grand",
    nb_locuteurs=2,
    reduction_bruit=True,
    type_environnement="2",
    methode_bruit="silero"
)

# Whisper
resultats = transcrire_whisper(
    Path("audio.wav"),
    config="cpu_rapide",
    nb_locuteurs=2,
    reduction_bruit=True,
    methode_bruit="noisereduce"
)

# Gladia
resultats = transcrire_gladia(
    Path("audio.wav"),
    nb_locuteurs=0  # 0 = auto
)
```

## 🛠️ Module: utils.py

Fonctions disponibles :

```python
from utils import generer_json, sauvegarder_fichier_texte, sauvegarder_json

# Générer JSON
json_data = generer_json("audio.wav", resultats, "Vosk")

# Sauvegarder fichier texte
fichier_txt = sauvegarder_fichier_texte(Path("audio.wav"), resultats, "Vosk")

# Sauvegarder JSON
fichier_json = sauvegarder_json(Path("audio.wav"), json_data)
```

## 📊 Comparaison des moteurs

| Moteur | Vitesse | Qualité | Gratuit | Local | GPU |
|--------|---------|---------|---------|-------|-----|
| **Vosk** | ⚡⚡⚡ | ⭐⭐⭐ | ✅ Illimité | ✅ | ❌ |
| **Whisper** | ⚡⚡ | ⭐⭐⭐⭐ | ✅ Illimité | ✅ | ✅ |
| **Gladia** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ 10h/mois | ❌ | N/A |

## 🔧 Options de réduction de bruit

### NoiseReduce (classique)
- **Type environnement** :
  - 1 : Salle silencieuse
  - 2 : Bureau/Normal
  - 3 : Environnement bruyant
  - 4 : Bruit constant

### Silero VAD (IA)
- Suppression intelligente des non-paroles
- Garde uniquement les segments de voix
- Nécessite PyTorch

## 📝 Exemples d'utilisation

### Script Python simple

```python
from pathlib import Path
from transcription_engines import transcrire_vosk
from utils import sauvegarder_fichier_texte

# Transcrire
resultats = transcrire_vosk(
    Path("audio.wav"),
    modele="grand",
    nb_locuteurs=2,
    reduction_bruit=True,
    methode_bruit="silero"
)

# Sauvegarder
fichier = sauvegarder_fichier_texte(Path("audio.wav"), resultats, "Vosk")
print(f"Sauvegardé: {fichier}")

# Ou utiliser directement
print(resultats['texte_brut'])
print(resultats['texte_diarise'])
```

### Intégration dans une app

```python
from transcription_engines import transcrire_whisper

def mon_app_transcrire(fichier_path):
    try:
        resultats = transcrire_whisper(
            fichier_path,
            config="cpu_rapide",
            nb_locuteurs=2
        )
        
        return {
            "success": True,
            "transcription": resultats['texte_brut'],
            "locuteurs": resultats['texte_diarise']
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

## 🎯 Cas d'usage

- **CLI** : Tests rapides, usage ponctuel
- **API REST** : Intégration serveur-serveur
- **WebSocket** : Applications web temps réel
- **Modules** : Intégration dans vos propres scripts

## 🔒 Production

Pour la production, ajoutez :
- Authentification (API Key, OAuth)
- Rate limiting
- Validation des fichiers (taille, format)
- HTTPS
- Logging
- Error handling robuste

## 📄 Licence

Open source - Utilisation libre
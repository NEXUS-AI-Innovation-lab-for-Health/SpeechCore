# 🎙️ API de Transcription Audio

API FastAPI pour transcrire des fichiers audio avec 3 moteurs différents : Vosk, Whisper et Gladia.

## 🚀 Installation

```bash
# Installer FastAPI et Uvicorn
pip install fastapi uvicorn python-multipart --break-system-packages

# Installer les dépendances de transcription (déjà installées normalement)
pip install vosk faster-whisper resemblyzer noisereduce soundfile scikit-learn requests --break-system-packages
```

## ▶️ Démarrer l'API

```bash
python3 api_transcription.py
```

L'API sera accessible sur : **http://localhost:8000**

Documentation interactive : **http://localhost:8000/docs**

## 📡 Endpoints disponibles

### 1. **POST /vosk** - Transcription avec Vosk (local, gratuit illimité)

```bash
curl -X POST "http://localhost:8000/vosk" \
  -F "file=@audio.wav" \
  -F "nb_locuteurs=2" \
  -F "reduction_bruit=true" \
  -F "type_environnement=2" \
  -F "modele_vosk=grand"
```

**Paramètres :**
- `file` : Fichier audio WAV (requis)
- `nb_locuteurs` : Nombre de locuteurs (défaut: 2, range: 1-10)
- `reduction_bruit` : Activer réduction bruit (défaut: true)
- `type_environnement` : Type d'environnement (défaut: "2")
  - "1" : Salle silencieuse
  - "2" : Bureau/Normal
  - "3" : Environnement bruyant
  - "4" : Bruit constant
- `modele_vosk` : Taille du modèle (défaut: "grand")
  - "petit" : Rapide (41 MB)
  - "grand" : Meilleure qualité (1.5 GB)

---

### 2. **POST /whisper** - Transcription avec Faster-Whisper (local, GPU supporté)

```bash
curl -X POST "http://localhost:8000/whisper" \
  -F "file=@audio.wav" \
  -F "nb_locuteurs=2" \
  -F "reduction_bruit=true" \
  -F "type_environnement=2" \
  -F "config_whisper=cpu_rapide"
```

**Paramètres :**
- `file` : Fichier audio WAV (requis)
- `nb_locuteurs` : Nombre de locuteurs (défaut: 2, range: 1-10)
- `reduction_bruit` : Activer réduction bruit (défaut: true)
- `type_environnement` : Type d'environnement (défaut: "2")
- `config_whisper` : Configuration Whisper (défaut: "cpu_rapide")
  - "cpu_rapide" : CPU i5/i7 sans GPU
  - "cpu_qualite" : CPU puissant (i7/i9)
  - "gpu_equilibre" : GPU NVIDIA (RTX 2060/3060)
  - "gpu_max" : GPU puissant (RTX 3080/4080)
  - "ultra_rapide" : Très rapide, qualité basique

---

### 3. **POST /gladia** - Transcription avec Gladia API (10h/mois gratuit)

```bash
curl -X POST "http://localhost:8000/gladia" \
  -F "file=@audio.wav" \
  -F "nb_locuteurs=0"
```

**Paramètres :**
- `file` : Fichier audio WAV (requis)
- `nb_locuteurs` : Nombre de locuteurs (défaut: 0)
  - 0 : Détection automatique
  - 1-10 : Nombre fixe

**Note :** Gladia gère automatiquement la réduction de bruit et l'optimisation.

---

## 📊 Format de réponse JSON

Tous les endpoints retournent le même format JSON :

```json
{
  "fichier_source": "audio.wav",
  "date_traitement": "2026-01-31 16:30:00",
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
    {
      "locuteur": "Locuteur 0",
      "texte": "Bonjour et bienvenue dans cette émission..."
    },
    {
      "locuteur": "Locuteur 1",
      "texte": "Merci de m'accueillir. C'est un sujet passionnant..."
    }
  ],
  "transcription_complete": "Bonjour et bienvenue dans cette émission. Merci de m'accueillir...",
  "transcription_avec_locuteurs": "[Locuteur 0] Bonjour et bienvenue...\n\n[Locuteur 1] Merci de m'accueillir..."
}
```

## 🐍 Exemple d'utilisation Python

```python
import requests

# Préparer le fichier
files = {'file': open('audio.wav', 'rb')}

# Configuration
data = {
    'nb_locuteurs': 2,
    'reduction_bruit': True,
    'type_environnement': '2',
    'modele_vosk': 'grand'
}

# Appeler l'API Vosk
response = requests.post(
    'http://localhost:8000/vosk',
    files=files,
    data=data
)

# Récupérer le JSON
result = response.json()

print(f"Transcription: {result['transcription_complete']}")
print(f"Nombre de locuteurs: {result['statistiques']['nombre_locuteurs']}")

# Afficher chaque intervention
for intervention in result['locution_separee']:
    print(f"{intervention['locuteur']}: {intervention['texte'][:50]}...")
```

## 🌐 Exemple avec JavaScript (Fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('nb_locuteurs', 2);
formData.append('reduction_bruit', true);
formData.append('type_environnement', '2');
formData.append('config_whisper', 'cpu_rapide');

fetch('http://localhost:8000/whisper', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Transcription:', data.transcription_complete);
  console.log('Locuteurs:', data.locution_separee);
});
```

## 📝 Documentation interactive

Une fois l'API démarrée, accédez à :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

Vous pourrez tester directement les endpoints depuis votre navigateur !

## ⚙️ Configuration

### Modifier le port

```python
# Dans api_transcription.py, dernière ligne :
uvicorn.run(app, host="0.0.0.0", port=8080)  # Changer 8000 en 8080
```

### Modifier la clé Gladia

```python
# Dans api_transcription.py, ligne 29 :
GLADIA_API_KEY = "votre_nouvelle_clé"
```

## 🔒 Sécurité

**⚠️ Important :**
- L'API accepte n'importe quel fichier WAV (vérifiez la taille max)
- Aucune authentification par défaut
- Pour la production, ajoutez :
  - Authentification (API Key, OAuth)
  - Limite de taille de fichier
  - Rate limiting
  - HTTPS

## 🛠️ Dépannage

### Erreur "Modèle Vosk non installé"
```bash
# Télécharger le modèle manquant
wget https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip
unzip vosk-model-fr-0.22.zip
```

### Erreur GPU Whisper
Si vous n'avez pas de GPU NVIDIA, utilisez :
- `config_whisper=cpu_rapide`
- `config_whisper=cpu_qualite`

### Timeout Gladia
Pour les fichiers longs (>5min), le timeout est fixé à 4 minutes. Augmentez `max_tentatives` dans la fonction `transcrire_gladia()`.

## 📊 Comparaison des moteurs

| Moteur | Vitesse | Qualité | Gratuit | GPU | Installation |
|--------|---------|---------|---------|-----|--------------|
| **Vosk** | ⚡⚡⚡ | ⭐⭐⭐ | ✅ Illimité | ❌ | Modèles locaux |
| **Whisper** | ⚡⚡ | ⭐⭐⭐⭐ | ✅ Illimité | ✅ | Modèles locaux |
| **Gladia** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ 10h/mois | N/A | API en ligne |

## 📄 Licence

Open source - Utilisation libre
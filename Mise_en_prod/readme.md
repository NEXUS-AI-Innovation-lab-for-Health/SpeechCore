# 🐳 SpeechCore — Guide Docker

Ce guide explique comment builder, publier et utiliser les images Docker du projet SpeechCore.

---

## 📦 Images disponibles

| Image | Port | Description |
|---|---|---|
| `ghcr.io/speechcore-sae/api-rest` | 8000 | API REST Transcription |
| `ghcr.io/speechcore-sae/api-websocket` | 8001 | API WebSocket Transcription |
| `ghcr.io/speechcore-sae/api-generation` | 8002 | API Génération de requêtes |
| `ghcr.io/speechcore-sae/api-completion` | 8003 | API Complétion de formulaire |

---

## 🚀 Utiliser les images (sans avoir le code)

### Prérequis
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé

### Lancement
1. Télécharge le fichier `docker-compose.yml` depuis le repo GitHub
2. Dans le dossier où se trouve le fichier :
```bash
docker compose -f docker-compose.prod.yml up -d
```
3. La première fois, Docker va :
   - Télécharger les 4 images (~500MB)
   - Télécharger Mistral (~4GB) ← ça prend du temps, c'est normal
4. Ensuite les APIs sont accessibles :
   - `http://localhost:8000/docs` → API REST Transcription
   - `http://localhost:8001` → API WebSocket Transcription
   - `http://localhost:8002/docs` → API Génération de requêtes
   - `http://localhost:8003/docs` → API Complétion de formulaire

### Arrêter
```bash
docker compose -f docker-compose.yml down
```

---

## 🔧 Publier une nouvelle version d'une image

### 1. Se connecter à GitHub Container Registry (une seule fois par machine)

Génère un token GitHub :
- GitHub → ton avatar → `Settings` → `Developer settings`
- `Personal access tokens` → `Tokens (classic)` → `Generate new token`
- Coche : `write:packages`, `read:packages`, `delete:packages`
- Expiration : `No expiration`
- Copie le token généré

Connecte-toi :
```bash
echo "TON_TOKEN_GITHUB" | docker login ghcr.io -u TON_PSEUDO_GITHUB --password-stdin
```

### 2. Builder l'image modifiée

```bash
# api-generation
cd API_generation_requetes
docker build -t ghcr.io/speechcore-sae/api-generation:latest .

# api-completion
cd Completion_formulaire
docker build -t ghcr.io/speechcore-sae/api-completion:latest .

# api-rest
docker build -f Dockerfile.api_rest -t ghcr.io/speechcore-sae/api-rest:latest .

# api-websocket
docker build -f Dockerfile.api_websocket -t ghcr.io/speechcore-sae/api-websocket:latest .
```

### 3. Pusher l'image

```bash
docker push ghcr.io/speechcore-sae/api-generation:latest
docker push ghcr.io/speechcore-sae/api-completion:latest
docker push ghcr.io/speechcore-sae/api-rest:latest
docker push ghcr.io/speechcore-sae/api-websocket:latest
```

### 4. Vérifier

L'image est visible sur :
`https://github.com/orgs/speechcore-sae/packages`

---

## 🔄 Mettre à jour les images en local

Si quelqu'un a pushé une nouvelle version :
```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

---

## ⚠️ Notes importantes

- **Mistral** est téléchargé automatiquement au premier lancement (~4GB), il est ensuite mis en cache dans un volume Docker — pas besoin de le re-télécharger
- **`localhost`** dans les configs des bases de données doit être remplacé par `host.docker.internal` si tes bases tournent aussi dans Docker
- Si un conteneur ne démarre pas, consulte les logs : `docker compose logs nom-du-service`
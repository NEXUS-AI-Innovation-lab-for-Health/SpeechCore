#!/usr/bin/env python3
"""
API WebSocket pour transcription audio en temps réel
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import base64
import tempfile
from pathlib import Path
import io

from transcription_engines import transcrire_vosk, transcrire_whisper, transcrire_gladia
from audio_processing import analyser_audio, reduire_bruit
import soundfile as sf
import numpy as np


app = FastAPI(
    title="API WebSocket Transcription",
    description="Transcription en temps réel via WebSocket",
    version="2.0.0"
)


# Page HTML de test
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Transcription WebSocket</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 20px; }
        .config { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }
        label { display: block; margin: 10px 0 5px; font-weight: bold; }
        select, input[type="number"] { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #4CAF50; color: white; padding: 10px 25px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #status { padding: 15px; margin: 15px 0; border-radius: 5px; text-align: center; font-weight: bold; }
        .status-connecting { background: #fff3cd; color: #856404; }
        .status-connected { background: #d4edda; color: #155724; }
        .status-disconnected { background: #f8d7da; color: #721c24; }
        .status-processing { background: #d1ecf1; color: #0c5460; }
        #result { background: #f9f9f9; padding: 20px; border-radius: 5px; min-height: 200px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; font-family: monospace; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Transcription WebSocket v2</h1>
        
        <div class="config">
            <label>Moteur:</label>
            <select id="engine">
                <option value="vosk">Vosk (rapide, local)</option>
                <option value="whisper">Whisper (qualité, local)</option>
                <option value="gladia">Gladia (cloud, meilleure qualité)</option>
            </select>
            
            <label>Modèle Vosk:</label>
            <select id="modele_vosk">
                <option value="petit">Petit (rapide)</option>
                <option value="grand">Grand (qualité)</option>
            </select>
            
            <label>Config Whisper:</label>
            <select id="config_whisper">
                <option value="cpu_rapide">CPU Rapide</option>
                <option value="cpu_qualite">CPU Qualité</option>
                <option value="gpu_equilibre">GPU Équilibré</option>
            </select>
            
            <label>Nombre de locuteurs:</label>
            <input type="number" id="nb_locuteurs" value="2" min="0" max="10">
            
            <label>Réduction de bruit:</label>
            <select id="methode_bruit">
                <option value="false">Désactivée</option>
                <option value="noisereduce">NoiseReduce</option>
                <option value="silero">Silero VAD</option>
            </select>
            
            <label>Environnement:</label>
            <select id="type_environnement">
                <option value="1">Salle silencieuse</option>
                <option value="2" selected>Bureau/Normal</option>
                <option value="3">Environnement bruyant</option>
                <option value="4">Bruit constant</option>
            </select>
        </div>
        
        <div style="text-align: center;">
            <input type="file" id="fileInput" accept=".wav" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">📁 Fichier WAV</button>
            <button id="sendBtn" onclick="sendAudio()" disabled>🚀 Transcrire</button>
        </div>
        
        <div id="status" class="status-disconnected">Sélectionnez un fichier</div>
        <div id="result">En attente...</div>
    </div>

    <script>
        let ws = null;
        let selectedFile = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('status').textContent = 'Fichier: ' + selectedFile.name;
                document.getElementById('status').className = 'status-connected';
            }
        });
        
        async function sendAudio() {
            if (!selectedFile) return;
            
            const arrayBuffer = await selectedFile.arrayBuffer();
            const bytes = new Uint8Array(arrayBuffer);
            
            // Conversion base64 optimisée pour gros fichiers
            let binary = '';
            const chunkSize = 0x8000; // 32KB chunks
            for (let i = 0; i < bytes.length; i += chunkSize) {
                const chunk = bytes.subarray(i, i + chunkSize);
                binary += String.fromCharCode.apply(null, chunk);
            }
            const base64Audio = btoa(binary);
            
            const config = {
                engine: document.getElementById('engine').value,
                modele_vosk: document.getElementById('modele_vosk').value,
                config_whisper: document.getElementById('config_whisper').value,
                nb_locuteurs: parseInt(document.getElementById('nb_locuteurs').value),
                methode_bruit: document.getElementById('methode_bruit').value,
                type_environnement: document.getElementById('type_environnement').value,
                audio: base64Audio
            };
            
            document.getElementById('status').textContent = 'Connexion...';
            document.getElementById('status').className = 'status-connecting';
            
            ws = new WebSocket('ws://localhost:8000/ws/transcribe');
            
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Envoi...';
                ws.send(JSON.stringify(config));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status') {
                    document.getElementById('status').textContent = data.message;
                    document.getElementById('status').className = 'status-processing';
                } else if (data.type === 'result') {
                    let output = '='.repeat(70) + '\\n';
                    output += 'TRANSCRIPTION TERMINÉE\\n';
                    output += '='.repeat(70) + '\\n\\n';
                    
                    if (data.stats) {
                        output += 'STATISTIQUES\\n';
                        output += `Mots: ${data.stats.nombre_mots}\\n`;
                        output += `Locuteurs: ${data.stats.nombre_locuteurs}\\n\\n`;
                    }
                    
                    if (data.transcription_avec_locuteurs) {
                        output += 'AVEC LOCUTEURS:\\n' + '='.repeat(70) + '\\n';
                        output += data.transcription_avec_locuteurs + '\\n\\n';
                    }
                    
                    output += 'TRANSCRIPTION BRUTE:\\n' + '='.repeat(70) + '\\n';
                    output += data.transcription_complete;
                    
                    document.getElementById('result').textContent = output;
                    document.getElementById('status').textContent = '✅ Terminé!';
                    document.getElementById('status').className = 'status-connected';
                } else if (data.type === 'error') {
                    document.getElementById('result').textContent = 'Erreur: ' + data.message;
                    document.getElementById('status').textContent = '❌ Erreur';
                    document.getElementById('status').className = 'status-disconnected';
                }
            };
            
            ws.onerror = () => {
                document.getElementById('status').textContent = '❌ Erreur WebSocket';
                document.getElementById('status').className = 'status-disconnected';
            };
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(content=HTML_PAGE)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Recevoir config
        data = await websocket.receive_text()
        config = json.loads(data)
        
        # Décoder audio
        await websocket.send_json({"type": "status", "message": "Chargement audio..."})
        audio_bytes = base64.b64decode(config['audio'])
        
        # Créer fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            # Traiter selon le moteur
            await websocket.send_json({"type": "status", "message": "Transcription..."})
            
            if config['engine'] == 'vosk':
                resultats = transcrire_vosk(
                    tmp_path,
                    modele=config.get('modele_vosk', 'grand'),
                    nb_locuteurs=config.get('nb_locuteurs', 2),
                    reduction_bruit=config.get('methode_bruit', 'false') != 'false',
                    type_environnement=config.get('type_environnement', '2'),
                    methode_bruit=config.get('methode_bruit', 'noisereduce')
                )
            
            elif config['engine'] == 'whisper':
                resultats = transcrire_whisper(
                    tmp_path,
                    config=config.get('config_whisper', 'cpu_rapide'),
                    nb_locuteurs=config.get('nb_locuteurs', 2),
                    reduction_bruit=config.get('methode_bruit', 'false') != 'false',
                    type_environnement=config.get('type_environnement', '2'),
                    methode_bruit=config.get('methode_bruit', 'noisereduce')
                )
            
            else:  # gladia
                resultats = transcrire_gladia(
                    tmp_path,
                    nb_locuteurs=config.get('nb_locuteurs', 0)
                )
            
            # Envoyer résultat
            await websocket.send_json({
                "type": "result",
                "transcription_complete": resultats['texte_brut'],
                "transcription_avec_locuteurs": resultats.get('texte_diarise', ''),
                "stats": {
                    "nombre_mots": resultats.get('nb_mots', 0),
                    "nombre_locuteurs": resultats.get('nb_locuteurs', 0)
                }
            })
        
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    print("🌐 Démarrage de l'API WebSocket...")
    print("📡 http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
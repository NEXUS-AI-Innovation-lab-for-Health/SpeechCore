#!/usr/bin/env python3
"""
Module des moteurs de transcription
Vosk, Whisper, Gladia, Groq
"""

from pathlib import Path
import os
import wave
import json
import soundfile as sf
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import requests
import time

from audio_processing import analyser_audio, reduire_bruit


# ========== CONFIGURATION ==========

MODELES_VOSK = {
    "petit": {
        "nom": "⚡ Petit Français (41 MB)",
        "path": "vosk-model-small-fr-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
    },
    "grand": {
        "nom": "🇫🇷 Grand Français (1.5 GB)",
        "path": "vosk-model-fr-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
    }
}

CONFIGS_WHISPER = {
    "cpu_rapide":    {"model_size": "base",     "device": "cpu",  "compute_type": "int8"},
    "cpu_qualite":   {"model_size": "small",    "device": "cpu",  "compute_type": "int8"},
    "gpu_equilibre": {"model_size": "medium",   "device": "cuda", "compute_type": "float16"},
    "gpu_max":       {"model_size": "large-v3", "device": "cuda", "compute_type": "float16"},
    "ultra_rapide":  {"model_size": "tiny",     "device": "auto", "compute_type": "int8"}
}

GLADIA_API_KEY = os.environ.get("GLADIA_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")   # Configurer dans docker-compose.yml ou .env


# ========== CACHE MODÈLES (singletons) ==========
# Les modèles sont chargés une seule fois au démarrage du serveur
# et réutilisés pour toutes les requêtes — gain majeur sur la latence.

_vosk_models:    dict                = {}
_whisper_models: dict                = {}
_voice_encoder:  VoiceEncoder | None = None


def _get_vosk_model(modele: str):
    """Retourne le modèle Vosk mis en cache, le charge si nécessaire."""
    if modele not in _vosk_models:
        from vosk import Model
        modele_info = MODELES_VOSK.get(modele, MODELES_VOSK["grand"])
        if not Path(modele_info['path']).exists():
            raise FileNotFoundError(
                f"Modèle Vosk non installé: {modele_info['path']}\n"
                f"Lancez setup_vosk_models.py pour le télécharger."
            )
        _vosk_models[modele] = Model(modele_info['path'])
    return _vosk_models[modele]


def _get_whisper_model(config: str):
    """Retourne le modèle Whisper mis en cache, le charge si nécessaire."""
    if config not in _whisper_models:
        from faster_whisper import WhisperModel
        wcfg = CONFIGS_WHISPER.get(config, CONFIGS_WHISPER["cpu_rapide"])
        _whisper_models[config] = WhisperModel(
            wcfg['model_size'],
            device=wcfg['device'],
            compute_type=wcfg['compute_type']
        )
    return _whisper_models[config]


def _get_voice_encoder() -> VoiceEncoder:
    """Retourne le VoiceEncoder mis en cache (Resemblyzer)."""
    global _voice_encoder
    if _voice_encoder is None:
        _voice_encoder = VoiceEncoder()
    return _voice_encoder


# ========== DIARIZATION ==========

def re_segmenter(segments_avec_temps, max_duration=5.0):
    """Découpe les longs segments pour améliorer la diarisation."""
    nouveaux_segments = []

    for segment in segments_avec_temps:
        duration = segment['end'] - segment['start']

        if duration <= max_duration:
            nouveaux_segments.append(segment)
        else:
            num_chunks = int(np.ceil(duration / max_duration))
            mots = segment['text'].split()

            if not mots:
                continue

            mots_par_chunk = max(1, len(mots) // num_chunks)

            for i in range(num_chunks):
                start     = segment['start'] + i * max_duration
                end       = min(segment['start'] + (i + 1) * max_duration, segment['end'])
                debut_mot = i * mots_par_chunk
                fin_mot   = (i + 1) * mots_par_chunk if i < num_chunks - 1 else len(mots)

                if debut_mot < len(mots):
                    nouveaux_segments.append({
                        'start': start,
                        'end':   end,
                        'text':  ' '.join(mots[debut_mot:fin_mot])
                    })

    return nouveaux_segments


def diarizer_avec_resemblyzer(fichier_audio: Path, segments_avec_temps, n_speakers: int):
    """Identifie les locuteurs avec Resemblyzer (encodeur mis en cache)."""
    try:
        encoder    = _get_voice_encoder()   # ← cache
        audio_data, sample_rate = sf.read(str(fichier_audio))

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        embeddings     = []
        valid_segments = []

        for segment in segments_avec_temps:
            start_sample = max(0, int(segment['start'] * sample_rate))
            end_sample   = min(len(audio_data), int(segment['end'] * sample_rate))
            seg_audio    = audio_data[start_sample:end_sample]

            if len(seg_audio) < sample_rate * 0.3:
                continue

            try:
                seg_wav = preprocess_wav(seg_audio, sample_rate)
                embed   = encoder.embed_utterance(seg_wav)
                embeddings.append(embed)
                valid_segments.append(segment)
            except Exception:
                continue

        if len(embeddings) < 2:
            return None

        n_speakers = min(n_speakers, len(embeddings))
        labels = AgglomerativeClustering(n_clusters=n_speakers, linkage='average') \
                     .fit_predict(np.array(embeddings))

        for segment, label in zip(valid_segments, labels):
            segment['speaker'] = int(label)

        return valid_segments

    except Exception:
        return None


def formater_transcription_avec_locuteurs(segments_diarises):
    """Formate la transcription avec les locuteurs."""
    if not segments_diarises:
        return None

    texte_avec_locuteurs = []
    current_speaker      = None
    current_text         = []

    for segment in segments_diarises:
        speaker_id = segment['speaker']

        if speaker_id == current_speaker:
            current_text.append(segment['text'])
        else:
            if current_text:
                texte_avec_locuteurs.append(
                    f"[Locuteur {current_speaker}] {' '.join(current_text)}"
                )
            current_speaker = speaker_id
            current_text    = [segment['text']]

    if current_text:
        texte_avec_locuteurs.append(
            f"[Locuteur {current_speaker}] {' '.join(current_text)}"
        )

    return "\n\n".join(texte_avec_locuteurs)


# ========== MOTEUR VOSK ==========

def transcrire_vosk(
    fichier_audio: Path,
    modele: str           = "grand",
    nb_locuteurs: int     = 2,
    reduction_bruit: bool = True,
    type_environnement: str = "2",
    methode_bruit: str    = "noisereduce"
):
    """Transcription avec Vosk (modèle mis en cache)."""
    from vosk import KaldiRecognizer

    stats_audio = analyser_audio(fichier_audio)
    model       = _get_vosk_model(modele)       # ← cache, pas de rechargement

    fichier_a_transcrire = fichier_audio
    fichier_temp         = None

    if reduction_bruit:
        fichier_nettoye = reduire_bruit(fichier_audio, type_environnement, methode_bruit)
        fichier_a_transcrire = Path(fichier_nettoye)
        if fichier_nettoye != str(fichier_audio):
            fichier_temp = Path(fichier_nettoye)

    wf  = wave.open(str(fichier_a_transcrire), "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    segments_avec_temps = []
    texte_brut_complet  = []
    total_mots          = 0

    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get('result'):
                words = result['result']
                segments_avec_temps.append({
                    'start': words[0]['start'],
                    'end':   words[-1]['end'],
                    'text':  ' '.join(w['word'] for w in words)
                })
                texte_brut_complet.append(' '.join(w['word'] for w in words))
                total_mots += len(words)

    final_result = json.loads(rec.FinalResult())
    if final_result.get('result'):
        words = final_result['result']
        segments_avec_temps.append({
            'start': words[0]['start'],
            'end':   words[-1]['end'],
            'text':  ' '.join(w['word'] for w in words)
        })
        texte_brut_complet.append(' '.join(w['word'] for w in words))
        total_mots += len(words)

    wf.close()

    segments_optimises = re_segmenter(segments_avec_temps, max_duration=5.0)
    segments_diarises  = diarizer_avec_resemblyzer(
        fichier_a_transcrire, segments_optimises, nb_locuteurs
    )

    if fichier_temp and fichier_temp.exists():
        fichier_temp.unlink()

    return {
        'texte_brut':       ' '.join(texte_brut_complet),
        'texte_diarise':    formater_transcription_avec_locuteurs(segments_diarises),
        'stats_audio':      stats_audio,
        'nb_mots':          total_mots,
        'nb_segments':      len(segments_avec_temps),
        'nb_locuteurs':     nb_locuteurs,
        'langue':           'fr',
        'confiance_langue': 1.0
    }


# ========== MOTEUR WHISPER ==========

def transcrire_whisper(
    fichier_audio: Path,
    config: str           = "cpu_rapide",
    nb_locuteurs: int     = 2,
    reduction_bruit: bool = True,
    type_environnement: str = "2",
    methode_bruit: str    = "noisereduce",
    initial_prompt: str   = ""
):
    """
    Transcription avec Whisper (modèle mis en cache).
    initial_prompt : derniers mots du chunk précédent pour améliorer la continuité
                     inter-chunks et réduire les hallucinations.
    """
    stats_audio = analyser_audio(fichier_audio)
    model       = _get_whisper_model(config)    # ← cache, pas de rechargement

    fichier_a_transcrire = fichier_audio
    fichier_temp         = None

    if reduction_bruit:
        fichier_nettoye = reduire_bruit(fichier_audio, type_environnement, methode_bruit)
        fichier_a_transcrire = Path(fichier_nettoye)
        if fichier_nettoye != str(fichier_audio):
            fichier_temp = Path(fichier_nettoye)

    segments, info = model.transcribe(
        str(fichier_a_transcrire),
        language='fr',
        beam_size=1,
        vad_filter=True,
        initial_prompt=initial_prompt or None   # ← contexte inter-chunks
    )

    segments_avec_temps = []
    texte_brut_complet  = []
    total_mots          = 0

    for segment in segments:
        segments_avec_temps.append({
            'start': segment.start,
            'end':   segment.end,
            'text':  segment.text.strip()
        })
        texte_brut_complet.append(segment.text.strip())
        total_mots += len(segment.text.split())

    segments_diarises = diarizer_avec_resemblyzer(
        fichier_a_transcrire, segments_avec_temps, nb_locuteurs
    )

    if fichier_temp and fichier_temp.exists():
        fichier_temp.unlink()

    return {
        'texte_brut':       ' '.join(texte_brut_complet),
        'texte_diarise':    formater_transcription_avec_locuteurs(segments_diarises),
        'stats_audio':      stats_audio,
        'nb_mots':          total_mots,
        'nb_segments':      len(segments_avec_temps),
        'nb_locuteurs':     nb_locuteurs,
        'langue':           info.language,
        'confiance_langue': info.language_probability
    }


# ========== MOTEUR GROQ ==========

def transcrire_groq(
    fichier_audio: Path,
    nb_locuteurs: int  = 0,
    initial_prompt: str = ""
):
    """
    Transcription avec l'API Groq (Whisper-large-v3, très rapide ~2-5x Whisper local).
    Nécessite : pip install groq
    Clé API   : https://console.groq.com → API Keys → mettre dans GROQ_API_KEY
    """
    from groq import Groq

    stats_audio = analyser_audio(fichier_audio)
    client      = Groq(api_key=GROQ_API_KEY)

    with open(fichier_audio, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(fichier_audio.name, f.read()),
            model="whisper-large-v3",
            language="fr",
            response_format="verbose_json",
            prompt=initial_prompt or None
        )

    full_transcript     = response.text or ""
    segments_avec_temps = []

    for seg in (response.segments or []):
        segments_avec_temps.append({
            'start': seg['start'],
            'end':   seg['end'],
            'text':  seg['text'].strip()
        })

    segments_diarises = None
    if nb_locuteurs > 0 and len(segments_avec_temps) >= 2:
        segments_diarises = diarizer_avec_resemblyzer(
            fichier_audio, segments_avec_temps, nb_locuteurs
        )

    return {
        'texte_brut':       full_transcript,
        'texte_diarise':    formater_transcription_avec_locuteurs(segments_diarises),
        'stats_audio':      stats_audio,
        'nb_mots':          len(full_transcript.split()),
        'nb_segments':      len(segments_avec_temps),
        'nb_locuteurs':     nb_locuteurs,
        'langue':           'fr',
        'confiance_langue': 1.0
    }


# ========== MOTEUR GLADIA ==========

def transcrire_gladia(
    fichier_audio: Path,
    nb_locuteurs: int = 0
):
    """
    Transcription avec Gladia API.
    Note : cette fonction est synchrone (time.sleep).
    Elle doit être appelée via run_in_executor côté FastAPI.
    """
    stats_audio = analyser_audio(fichier_audio)

    # Upload
    with open(fichier_audio, "rb") as audio_file:
        upload_response = requests.post(
            "https://api.gladia.io/v2/upload",
            headers={"x-gladia-key": GLADIA_API_KEY},
            files={"audio": (fichier_audio.name, audio_file, "audio/wav")},
            timeout=120
        )

    if upload_response.status_code != 200:
        raise Exception(f"Erreur upload Gladia: {upload_response.text}")

    audio_url = upload_response.json()["audio_url"]

    # Transcription
    transcription_response = requests.post(
        "https://api.gladia.io/v2/transcription",
        headers={"x-gladia-key": GLADIA_API_KEY, "Content-Type": "application/json"},
        json={
            "audio_url": audio_url,
            "language_config": {"languages": ["fr"]},
            "diarization": True,
            "diarization_config": {
                "number_of_speakers": nb_locuteurs if nb_locuteurs > 0 else None,
                "min_speakers": 1,
                "max_speakers": 10
            }
        },
        timeout=30
    )

    if transcription_response.status_code not in [200, 201]:
        raise Exception(f"Erreur transcription Gladia: {transcription_response.text}")

    result_url = transcription_response.json()["result_url"]

    # Polling (run_in_executor côté API pour ne pas bloquer la boucle asyncio)
    for _ in range(120):
        result_response = requests.get(
            result_url,
            headers={"x-gladia-key": GLADIA_API_KEY},
            timeout=30
        )

        if result_response.status_code != 200:
            raise Exception(f"Erreur résultat Gladia: {result_response.text}")

        result_data = result_response.json()
        status      = result_data.get("status")

        if status == "done":
            transcription_result  = result_data["result"]["transcription"]
            utterances            = transcription_result.get("utterances", [])
            full_transcript       = transcription_result.get("full_transcript", "")
            locuteurs_uniques     = set(u.get("speaker", 0) for u in utterances)

            texte_avec_locuteurs = []
            current_speaker      = None
            current_text         = []

            for utterance in utterances:
                speaker = utterance.get("speaker", 0)
                text    = utterance.get("text", "").strip()
                if speaker == current_speaker:
                    current_text.append(text)
                else:
                    if current_text:
                        texte_avec_locuteurs.append(
                            f"[Locuteur {current_speaker}] {' '.join(current_text)}"
                        )
                    current_speaker = speaker
                    current_text    = [text]

            if current_text:
                texte_avec_locuteurs.append(
                    f"[Locuteur {current_speaker}] {' '.join(current_text)}"
                )

            return {
                'texte_brut':       full_transcript,
                'texte_diarise':    "\n\n".join(texte_avec_locuteurs),
                'stats_audio':      stats_audio,
                'nb_mots':          len(full_transcript.split()) if full_transcript else 0,
                'nb_segments':      len(utterances),
                'nb_locuteurs':     len(locuteurs_uniques),
                'langue':           'fr',
                'confiance_langue': 1.0
            }

        elif status == "error":
            raise Exception(f"Erreur Gladia: {result_data.get('error')}")

        time.sleep(2)

    raise Exception("Timeout: transcription Gladia trop longue (>4 min)")

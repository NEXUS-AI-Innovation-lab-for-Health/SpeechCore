/**
 * voice-recognition-page.tsx — Transcription temps réel via WebSocket
 *
 * PROTOCOLE /ws/transcribe (port 8000) :
 *   - vosk    → { engine, audio, nb_locuteurs, modele_vosk, methode_bruit, type_environnement }
 *   - whisper → { engine, audio, nb_locuteurs, config_whisper, methode_bruit, type_environnement }
 *   - gladia  → { engine, audio, nb_locuteurs }
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Square, Mic, Save, Trash2, AlertCircle, Radio, Users, ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";

// ── Constantes ────────────────────────────────────────────────────────────────
const WS_URL    = "ws://localhost:8000/ws/transcribe";
const TARGET_SR = 16000;
const SILENCE   = 0.004;
const NB_MIN: Record<string, number> = { vosk: 1, whisper: 1, gladia: 0 };

// ── Utilitaires audio ─────────────────────────────────────────────────────────
function resample(f32: Float32Array, srcSR: number): Float32Array {
  if (srcSR === TARGET_SR) return f32;
  const ratio = srcSR / TARGET_SR;
  const out = new Float32Array(Math.round(f32.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio, idx = Math.floor(pos), frac = pos - idx;
    out[i] = (f32[idx] ?? 0) + frac * ((f32[idx + 1] ?? 0) - (f32[idx] ?? 0));
  }
  return out;
}

function toWav(f32: Float32Array, sr: number): Blob {
  const n = f32.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const s = (o: number, str: string) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
  s(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); s(8, "WAVE"); s(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const x = Math.max(-1, Math.min(1, f32[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ── Types ─────────────────────────────────────────────────────────────────────
type Engine    = "whisper" | "vosk" | "gladia";
type WsStatus  = "disconnected" | "connecting" | "ready" | "error";
type LogTag    = "SEND" | "RECV" | "DROP" | "WARN" | "ERR";

interface LogLine {
  id:  number;
  ts:  string;
  tag: LogTag;
  msg: string;
}

// ── Composant ─────────────────────────────────────────────────────────────────
export default function VoiceRecognitionPage() {

  // ─ État principal
  const [isRecording, setIsRecording]     = useState(false);
  const [transcript, setTranscript]       = useState("");
  const [wsStatus, setWsStatus]           = useState<WsStatus>("disconnected");
  const [error, setError]                 = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);

  // ─ Config soignant
  const [engine, setEngine]           = useState<Engine>("whisper");
  const [nbLocuteurs, setNbLocuteurs] = useState<1 | 2>(2);
  const [chunkMs, setChunkMs]         = useState(2000);
  const [showConfig, setShowConfig]   = useState(false);

  // ─ Log réseau
  const [logs, setLogs]           = useState<LogLine[]>([]);
  const [showLog, setShowLog]     = useState(false);
  const logCountRef               = useRef(0);
  const logBoxRef                 = useRef<HTMLDivElement>(null);

  // ─ Stats
  const [statSent, setStatSent] = useState(0);
  const [statDone, setStatDone] = useState(0);
  const [statRtt,  setStatRtt]  = useState<number | null>(null);

  // ─ Refs audio
  const streamRef      = useRef<MediaStream | null>(null);
  const actxRef        = useRef<AudioContext | null>(null);
  const procRef        = useRef<ScriptProcessorNode | null>(null);
  const pcmBufRef      = useRef<Float32Array[]>([]);
  const nativeSRRef    = useRef<number>(TARGET_SR);
  const wsQueueRef     = useRef<Blob[]>([]);
  const wsBusyRef      = useRef(false);
  const chunkTmrRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const runningRef     = useRef(false);
  const engineRef      = useRef(engine);
  const nbLocuteursRef = useRef(nbLocuteurs);
  const chunkMsRef     = useRef(chunkMs);

  useEffect(() => { engineRef.current      = engine;      }, [engine]);
  useEffect(() => { nbLocuteursRef.current = nbLocuteurs; }, [nbLocuteurs]);
  useEffect(() => { chunkMsRef.current     = chunkMs;     }, [chunkMs]);

  // Ajuster nb_locuteurs si on passe sur Gladia
  useEffect(() => {
    if (engine === "gladia" && nbLocuteurs < 1) setNbLocuteurs(1);
  }, [engine]);

  // Auto-scroll du log
  useEffect(() => {
    if (logBoxRef.current) logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [logs]);

  // Chronomètre
  useEffect(() => {
    if (isRecording) {
      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording]);

  const formatTime = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;

  // ── Logger ────────────────────────────────────────────────────────────────
  const addLog = useCallback((tag: LogTag, msg: string) => {
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8) + "." + String(now.getMilliseconds()).padStart(3, "0");
    logCountRef.current += 1;
    setLogs(prev => [...prev.slice(-199), { id: logCountRef.current, ts, tag, msg }]);
  }, []);

  // ── Payload builder ───────────────────────────────────────────────────────
  const buildPayload = (b64: string): Record<string, unknown> => {
    const eng = engineRef.current;
    const base: Record<string, unknown> = { audio: b64, engine: eng, nb_locuteurs: nbLocuteursRef.current };
    if (eng === "vosk") {
      base.modele_vosk        = "grand";
      base.methode_bruit      = "false";
      base.type_environnement = "2";
    } else if (eng === "whisper") {
      base.config_whisper     = "cpu_rapide";
      base.methode_bruit      = "false";
      base.type_environnement = "2";
    }
    return base;
  };

  // ── Pipeline WebSocket séquentielle ───────────────────────────────────────
  const processQueue = useCallback(() => {
    if (wsBusyRef.current || wsQueueRef.current.length === 0) return;
    const wav = wsQueueRef.current.shift()!;
    wsBusyRef.current = true;
    const t0 = Date.now();

    blobToBase64(wav).then((b64) => {
      const kb = (wav.size / 1024).toFixed(1);
      const payload = buildPayload(b64);

      let ws: WebSocket;
      try { ws = new WebSocket(WS_URL); }
      catch (e: any) {
        addLog("ERR", `WebSocket: ${e?.message ?? e}`);
        wsBusyRef.current = false;
        processQueue();
        return;
      }

      addLog("SEND", `→ WS ouvert, envoi ${kb} KB (${engineRef.current})`);

      ws.onopen = () => ws.send(JSON.stringify(payload));

      ws.onmessage = (ev) => {
        try {
          const d = JSON.parse(ev.data as string);
          if (d.type === "status") {
            addLog("RECV", `← status: "${d.message}"`);
          } else if (d.type === "result") {
            const rtt = Date.now() - t0;
            setStatRtt(rtt);
            setStatDone(n => n + 1);
            const txt = (d.transcription_complete || "").trim();
            if (txt) {
              addLog("RECV", `← ${rtt}ms : "${txt.slice(0, 80)}"`);
              setTranscript(prev => prev + (prev && !prev.endsWith(" ") ? " " : "") + txt);
            } else {
              addLog("WARN", `← ${rtt}ms — résultat vide`);
            }
          } else if (d.type === "error") {
            addLog("ERR", `← erreur serveur: ${d.message}`);
          }
        } catch { /* ignore */ }
      };

      ws.onerror = () => {
        addLog("ERR", "WS onerror — serveur sur :8000 ?");
        setError("Impossible de joindre le serveur (port 8000). Vérifiez que api_websocket.py est lancé.");
        setWsStatus("error");
      };

      ws.onclose = (ev) => {
        addLog("RECV", `← WS fermé (code ${ev.code}) — ${Date.now() - t0}ms`);
        wsBusyRef.current = false;
        processQueue();
      };
    });
  }, [addLog]);

  // ── Flush PCM → WAV → queue ───────────────────────────────────────────────
  const flushChunk = useCallback(() => {
    if (pcmBufRef.current.length === 0) {
      addLog("DROP", "chunk vide (silence)");
      return;
    }
    const total = pcmBufRef.current.reduce((s, a) => s + a.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const a of pcmBufRef.current) { merged.set(a, off); off += a.length; }
    pcmBufRef.current = [];

    const rms = Math.sqrt(merged.reduce((s, v) => s + v * v, 0) / merged.length);
    if (rms < SILENCE) {
      addLog("DROP", `silence RMS=${rms.toFixed(5)}`);
      return;
    }

    const wav = toWav(resample(merged, nativeSRRef.current), TARGET_SR);
    const kb  = (wav.size / 1024).toFixed(1);
    setStatSent(n => n + 1);
    addLog("SEND", `chunk — ${kb} KB, RMS=${rms.toFixed(4)}`);
    wsQueueRef.current.push(wav);
    processQueue();
  }, [addLog, processQueue]);

  // ── Démarrer ──────────────────────────────────────────────────────────────
  const startRecording = async () => {
    setError(null);
    setWsStatus("connecting");
    setStatSent(0); setStatDone(0); setStatRtt(null);
    addLog("RECV", `▶ START — moteur=${engine}, chunk=${chunkMs}ms, locuteurs=${nbLocuteurs}`);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
      });
      streamRef.current = stream;

      const actx = new (window.AudioContext || (window as any).webkitAudioContext)();
      actxRef.current     = actx;
      nativeSRRef.current = actx.sampleRate;
      addLog("RECV", `Micro ouvert — ${actx.sampleRate}Hz → ${TARGET_SR}Hz`);

      const src  = actx.createMediaStreamSource(stream);
      const proc = actx.createScriptProcessor(4096, 1, 1);
      procRef.current = proc;
      proc.onaudioprocess = (e) => {
        if (runningRef.current)
          pcmBufRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc);
      proc.connect(actx.destination);

      runningRef.current  = true;
      pcmBufRef.current   = [];
      wsQueueRef.current  = [];
      wsBusyRef.current   = false;
      chunkTmrRef.current = setInterval(flushChunk, chunkMsRef.current);

      setWsStatus("ready");
      setIsRecording(true);
      setShowConfig(false);
    } catch (e: any) {
      addLog("ERR", `getUserMedia: ${e?.message ?? e}`);
      setError("Impossible d'accéder au microphone.");
      setWsStatus("error");
    }
  };

  // ── Arrêter ───────────────────────────────────────────────────────────────
  const stopRecording = () => {
    runningRef.current = false;
    setIsRecording(false);
    setWsStatus("disconnected");
    if (chunkTmrRef.current) clearInterval(chunkTmrRef.current);
    flushChunk();
    addLog("RECV", `■ STOP`);
    try { procRef.current?.disconnect(); }  catch { /* ignore */ }
    try { streamRef.current?.getTracks().forEach(t => t.stop()); } catch { /* ignore */ }
    try { actxRef.current?.close(); }       catch { /* ignore */ }
    procRef.current = streamRef.current = actxRef.current = null;
  };

  // ── Sauvegarde ────────────────────────────────────────────────────────────
  const saveRecording = () => {
    if (!transcript.trim()) return;
    const recordings = JSON.parse(localStorage.getItem("medvoice_recordings") || "[]");
    recordings.push({
      id: Date.now().toString(), text: transcript, date: new Date().toISOString(),
      nom:      extractField(transcript, ["nom", "appelle", "s'appelle"]),
      prenom:   extractField(transcript, ["prénom", "prenom"]),
      symptomes: ["douleur","mal","fièvre","toux","fatigue","nausée"]
        .filter(s => transcript.toLowerCase().includes(s)).join(", "),
    });
    localStorage.setItem("medvoice_recordings", JSON.stringify(recordings));
    alert("Enregistrement sauvegardé !");
    setTranscript("");
  };

  const extractField = (text: string, kws: string[]) => {
    const lower = text.toLowerCase();
    for (const kw of kws) {
      const idx = lower.indexOf(kw);
      if (idx !== -1) return text.substring(idx + kw.length).trim().split(/[\s,.]+/)[0] || "";
    }
    return "";
  };

  const isConnecting = wsStatus === "connecting";
  const isGladia     = engine === "gladia";

  const engineLabel: Record<Engine, string> = {
    whisper: "🎯 Whisper", vosk: "⚡ Vosk", gladia: "☁️ Gladia",
  };

  const tagColor: Record<LogTag, string> = {
    SEND: "text-blue-400",
    RECV: "text-emerald-400",
    DROP: "text-gray-500",
    WARN: "text-amber-400",
    ERR:  "text-red-400",
  };

  // ── JSX ───────────────────────────────────────────────────────────────────
  return (
    <div className="max-w-4xl mx-auto space-y-5 px-4 py-6">

      {/* ── En-tête ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Consultation</h1>
          <p className="text-sm text-gray-500 mt-0.5">Transcription en direct — toutes les {chunkMs / 1000}s</p>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium border ${
          wsStatus === "ready"      ? "bg-green-50 border-green-200 text-green-700"    :
          wsStatus === "connecting" ? "bg-yellow-50 border-yellow-200 text-yellow-700" :
          wsStatus === "error"      ? "bg-red-50 border-red-200 text-red-700"          :
          "bg-gray-50 border-gray-200 text-gray-500"
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            wsStatus === "ready"      ? "bg-green-500"               :
            wsStatus === "connecting" ? "bg-yellow-400 animate-pulse" :
            wsStatus === "error"      ? "bg-red-500"                  : "bg-gray-300"
          }`} />
          {wsStatus === "ready"        && "En écoute"}
          {wsStatus === "connecting"   && "Connexion..."}
          {wsStatus === "error"        && "Erreur"}
          {wsStatus === "disconnected" && "En attente"}
        </div>
      </div>

      {/* ── Erreur ── */}
      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-red-700">{error}</p>
            <p className="text-xs text-red-500 mt-1">
              Lancez : <code className="bg-red-100 px-1 rounded">python api_websocket.py</code> (port 8000)
            </p>
          </div>
        </div>
      )}

      {/* ── Config accordéon ── */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <button
          onClick={() => !isRecording && setShowConfig(o => !o)}
          disabled={isRecording}
          className={`w-full flex items-center justify-between px-5 py-3.5 text-left transition-colors
            ${isRecording ? "opacity-50 cursor-not-allowed" : "hover:bg-gray-50 cursor-pointer"}`}
        >
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-gray-700">⚙️ Paramètres</span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              engine === "whisper" ? "bg-blue-100 text-blue-700"     :
              engine === "vosk"    ? "bg-purple-100 text-purple-700"  :
              "bg-amber-100 text-amber-700"
            }`}>{engineLabel[engine]}</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 font-medium flex items-center gap-1">
              <Users className="w-3 h-3" />{nbLocuteurs === 1 ? "1 voix" : "2 voix"}
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 font-medium">
              ⏱ {chunkMs / 1000}s
            </span>
          </div>
          {showConfig ? <ChevronUp className="w-4 h-4 text-gray-400 flex-shrink-0" /> : <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />}
        </button>

        {showConfig && (
          <div className="px-5 pb-5 pt-1 border-t border-gray-100 space-y-5">

            {/* Moteur */}
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Moteur de reconnaissance</p>
              <div className="grid grid-cols-3 gap-2">
                {(["whisper", "vosk", "gladia"] as Engine[]).map(e => (
                  <button key={e} onClick={() => setEngine(e)}
                    className={`flex flex-col items-start p-3 rounded-xl border-2 transition-all text-left ${
                      engine === e
                        ? e === "whisper" ? "border-blue-400 bg-blue-50"
                        : e === "vosk"    ? "border-purple-400 bg-purple-50"
                        :                  "border-amber-400 bg-amber-50"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                  >
                    <span className="text-base mb-0.5">{e === "whisper" ? "🎯" : e === "vosk" ? "⚡" : "☁️"}</span>
                    <span className="text-sm font-semibold text-gray-800 capitalize">{e}</span>
                    <span className="text-xs text-gray-500 mt-0.5">
                      {e === "whisper" ? "Meilleure précision" : e === "vosk" ? "Réponse plus rapide" : "Service cloud"}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Avertissement Gladia */}
            {isGladia && (
              <div className="flex items-start gap-2.5 px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl">
                <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-700 leading-relaxed">
                  <span className="font-semibold">Service cloud</span> — l'audio est envoyé à l'API externe Gladia.
                  Quota : 10 h/mois gratuit. Ne pas utiliser pour des données strictement confidentielles.
                </p>
              </div>
            )}

            {/* Locuteurs */}
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Nombre de personnes</p>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={() => setNbLocuteurs(1)} disabled={isGladia}
                  className={`flex flex-col items-start p-3 rounded-xl border-2 transition-all text-left disabled:opacity-40 disabled:cursor-not-allowed ${
                    nbLocuteurs === 1 && !isGladia ? "border-green-400 bg-green-50" : "border-gray-200 bg-white hover:border-gray-300"
                  }`}
                >
                  <span className="text-base mb-0.5">🧑‍⚕️</span>
                  <span className="text-sm font-semibold text-gray-800">Seul</span>
                  <span className="text-xs text-gray-500 mt-0.5">Dictée, compte-rendu</span>
                </button>
                <button onClick={() => setNbLocuteurs(2)}
                  className={`flex flex-col items-start p-3 rounded-xl border-2 transition-all text-left ${
                    nbLocuteurs === 2 || isGladia ? "border-green-400 bg-green-50" : "border-gray-200 bg-white hover:border-gray-300"
                  }`}
                >
                  <span className="text-base mb-0.5">🧑‍⚕️👤</span>
                  <span className="text-sm font-semibold text-gray-800">Avec patient</span>
                  <span className="text-xs text-gray-500 mt-0.5">{isGladia ? "Détection automatique" : "Consultation, entretien"}</span>
                </button>
              </div>
            </div>

            {/* Durée chunk */}
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Durée chunk</p>
              <div className="flex gap-1 p-1 bg-gray-100 rounded-xl">
                {[1000, 2000, 3000].map(ms => (
                  <button key={ms} onClick={() => setChunkMs(ms)}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                      chunkMs === ms ? "bg-emerald-500 text-white shadow-sm" : "text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    {ms / 1000} s
                  </button>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>

      {/* ── Zone transcription ── */}
      <div className={`relative bg-white rounded-2xl border-2 transition-all duration-300 shadow-sm ${
        isRecording ? "border-red-400 shadow-red-100 shadow-lg" :
        isConnecting ? "border-yellow-300" : "border-gray-200"
      }`}>
        <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-gray-100">
          <div className="flex items-center gap-2">
            {isRecording && (<><Radio className="w-4 h-4 text-red-500 animate-pulse" /><span className="text-sm font-semibold text-red-500">En direct — {formatTime(recordingTime)}</span></>)}
            {isConnecting && !isRecording && (<><div className="w-4 h-4 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" /><span className="text-sm text-yellow-600">Connexion...</span></>)}
            {!isRecording && !isConnecting && <span className="text-sm font-medium text-gray-500">Transcription</span>}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setTranscript("")} disabled={!transcript}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed">
              <Trash2 className="w-3.5 h-3.5" /> Effacer
            </button>
            <button onClick={saveRecording} disabled={!transcript.trim() || isRecording}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all disabled:opacity-40 disabled:cursor-not-allowed font-medium">
              <Save className="w-3.5 h-3.5" /> Sauvegarder
            </button>
          </div>
        </div>

        <div className="px-5 py-4 min-h-[320px] max-h-[480px] overflow-y-auto">
          {transcript ? (
            <p className="text-gray-900 text-base leading-relaxed whitespace-pre-wrap">
              {transcript}
              {isRecording && <span className="inline-block w-0.5 h-5 bg-red-400 ml-1 animate-pulse align-middle" />}
            </p>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <Mic className="w-8 h-8 text-gray-300" />
              </div>
              <p className="text-gray-400 font-medium">{isConnecting ? "Connexion en cours..." : "Appuyez sur le bouton pour commencer"}</p>
              <p className="text-gray-300 text-sm mt-1">Le texte apparaîtra toutes les {chunkMs / 1000} secondes</p>
            </div>
          )}
        </div>

        {isRecording && (
          <div className="px-5 pb-4 flex items-end gap-0.5 h-10">
            {[...Array(30)].map((_, i) => (
              <div key={i} className="flex-1 bg-red-400 rounded-sm opacity-60"
                style={{ height: `${Math.random() * 24 + 6}px`, animation: `waveBar ${0.4 + Math.random() * 0.4}s ${i * 0.03}s ease-in-out infinite alternate` }} />
            ))}
          </div>
        )}
      </div>

      {/* ── Bouton principal ── */}
      <div className="flex justify-center">
        <button onClick={isRecording ? stopRecording : startRecording} disabled={isConnecting}
          className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold text-base transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
            isRecording ? "bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-200"
                        : "bg-gray-900 hover:bg-gray-700 text-white shadow-lg shadow-gray-200"
          }`}
        >
          {isRecording ? (<><Square className="w-5 h-5 fill-white" /><span>Arrêter — {formatTime(recordingTime)}</span><span className="w-2 h-2 bg-white rounded-full animate-pulse" /></>)
                       : (<><Mic className="w-5 h-5" /><span>{isConnecting ? "Connexion..." : "Démarrer la transcription"}</span></>)}
        </button>
      </div>

      {/* ── Stats + Log réseau ── */}
      <div className="bg-gray-900 rounded-2xl overflow-hidden border border-gray-800">

        {/* Barre stats */}
        <div className="flex items-center gap-5 px-4 py-2.5 border-b border-gray-800 flex-wrap">
          {[
            { label: "Envoyés",  value: statSent },
            { label: "Traités",  value: statDone },
            { label: "En file",  value: wsQueueRef.current.length + (wsBusyRef.current ? 1 : 0) },
            { label: "RTT",      value: statRtt != null ? `${statRtt}ms` : "—" },
          ].map(s => (
            <span key={s.label} className="text-xs text-gray-500 font-mono">
              {s.label}: <b className="text-gray-200 font-normal">{s.value}</b>
            </span>
          ))}
          <button onClick={() => setShowLog(o => !o)}
            className="ml-auto text-xs text-gray-500 hover:text-gray-300 font-mono uppercase tracking-wide flex items-center gap-1">
            Log réseau
            {showLog ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>

        {/* Log */}
        {showLog && (
          <>
            <div ref={logBoxRef}
              className="h-40 overflow-y-auto px-3 py-2 font-mono text-xs leading-relaxed"
              style={{ background: "#080808" }}
            >
              {logs.length === 0 && <span className="text-gray-600 italic">Aucune activité</span>}
              {logs.map(l => (
                <div key={l.id} className="flex gap-2 py-0.5 border-b border-gray-900">
                  <span className="text-gray-600 flex-shrink-0 w-[72px]">{l.ts}</span>
                  <span className={`flex-shrink-0 w-10 uppercase font-semibold text-[10px] ${tagColor[l.tag]}`}>{l.tag}</span>
                  <span className="text-gray-400 break-all">{l.msg}</span>
                </div>
              ))}
            </div>
            <div className="flex justify-end px-3 py-1.5 border-t border-gray-800">
              <button onClick={() => setLogs([])}
                className="text-xs text-gray-600 hover:text-gray-400 font-mono uppercase tracking-wide">
                effacer
              </button>
            </div>
          </>
        )}
      </div>

      {/* ── Aide ── */}
      <div className="bg-blue-50 rounded-xl p-5 border border-blue-100">
        <p className="text-sm font-semibold text-blue-800 mb-2">💡 Comment utiliser</p>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Choisissez le moteur et la durée chunk dans <strong>Paramètres</strong> avant de démarrer</li>
          <li>• Le log réseau (en bas) montre les échanges WebSocket en temps réel pour le débogage</li>
          <li>• Si vous voyez des <code className="bg-blue-100 px-1 rounded text-xs">WS onerror</code>, vérifiez que <code className="bg-blue-100 px-1 rounded text-xs">python api_websocket.py</code> est lancé</li>
        </ul>
      </div>

      <style>{`
        @keyframes waveBar { from { transform: scaleY(0.3); } to { transform: scaleY(1); } }
      `}</style>
    </div>
  );
}
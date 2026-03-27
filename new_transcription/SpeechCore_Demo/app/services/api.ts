/**
 * api.ts — Services pour communiquer avec les APIs Python
 *
 * PORT 8000 → api-websocket  (WebSocket /ws/transcribe + REST /extract Ollama via host)
 * PORT 8001 → api-completion (REST /extract — Ollama via conteneur Docker)
 */

const API_WEBSOCKET_URL  = "http://localhost:8000";
const API_COMPLETION_URL = "http://localhost:8001";

// ══════════════════════════════════════════════════════════
// TYPES
// ══════════════════════════════════════════════════════════

export interface FormField {
  name:          string;
  label:         string;
  type:          string;
  required:      boolean;
  semantic_hint?: string;
}

export interface FormSchema {
  fields: FormField[];
}

export interface TranscriptionResponse {
  transcription_complete:       string;
  transcription_avec_locuteurs: string;
  statistiques: {
    nombre_mots:      number;
    nombre_locuteurs: number;
    nombre_segments:  number;
    langue_detectee:  string;
  };
  locution_separee: Array<{ locuteur: string; texte: string }>;
}

export interface ExtractResponse {
  success: boolean;
  data:    Record<string, string>;
}

// ══════════════════════════════════════════════════════════
// SANTÉ DES SERVEURS
// ══════════════════════════════════════════════════════════

export const checkServerHealth = async (): Promise<void> => {
  const response = await fetch(`${API_WEBSOCKET_URL}/`, {
    method: "GET",
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error("api-websocket inaccessible (port 8000)");
};

export const checkCompletionHealth = async (): Promise<void> => {
  const response = await fetch(`${API_COMPLETION_URL}/`, {
    method: "GET",
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error("api-completion inaccessible (port 8001)");
};

// ══════════════════════════════════════════════════════════
// TRANSCRIPTION FICHIER COMPLET (port 8000)
// ══════════════════════════════════════════════════════════

export const transcribeAudio = async (
  audioBlob: Blob,
  engine: "vosk" | "whisper" | "gladia" = "whisper"
): Promise<TranscriptionResponse> => {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.wav");

  const response = await fetch(`${API_WEBSOCKET_URL}/${engine}`, {
    method: "POST",
    body:   formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Erreur ${response.status}`);
  }

  return response.json();
};

// ══════════════════════════════════════════════════════════
// EXTRACTION FORMULAIRE — PORT 8001 (api-completion, Docker)
// Format attendu par api-completion : { form: { fields }, text }
// ══════════════════════════════════════════════════════════

export const extractFormData = async (
  form:       FormSchema,
  transcript: string
): Promise<ExtractResponse> => {
  const response = await fetch(`${API_COMPLETION_URL}/extract`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      form: { fields: form.fields },   // ← format api-completion
      text: transcript,                // ← "text" et non "transcript"
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      error.detail ||
      "Erreur extraction — vérifiez que api-completion (port 8001) et Ollama sont lancés"
    );
  }

  return response.json();
};
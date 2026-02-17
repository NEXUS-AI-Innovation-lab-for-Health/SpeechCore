"""
Service LLM — Communication avec Ollama/Mistral
"""
import os
import ollama
from typing import Optional

class LLMService:
    def __init__(self, host: str = None, model: str = None, check_connection: bool = True):
        """
        Initialise le service LLM.
        
        Args:
            host:             URL d'Ollama (défaut: depuis .env ou localhost:11434)
            model:            Nom du modèle (défaut: depuis .env ou mistral)
            check_connection: Si True, vérifie qu'Ollama est accessible au démarrage
        """
        self.host  = host  or os.getenv("OLLAMA_HOST",  "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral")
        
        # Configure le client
        ollama._client._base_url = self.host
        
        # Vérifier qu'Ollama est accessible
        if check_connection:
            self._check_connection()
    
    def _check_connection(self):
        """Vérifie que le modèle est disponible."""
        try:
            models_response = ollama.list()
            
            # La réponse peut être un dict avec 'models' ou directement une liste
            if isinstance(models_response, dict):
                models_list = models_response.get("models", [])
            else:
                models_list = models_response
            
            # Extraire les noms de modèles
            available = []
            for m in models_list:
                if isinstance(m, dict):
                    # Le nom peut être dans 'name' ou 'model'
                    name = m.get("name") or m.get("model", "")
                    available.append(name)
                else:
                    available.append(str(m))
            
            # Vérifier si le modèle existe (avec ou sans tag)
            model_base = self.model.split(":")[0]
            if not any(model_base in m for m in available):
                raise ConnectionError(
                    f"Le modèle '{self.model}' n'est pas disponible. "
                    f"Modèles disponibles: {', '.join(available)}"
                )
        except ConnectionError:
            raise  # Re-raise si c'est déjà une ConnectionError
        except Exception as e:
            raise ConnectionError(
                f"Impossible de se connecter à Ollama sur {self.host}. "
                f"Vérifiez qu'Ollama est lancé. Erreur: {e}"
            )
    
    def generate(self, prompt: str, user_input: str, temperature: float = 0.1) -> str:
        """
        Génère une réponse via Mistral.
        
        Args:
            prompt:      Prompt système (instructions)
            user_input:  Question de l'utilisateur
            temperature: Température pour la génération (0.1 = plus déterministe)
            
        Returns:
            Réponse générée (requête structurée)
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": user_input}
                ],
                options={
                    "temperature": temperature,
                    # "num_predict": 500  # Limite de tokens si besoin
                }
            )
            
            return response["message"]["content"].strip()
        
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la génération avec {self.model}: {e}")

    def generate_json(self, prompt: str, user_input: str, temperature: float = 0.1) -> dict:
        """
        Génère une réponse et la parse automatiquement en JSON.
        Nettoie les artefacts courants (backticks markdown, point-virgules, etc.)
        
        Args:
            prompt:      Prompt système (instructions)
            user_input:  Question de l'utilisateur
            temperature: Température pour la génération
            
        Returns:
            Dict Python parsé depuis le JSON généré
            
        Raises:
            ValueError: Si la réponse n'est pas du JSON valide après nettoyage
        """
        raw_response = self.generate(prompt, user_input, temperature)
        
        # Nettoyer la réponse
        cleaned = raw_response.strip()
        
        # Retirer les backticks markdown
        cleaned = cleaned.replace("```json", "").replace("```", "")
        
        # Retirer les point-virgules en fin de ligne
        cleaned = cleaned.rstrip(";")
        
        # Retirer espaces/retours à la ligne superflus
        cleaned = cleaned.strip()
        
        # Parser le JSON
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Impossible de parser la réponse comme JSON. "
                f"Erreur: {e}. Réponse brute: {raw_response}"
            )
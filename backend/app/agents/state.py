# =============================================================================
# EcoFibre Sync — État du graphe LangGraph (agents/state.py)
# =============================================================================
# Ce module définit la structure de données qui transite entre les nœuds
# (agents) du graphe LangGraph.
#
# Pourquoi un TypedDict plutôt qu'une dataclass ou un Pydantic model ?
#   LangGraph utilise des TypedDict pour l'état du graphe car :
#   - Ils sont sérialisables nativement (JSON, pickle)
#   - Ils supportent la fusion automatique d'états partiels (réducers)
#   - Ils sont légers (pas d'instanciation d'objet)
#   - Ils sont le format officiellement recommandé par LangGraph
#
# Flux de l'état dans le graphe :
#   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
#   │  Extracteur  │───▶│  RAG / Retriever │───▶│  RSE / Générateur │
#   └──────────────┘    └──────────────┘    └──────────────┘
#
#   L'état est enrichi à chaque étape :
#   1. Extracteur : remplit `equipment`, `error_codes`, `error_description`
#   2. RAG        : remplit `retrieved_context` (chunks pgvector)
#   3. RSE        : remplit `procedure`, `rse_score`
# =============================================================================

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """
    État partagé entre les agents du graphe de résolution de pannes.

    Chaque nœud du graphe lit et enrichit cet état. Les champs sont
    marqués `total=False` (tous optionnels) car chaque nœud ne remplit
    qu'un sous-ensemble des champs.

    Lifecycle :
      1. L'API crée l'état initial avec `raw_report`
      2. L'Extracteur analyse le texte brut et extrait les métadonnées
      3. Le RAG recherche les documents pertinents dans pgvector
      4. Le Générateur RSE produit la procédure finale
    """

    # ── Entrée initiale (fournie par l'API) ──────────────────────────────
    # Texte brut du technicien décrivant la panne sur le terrain.
    # Exemple : "Perte de signal sur PBO-12, LED ONT éteinte, client
    # sans connexion depuis 2h. Câble en façade semble endommagé."
    raw_report: str

    # Image optionnelle associée au rapport (base64 ou URL)
    image: str | None

    # ── Sortie de l'Agent Extracteur ─────────────────────────────────────
    # L'extracteur utilise le LLM pour parser le texte non structuré
    # et en extraire des informations normalisées.

    # Équipement identifié (ex: "ONT", "PBO-12", "Jarretière optique")
    equipment: str

    # Codes d'erreur détectés ou déduits (ex: ["LOS", "PON_DOWN"])
    # Liste car un rapport peut mentionner plusieurs erreurs.
    error_codes: list[str]

    # Description synthétique de l'erreur, reformulée par le LLM
    # pour être exploitable par les agents suivants.
    error_description: str

    # ── Sortie de l'Agent RAG (Retriever) ────────────────────────────────
    # Contexte documentaire récupéré depuis pgvector.
    # Contient les passages les plus pertinents des rapports d'intervention
    # précédents et des normes techniques Bouygues Telecom.
    #
    # Format : texte concaténé des top-K chunks les plus similaires.
    # On concatène plutôt que de passer une liste pour simplifier le
    # prompt du Générateur (un seul bloc de contexte).
    retrieved_context: str

    # ── Sortie de l'Agent RSE (Générateur) ───────────────────────────────
    # Procédure technique finale, rédigée avec une logique RSE :
    # Nettoyage → Réparation → Remplacement (en dernier recours).
    #
    # Formatée en Markdown pour un affichage riche dans le frontend.
    procedure: str

    # Score RSE (0-100) évaluant le niveau d'éco-responsabilité
    # de la procédure proposée. Plus le score est élevé, plus la
    # solution privilégie la réparation sur le remplacement.
    rse_score: int

    # ── Métadonnées de traçabilité ───────────────────────────────────────
    # Erreur éventuelle survenue pendant le traitement.
    # Si non vide, le graphe s'arrête et retourne l'erreur au client.
    error: str

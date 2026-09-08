# =============================================================================
# EcoFibre Sync — Graphe LangGraph de résolution de pannes (agents/graph.py)
# =============================================================================
# Ce module assemble les trois agents (Extracteur, RAG, RSE) en un graphe
# d'exécution séquentiel avec LangGraph.
#
# Architecture du graphe :
#
#   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
#   │  Extracteur  │────▶│     RAG     │────▶│ RSE Générateur │
#   │  (parsing)   │     │ (retrieval) │     │  (procédure)  │
#   └─────────────┘     └─────────────┘     └─────────────┘
#         ▲                                         │
#         │                                         ▼
#     raw_report                              procedure +
#     (entrée)                                rse_score
#                                             (sortie)
#
# Pourquoi LangGraph plutôt qu'un simple enchaînement de fonctions ?
#   1. Visualisation : le graphe est inspectable et debuggable
#   2. Persistence : l'état peut être sauvegardé entre les nœuds
#   3. Branching : on pourra ajouter des conditions (ex: si score < 30,
#      relancer le générateur avec un prompt plus strict)
#   4. Streaming : LangGraph supporte le streaming token par token
#   5. Observabilité : intégration native avec Langfuse/LangSmith
#
# Exécution :
#   from app.agents.graph import resolution_graph
#   result = await resolution_graph.ainvoke({"raw_report": "..."})
# =============================================================================

import logging

from langgraph.graph import StateGraph, START, END

from app.agents.state import GraphState
from app.agents.nodes import extractor_node, retriever_node, generator_node

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe
# ─────────────────────────────────────────────────────────────────────────────
# StateGraph prend en paramètre le type de l'état (GraphState).
# Chaque nœud est une fonction async qui reçoit l'état et retourne
# un dictionnaire partiel qui sera fusionné (merged) dans l'état global.
#
# La fusion est simple : les clés retournées par un nœud écrasent
# les valeurs existantes dans l'état. Les clés non retournées sont
# préservées. Cela permet à chaque agent de ne modifier que "sa" partie.
# ─────────────────────────────────────────────────────────────────────────────

def build_resolution_graph() -> StateGraph:
    """
    Construit et compile le graphe de résolution de pannes.

    Returns:
        Le graphe compilé, prêt à être invoqué via .ainvoke().

    Structure :
        START → extractor → retriever → generator → END
    """
    logger.info("🏗️ Construction du graphe de résolution LangGraph...")

    # Créer le graphe avec le type d'état
    graph = StateGraph(GraphState)

    # ── Ajout des nœuds ──────────────────────────────────────────────────
    # Chaque nœud est identifié par un nom (string) et associé à une
    # fonction async. Le nom est utilisé pour les arêtes et le debugging.

    graph.add_node("extractor", extractor_node)
    # L'Extracteur parse le texte brut du technicien et produit :
    # equipment, error_codes, error_description

    graph.add_node("retriever", retriever_node)
    # Le Retriever RAG cherche dans pgvector les documents pertinents
    # et produit : retrieved_context

    graph.add_node("generator", generator_node)
    # Le Générateur RSE rédige la procédure technique éco-responsable
    # et produit : procedure, rse_score

    # ── Définition des arêtes (transitions) ──────────────────────────────
    # Pipeline séquentiel : chaque agent dépend du résultat du précédent.
    #
    # START → extractor : l'extraction est toujours la première étape
    # extractor → retriever : on a besoin des données extraites pour
    #   construire la requête de recherche vectorielle
    # retriever → generator : le générateur a besoin du contexte RAG
    #   pour produire une procédure basée sur les normes Bouygues
    # generator → END : la procédure est le résultat final

    graph.add_edge(START, "extractor")
    graph.add_edge("extractor", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    # ── Compilation ──────────────────────────────────────────────────────
    # La compilation vérifie la cohérence du graphe (pas de nœud orphelin,
    # pas de cycle non-voulu) et produit un objet exécutable.
    compiled_graph = graph.compile()

    logger.info(
        "✅ Graphe de résolution compilé : "
        "extractor → retriever → generator"
    )

    return compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Instance globale du graphe compilé
# ─────────────────────────────────────────────────────────────────────────────
# On compile le graphe une seule fois au chargement du module.
# L'objet compilé est thread-safe et peut être invoqué concurremment
# par plusieurs requêtes FastAPI.
# ─────────────────────────────────────────────────────────────────────────────
resolution_graph = build_resolution_graph()

# =============================================================================
# EcoFibre Sync — Nœuds du graphe LangGraph (agents/nodes.py)
# =============================================================================
# Ce module implémente les trois agents autonomes du pipeline de résolution
# de pannes fibre optique. Chaque agent est un nœud du graphe LangGraph
# qui lit l'état partagé, effectue son traitement, et enrichit l'état.
#
# Agents :
#   1. Agent Extracteur — Parse le texte brut du technicien
#   2. Agent RAG (Retriever) — Recherche le contexte dans pgvector
#   3. Agent RSE (Générateur) — Rédige la procédure éco-responsable
#
# Tous les agents utilisent ChatGoogleGenerativeAI (Gemini) comme LLM
# pour rester cohérent avec l'écosystème Google déjà utilisé pour les
# embeddings. LangChain permet de swapper vers un autre provider en
# changeant une seule ligne.
#
# Philosophie RSE (Numérique Responsable) :
#   Le prompt système de l'Agent RSE impose une hiérarchie stricte :
#     1. Nettoyage (coût carbone ≈ 0)
#     2. Réparation / Soudure (coût carbone faible)
#     3. Remplacement partiel (coût carbone moyen)
#     4. Remplacement complet (coût carbone élevé — dernier recours)
# =============================================================================

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.agents.state import GraphState
from app.agents.prompts import ECOFIBRE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Factory du modèle LLM
# ─────────────────────────────────────────────────────────────────────────────
# Centralise la création du ChatModel pour que tous les agents utilisent
# la même configuration (clé API, température, etc.).
#
# temperature=0.2 : on veut des réponses déterministes et factuelles
# pour des procédures techniques. Une température trop haute (>0.7)
# risquerait de "halluciner" des étapes de réparation incorrectes.
# ─────────────────────────────────────────────────────────────────────────────
def _get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Crée une instance ChatGoogleGenerativeAI configurée."""
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_CHAT_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=2048,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Agent Extracteur — Analyse structurée du rapport de panne
# ─────────────────────────────────────────────────────────────────────────────
# Cet agent transforme le texte libre du technicien en données structurées.
# C'est l'étape la plus critique : si l'extraction est mauvaise, tout le
# pipeline en aval sera dégradé (garbage in, garbage out).
#
# Le prompt force une sortie JSON pour garantir un parsing fiable.
# On utilise des exemples in-context (few-shot) pour guider le modèle
# vers le format attendu.
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTOR_SYSTEM_PROMPT = """\
Tu es un expert en diagnostic de pannes sur les réseaux fibre optique FTTH.
Tu travailles pour la Direction Déploiement et Exploitation de Bouygues Telecom.

Ton rôle : analyser le rapport de terrain d'un technicien (texte et/ou photo) et en extraire
les informations clés de manière structurée. Si une image est fournie, cherche à y lire le modèle de l'équipement (ex: étiquette arrière) ou l'état des voyants lumineux.

Tu DOIS répondre UNIQUEMENT avec un objet JSON valide contenant :
{
    "equipment": "Nom de l'équipement principal concerné (ONT, OLT, PBO, PM, jarretière, câble, etc.)",
    "error_codes": ["Liste des codes d'erreur ou symptômes identifiés (LOS, PON_DOWN, SIGNAL_FAIBLE, COUPURE, etc.)"],
    "error_description": "Description synthétique et technique du problème en 1-2 phrases."
}

Règles :
- Si le technicien ne mentionne pas de code d'erreur explicite, DÉDUIS les codes
  à partir des symptômes décrits (ex: "pas de lumière sur l'ONT" → ["LOS"]).
- Utilise la terminologie standard FTTH (NRO, SRO, PM, PBO, ONT, OLT, ONU, etc.).
- Sois concis et factuel dans error_description.
- Ne retourne RIEN d'autre que le JSON. Pas de commentaire, pas de markdown.

Exemple d'entrée :
"Le client signale une coupure internet depuis ce matin. Sur place, l'ONT
Nokia G-010G-A affiche un voyant LOS rouge fixe. J'ai vérifié le PBO en
pied d'immeuble, connecteur propre. Le signal semble se perdre entre le
PM et le PBO, possible coupure sur le tronçon aérien."

Exemple de sortie :
{"equipment": "ONT Nokia G-010G-A", "error_codes": ["LOS", "COUPURE_FIBRE"], "error_description": "Perte de signal optique (LOS) sur l'ONT client. Coupure suspectée sur le tronçon aérien entre PM et PBO. Connecteur PBO vérifié OK."}
"""


async def extractor_node(state: GraphState) -> dict:
    """
    Agent Extracteur : parse le texte brut du technicien.

    Entrée (state) : raw_report
    Sortie (state) : equipment, error_codes, error_description
    """
    raw_report = state.get("raw_report", "")
    image_data = state.get("image")

    if not raw_report.strip() and not image_data:
        logger.warning("⚠️ Rapport vide reçu par l'Extracteur")
        return {
            "equipment": "Inconnu",
            "error_codes": [],
            "error_description": "Rapport vide — impossible d'analyser",
            "error": "Le rapport de panne est vide",
        }

    logger.info(f"🔍 Agent Extracteur : analyse de {len(raw_report)} caractères...")

    try:
        llm = _get_llm(temperature=0.1)  # Température basse pour du JSON fiable
        
        # Préparation du contenu de la requête (multimodal support)
        human_content = []
        if raw_report.strip():
            human_content.append({"type": "text", "text": raw_report})
        else:
            human_content.append({"type": "text", "text": "Analyse l'image jointe."})
            
        if image_data:
            human_content.append({
                "type": "image_url", 
                "image_url": {"url": image_data}
            })

        messages = [
            SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content.strip()

        # Nettoyage : le LLM peut entourer le JSON de ```json ... ```
        if response_text.startswith("```"):
            # Retirer le bloc de code markdown
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        # Parser le JSON
        extracted = json.loads(response_text)

        result = {
            "equipment": extracted.get("equipment", "Inconnu"),
            "error_codes": extracted.get("error_codes", []),
            "error_description": extracted.get("error_description", ""),
        }

        logger.info(
            f"✅ Extraction terminée : équipement={result['equipment']}, "
            f"codes={result['error_codes']}"
        )

        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ Échec du parsing JSON de l'Extracteur : {e}")
        return {
            "equipment": "Inconnu",
            "error_codes": [],
            "error_description": raw_report[:200],
            "error": f"Échec de l'extraction structurée : {e}",
        }
    except Exception as e:
        logger.error(f"❌ Erreur de l'Agent Extracteur : {e}", exc_info=True)
        return {
            "equipment": "Inconnu",
            "error_codes": [],
            "error_description": raw_report[:200],
            "error": f"Erreur de l'Agent Extracteur : {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent RAG (Retriever) — Recherche de contexte dans pgvector
# ─────────────────────────────────────────────────────────────────────────────
# Cet agent construit une requête de recherche à partir des informations
# extraites et interroge le vector store pour récupérer les documents
# techniques les plus pertinents.
#
# La requête est construite en combinant l'équipement, les codes d'erreur
# et la description — ce qui produit un embedding de requête plus riche
# qu'un simple mot-clé, et donc de meilleurs résultats de similarité.
#
# Si aucun document n'est trouvé dans pgvector (base vide au début),
# un contexte générique "normes Bouygues Telecom" est fourni pour que
# le Générateur puisse quand même produire une réponse utile.
# ─────────────────────────────────────────────────────────────────────────────
async def retriever_node(state: GraphState) -> dict:
    """
    Agent RAG : recherche les documents pertinents dans pgvector.

    Entrée (state) : equipment, error_codes, error_description
    Sortie (state) : retrieved_context
    """
    equipment = state.get("equipment", "Inconnu")
    error_codes = state.get("error_codes", [])
    error_description = state.get("error_description", "")

    # Construire la requête de recherche sémantique
    # On combine tous les éléments pour maximiser la pertinence
    search_query = (
        f"Équipement: {equipment}. "
        f"Codes erreur: {', '.join(error_codes) if error_codes else 'N/A'}. "
        f"Problème: {error_description}"
    )

    logger.info(f"🔍 Agent RAG : recherche pour '{search_query[:80]}...'")

    try:
        # Import ici pour éviter les imports circulaires
        from app.services.vectorstore import VectorStoreService
        from app.core.database import AsyncSessionLocal

        vectorstore = VectorStoreService()

        # Recherche dans pgvector avec une session async
        async with AsyncSessionLocal() as session:
            chunks = await vectorstore.similarity_search(
                session=session,
                query=search_query,
                top_k=5,
            )

        if chunks:
            # Concaténer les chunks pertinents avec des séparateurs
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"--- Document {i} ---\n{chunk.content}"
                )
            retrieved_context = "\n\n".join(context_parts)

            logger.info(
                f"✅ RAG : {len(chunks)} documents trouvés dans pgvector"
            )
        else:
            # Aucun document en base → contexte générique pour ne pas
            # bloquer le pipeline. Ce cas est normal au début quand la
            # base documentaire n'est pas encore alimentée.
            logger.info(
                "ℹ️ RAG : base pgvector vide, utilisation du contexte générique"
            )
            retrieved_context = _get_fallback_context()

        return {"retrieved_context": retrieved_context}

    except Exception as e:
        logger.error(f"❌ Erreur de l'Agent RAG : {e}", exc_info=True)
        # En cas d'erreur DB, on utilise le contexte générique pour ne
        # pas bloquer le pipeline. Le Générateur pourra quand même
        # produire une recommandation (moins précise).
        return {"retrieved_context": _get_fallback_context()}


def _get_fallback_context() -> str:
    """
    Contexte technique générique utilisé quand pgvector est vide.

    Ce texte contient les normes et procédures standard Bouygues Telecom
    pour la maintenance FTTH. Il sert de "base de connaissances minimale"
    pour que l'Agent RSE puisse produire des recommandations utiles même
    sans documents indexés.
    """
    return """\
--- Normes et Procédures Standard — Bouygues Telecom FTTH ---

PROCÉDURES DE DIAGNOSTIC :
1. Vérifier l'état des voyants ONT (PON, LOS, POWER)
2. Contrôler la puissance optique reçue (seuil : -27 dBm min)
3. Effectuer un test OTDR si perte > 0.5 dB/km
4. Inspecter visuellement les connecteurs (SC/APC vert, SC/UPC bleu)

SEUILS DE RÉFÉRENCE :
- Atténuation fibre G.657.A2 : 0.35 dB/km @ 1310nm
- Perte par soudure fusion : < 0.1 dB
- Perte par connecteur : < 0.5 dB
- Réflectance connecteur : < -40 dB
- Rayon de courbure minimal G.657.A2 : 15 mm

HIÉRARCHIE DE RÉSOLUTION (Politique RSE Bouygues Telecom) :
1. NETTOYAGE : connecteurs, coupleurs, tiroirs optiques
   → Résout ~40% des problèmes de signal (coût carbone : ~0)
2. RÉPARATION : soudure fusion, repositionnement câble
   → Résout ~35% des problèmes (coût carbone : faible)
3. REMPLACEMENT PARTIEL : jarretière, pigtail, manchon
   → Résout ~20% des problèmes (coût carbone : moyen)
4. REMPLACEMENT COMPLET : câble, PBO, ONT
   → Dernier recours, ~5% des cas (coût carbone : élevé)

MATÉRIEL RÉUTILISABLE :
- ONT réconditionné : filière de reconditionnement interne
- Jarretières optiques : recyclage des connecteurs SC/APC
- PBO : boîtiers nettoyés et recertifiés disponibles en stock
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent RSE (Générateur) — Procédure technique éco-responsable
# ─────────────────────────────────────────────────────────────────────────────
# Cet agent est le cœur de la valeur ajoutée "Numérique Responsable".
# Il rédige une procédure technique complète en imposant systématiquement
# la hiérarchie : Nettoyage → Réparation → Remplacement.
# ─────────────────────────────────────────────────────────────────────────────

async def generator_node(state: GraphState) -> dict:
    """
    Agent RSE (Générateur) : rédige la procédure éco-responsable.

    Entrée (state) : equipment, error_codes, error_description, retrieved_context
    Sortie (state) : procedure, rse_score
    """
    equipment = state.get("equipment", "Inconnu")
    error_codes = state.get("error_codes", [])
    error_description = state.get("error_description", "")
    retrieved_context = state.get("retrieved_context", "")
    raw_report = state.get("raw_report", "").strip()
    image_data = state.get("image")

    logger.info(
        f"📝 Agent RSE : génération de la procédure pour {equipment} "
        f"(codes: {error_codes})"
    )

    # Enrichir le rapport brut avec les éléments structurés extraits
    if raw_report:
        report_text = f"RAPPORT ORIGINAL :\n{raw_report}"
    else:
        report_text = "RAPPORT ORIGINAL :\nAnalyse visuelle requise. Le technicien n'a fourni aucun texte, base-toi uniquement sur l'image."

    enriched_report = f"""\
{report_text}

ANALYSE STRUCTURÉE :
- Équipement : {equipment}
- Codes erreur : {', '.join(error_codes) if error_codes else 'Aucun code spécifique'}
- Description : {error_description}
"""

    formatted_prompt = ECOFIBRE_SYSTEM_PROMPT.format(
        context=retrieved_context,
        report=enriched_report
    )

    user_content = []
    
    if image_data:
        if not image_data.startswith("data:image"):
            image_data = f"data:image/jpeg;base64,{image_data}"
        user_content.append({
            "type": "image_url", 
            "image_url": {"url": image_data}
        })
        # Note: formatted_prompt contains the enriched_report text, which we can pass as the text block
        user_content.insert(0, {"type": "text", "text": formatted_prompt})
    else:
        user_content.append({"type": "text", "text": formatted_prompt})

    try:
        llm = _get_llm(temperature=0.3)  # Légèrement plus créatif pour la rédaction
        messages = [
            HumanMessage(content=user_content),
        ]

        response = await llm.ainvoke(messages)
        procedure = response.content.strip()

        # Extraire le score RSE du texte généré
        rse_score = _extract_rse_score(procedure)

        logger.info(
            f"✅ Procédure générée : {len(procedure)} caractères, "
            f"score RSE = {rse_score}/100"
        )

        return {
            "procedure": procedure,
            "rse_score": rse_score,
        }

    except Exception as e:
        logger.error(f"❌ Erreur de l'Agent RSE : {e}", exc_info=True)
        return {
            "procedure": (
                f"## ⚠️ Erreur de génération\n\n"
                f"L'agent RSE n'a pas pu générer la procédure.\n"
                f"Erreur : {e}\n\n"
                f"**Fallback** : appliquer la procédure standard de diagnostic :\n"
                f"1. Nettoyage des connecteurs\n"
                f"2. Mesure de puissance optique\n"
                f"3. Test OTDR si nécessaire\n"
                f"4. Escalade vers N2 si non résolu"
            ),
            "rse_score": 50,
            "error": f"Erreur de l'Agent RSE : {e}",
        }


def _extract_rse_score(procedure_text: str) -> int:
    """
    Extrait le score RSE du texte de la procédure.

    Cherche un pattern "Score RSE : XX" ou "Score RSE** : XX" dans le
    texte généré par le LLM. Retourne 50 par défaut si non trouvé.
    """
    import re

    # Patterns possibles générés par le LLM
    patterns = [
        r"\*?\*?Score RSE\*?\*?\s*[:：]\s*(\d+)",
        r"score.?rse\s*[:：]\s*(\d+)",
        r"RSE\s*[:：]\s*(\d+)\s*/\s*100",
        r"(\d+)\s*/\s*100",
    ]

    for pattern in patterns:
        match = re.search(pattern, procedure_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Clamp entre 0 et 100
            return max(0, min(100, score))

    logger.warning("⚠️ Score RSE non trouvé dans la procédure, défaut = 50")
    return 50

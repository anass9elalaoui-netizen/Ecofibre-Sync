# =============================================================================
# EcoFibre Sync — Router de résolution de pannes (api/v1/resolution.py)
# =============================================================================
# Ce module expose l'endpoint REST pour déclencher le pipeline LangGraph
# de résolution de pannes fibre optique.
#
# Endpoint principal :
#   POST /api/v1/resolution/solve
#     → Reçoit un texte de technicien en entrée
#     → Exécute le workflow LangGraph (Extraction → RAG → Génération)
#     → Retourne la procédure éco-responsable avec le score RSE
#
# Pourquoi synchrone (pas de Celery) ?
#   Contrairement à l'ingestion de documents (longue, >30s), la résolution
#   de pannes est rapide (~5-15s pour 3 appels LLM). On peut donc
#   attendre la réponse dans le cycle HTTP. Si la latence augmente en
#   production, on pourra basculer vers un mode async + SSE streaming.
# =============================================================================

import logging
import time

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.graph import resolution_graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic pour l'API de résolution
# ─────────────────────────────────────────────────────────────────────────────

class ResolutionRequest(BaseModel):
    """Requête de résolution de panne envoyée par le technicien."""

    report: str = Field(
        default="",
        max_length=10000,
        description=(
            "Texte brut décrivant la panne. Peut être un rapport de terrain, "
            "un message du technicien, ou une description du client."
        ),
        json_schema_extra={
            "example": (
                "Client sans internet depuis ce matin. ONT Nokia G-010G-A "
                "avec voyant LOS rouge fixe. PBO en pied d'immeuble vérifié, "
                "connecteur propre. Signal perdu entre PM-045 et PBO-12, "
                "possible coupure sur le tronçon aérien de 340m."
            )
        },
    )

    image: str | None = Field(
        default=None,
        description="Image en base64 (optionnelle) fournie par le technicien.",
    )


class ResolutionResponse(BaseModel):
    """Réponse complète du pipeline de résolution LangGraph."""

    # ── Données extraites par l'Agent Extracteur ─────────────────────────
    equipment: str = Field(
        ...,
        description="Équipement principal identifié",
        json_schema_extra={"example": "ONT Nokia G-010G-A"},
    )

    error_codes: list[str] = Field(
        default_factory=list,
        description="Codes d'erreur détectés ou déduits",
        json_schema_extra={"example": ["LOS", "COUPURE_FIBRE"]},
    )

    error_description: str = Field(
        ...,
        description="Description synthétique du problème",
    )

    # ── Contexte RAG ─────────────────────────────────────────────────────
    context_used: bool = Field(
        ...,
        description="Indique si du contexte documentaire a été trouvé dans pgvector",
    )

    # ── Procédure RSE générée ────────────────────────────────────────────
    procedure: str = Field(
        ...,
        description="Procédure de résolution complète en Markdown",
    )

    rse_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score RSE (0-100). Plus élevé = plus éco-responsable",
        json_schema_extra={"example": 85},
    )

    # ── Métadonnées ──────────────────────────────────────────────────────
    processing_time_seconds: float = Field(
        ...,
        description="Temps de traitement total du pipeline en secondes",
        json_schema_extra={"example": 8.42},
    )

    pipeline_steps: list[str] = Field(
        default_factory=lambda: ["extractor", "retriever", "generator"],
        description="Étapes du pipeline exécutées",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration du router
# ─────────────────────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/api/v1/resolution",
    tags=["Résolution de Pannes"],
)


from langfuse.callback import CallbackHandler
from app.core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/resolution/solve — Résolution de panne via LangGraph
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/solve",
    response_model=ResolutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Résoudre une panne fibre via les agents IA",
    response_description="Procédure de résolution éco-responsable",
)
async def solve_issue(request: ResolutionRequest):
    """... docstring ..."""
    logger.info(
        f"🚀 Résolution de panne demandée : "
        f"'{request.report[:80]}...'"
    )

    start_time = time.time()

    try:
        # ── Exécution du graphe LangGraph ────────────────────────────────
        result = await resolution_graph.ainvoke(
            {"raw_report": request.report, "image": request.image}
        )

        processing_time = round(time.time() - start_time, 2)

        # ── Vérification d'erreur dans l'état ────────────────────────────
        if result.get("error"):
            logger.warning(
                f"⚠️ Pipeline terminé avec avertissement : {result['error']}"
            )

        # ── Déterminer si du contexte réel a été utilisé ─────────────────
        retrieved_context = result.get("retrieved_context", "")
        context_used = bool(
            retrieved_context
            and "Normes et Procédures Standard" not in retrieved_context
        )

        # ── Construction de la réponse ───────────────────────────────────
        response = ResolutionResponse(
            equipment=result.get("equipment", "Inconnu"),
            error_codes=result.get("error_codes", []),
            error_description=result.get("error_description", ""),
            context_used=context_used,
            procedure=result.get("procedure", "Erreur : aucune procédure générée"),
            rse_score=result.get("rse_score", 50),
            processing_time_seconds=processing_time,
            pipeline_steps=["extractor", "retriever", "generator"],
        )

        logger.info(
            f"✅ Résolution terminée en {processing_time}s — "
            f"Score RSE : {response.rse_score}/100 — "
            f"Équipement : {response.equipment}"
        )

        return response

    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(
            f"❌ Erreur du pipeline de résolution après {processing_time}s : {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Erreur lors de l'exécution du pipeline de résolution",
                "detail": str(e),
                "processing_time_seconds": processing_time,
            },
        )

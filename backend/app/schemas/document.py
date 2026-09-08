# =============================================================================
# EcoFibre Sync — Schémas Pydantic pour les documents (schemas/document.py)
# =============================================================================
# Ces schémas définissent les contrats d'API pour le pipeline d'ingestion :
#
#   - Entrée : validation du fichier uploadé
#   - Sortie : réponse structurée avec task_id et statut
#
# Pourquoi séparer les schémas des modèles ORM ?
#   1. Les modèles ORM reflètent la structure de la DB (colonnes, relations)
#   2. Les schémas Pydantic reflètent le contrat de l'API (ce que le client
#      envoie et reçoit). Ils peuvent être plus restrictifs ou différents.
#   3. Cette séparation évite d'exposer des champs internes (ex: embeddings)
#      et permet de versionner l'API indépendamment du schéma DB.
#
# Convention de nommage :
#   - *Response : schéma retourné au client
#   - *Status   : schéma pour consulter l'état d'un traitement
# =============================================================================

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import ProcessingStatus


# ─────────────────────────────────────────────────────────────────────────────
# 1. Réponse à l'upload — POST /api/v1/ingest
# ─────────────────────────────────────────────────────────────────────────────
# Retournée immédiatement après l'upload, avant que le traitement ne commence.
# Le client utilise le document_id pour poller le statut ensuite.
#
# task_id : identifiant Celery de la tâche d'ingestion.
#   Permet de consulter le statut Celery directement si besoin,
#   mais en général on passe par document_id qui est plus stable.
# ─────────────────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    """Réponse immédiate après soumission d'un document à l'ingestion."""

    document_id: uuid.UUID = Field(
        ...,
        description="Identifiant unique du document créé en base",
        json_schema_extra={"example": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
    )

    task_id: str = Field(
        ...,
        description="Identifiant de la tâche Celery d'ingestion",
        json_schema_extra={"example": "c9d8e7f6-5a4b-3c2d-1e0f-a9b8c7d6e5f4"},
    )

    filename: str = Field(
        ...,
        description="Nom du fichier original uploadé",
        json_schema_extra={"example": "rapport_ftth_orleans_2024.pdf"},
    )

    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Statut initial du traitement",
    )

    message: str = Field(
        default="Document soumis pour traitement",
        description="Message informatif pour le client",
    )

    model_config = ConfigDict(
        # from_attributes=True permet de créer ce schéma directement
        # depuis un objet ORM SQLAlchemy (ex: IngestResponse.model_validate(doc))
        from_attributes=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statut du document — GET /api/v1/ingest/{document_id}/status
# ─────────────────────────────────────────────────────────────────────────────
# Permet au frontend de poller l'avancement du traitement.
# En production, on pourrait utiliser des WebSockets ou SSE (Server-Sent Events)
# pour du push plutôt que du polling, mais le polling reste plus simple
# et suffisant pour un MVP.
# ─────────────────────────────────────────────────────────────────────────────
class DocumentStatusResponse(BaseModel):
    """Statut détaillé du traitement d'un document."""

    document_id: uuid.UUID = Field(
        ...,
        description="Identifiant unique du document",
        validation_alias="id",
    )

    filename: str = Field(
        ...,
        description="Nom du fichier original",
    )

    status: ProcessingStatus = Field(
        ...,
        description="Statut actuel du traitement",
    )

    chunk_count: int = Field(
        default=0,
        description="Nombre de chunks vectorisés générés",
    )

    error_message: str | None = Field(
        default=None,
        description="Message d'erreur si le traitement a échoué",
    )

    created_at: datetime = Field(
        ...,
        description="Date d'upload du document (UTC)",
    )

    updated_at: datetime = Field(
        ...,
        description="Dernière mise à jour du statut (UTC)",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Détail d'un chunk — usage interne et debug
# ─────────────────────────────────────────────────────────────────────────────
# Ce schéma n'expose PAS l'embedding (vecteur de 768 floats) au client
# pour des raisons de performance (payload trop lourd) et de sécurité
# (les embeddings pourraient permettre une reconstruction partielle du texte).
# ─────────────────────────────────────────────────────────────────────────────
class DocumentChunkResponse(BaseModel):
    """Représentation d'un chunk sans son embedding vectoriel."""

    id: uuid.UUID = Field(..., description="Identifiant unique du chunk")
    document_id: uuid.UUID = Field(..., description="ID du document parent")
    content: str = Field(..., description="Texte brut du chunk")
    chunk_index: int = Field(..., description="Position dans le document")
    created_at: datetime = Field(..., description="Date de création")

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Liste de documents — GET /api/v1/ingest/
# ─────────────────────────────────────────────────────────────────────────────
class DocumentListResponse(BaseModel):
    """Liste paginée de documents avec leur statut."""

    documents: list[DocumentStatusResponse] = Field(
        default_factory=list,
        description="Liste des documents et leur statut",
    )
    total: int = Field(
        ...,
        description="Nombre total de documents",
    )

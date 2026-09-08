# =============================================================================
# EcoFibre Sync — Router d'ingestion (api/v1/ingestion.py)
# =============================================================================
# Ce module expose les endpoints REST pour le pipeline d'ingestion :
#
#   POST   /api/v1/ingest              → Upload et traitement d'un document
#   GET    /api/v1/ingest/{id}/status   → Statut du traitement
#   GET    /api/v1/ingest/              → Liste de tous les documents
#
# Flux de l'upload :
#   1. Le client envoie un fichier via multipart/form-data
#   2. L'API crée un enregistrement Document en base (statut PENDING)
#   3. L'API lance une tâche Celery (ingestion asynchrone)
#   4. L'API retourne immédiatement le document_id + task_id (< 100ms)
#   5. Le client peut poller le statut via GET .../status
#
# Pourquoi cette approche asynchrone ?
#   Le parsing d'un PDF + génération d'embeddings prend 10-60 secondes.
#   On ne peut pas bloquer la requête HTTP aussi longtemps (timeout,
#   mauvaise UX). Le pattern "accepter + traiter en background" est
#   le standard pour ce type d'opération.
# =============================================================================

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.models.document import Document, ProcessingStatus
from app.schemas.document import (
    DocumentListResponse,
    DocumentStatusResponse,
    IngestResponse,
)
from app.tasks.ingestion import ingest_document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration du router
# ─────────────────────────────────────────────────────────────────────────────
# prefix="/api/v1/ingest" : toutes les routes seront préfixées par ce chemin.
# tags=["Ingestion"] : regroupe les endpoints dans Swagger UI sous un onglet.
# ─────────────────────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/api/v1/ingest",
    tags=["Ingestion"],
)

# Types MIME acceptés pour l'upload
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/octet-stream",  # Fallback pour les fichiers sans type
}

# Taille maximale du fichier (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/v1/ingest — Upload d'un document
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload et traitement d'un rapport d'intervention",
    response_description="Document accepté pour traitement asynchrone",
)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Fichier du rapport d'intervention (PDF, DOCX, TXT)",
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload un rapport d'intervention et lance le pipeline d'ingestion.

    Le traitement est asynchrone : l'API retourne immédiatement un
    `document_id` et un `task_id` Celery. Le client peut ensuite
    suivre l'avancement via `GET /api/v1/ingest/{document_id}/status`.

    **Formats supportés :** PDF, DOCX, TXT

    **Étapes du pipeline (en arrière-plan) :**
    1. Extraction du texte (parsing)
    2. Découpage en chunks
    3. Génération des embeddings vectoriels
    4. Indexation dans pgvector
    """
    # ── Validation du fichier ────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le nom du fichier est requis",
        )

    # Lire le contenu du fichier
    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier est vide",
        )

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Le fichier dépasse la taille maximale ({MAX_FILE_SIZE // (1024*1024)} MB)",
        )

    logger.info(
        f"📥 Upload reçu : {file.filename} "
        f"({len(file_content)} bytes, type={file.content_type})"
    )

    # ── Création du document en base ─────────────────────────────────────
    document = Document(
        id=uuid.uuid4(),
        filename=file.filename,
        content_type=file.content_type,
        file_size=len(file_content),
        status=ProcessingStatus.PENDING,
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    logger.info(f"📋 Document créé en base : {document.id}")

    # ── Lancement de la tâche Celery ─────────────────────────────────────
    # On décode le contenu en string pour la sérialisation JSON Celery.
    # Les fichiers binaires (PDF) sont passés comme texte car la tâche
    # utilise le parser simulé pour l'instant. En production, on stockera
    # le fichier sur S3/GCS et on passera l'URL au worker.
    try:
        file_text = file_content.decode("utf-8", errors="replace")
    except Exception:
        file_text = file_content.decode("latin-1", errors="replace")

    task = ingest_document.delay(
        document_id=str(document.id),
        file_content=file_text,
        filename=file.filename,
        content_type=file.content_type,
    )

    logger.info(f"🚀 Tâche Celery lancée : task_id={task.id}")

    # ── Réponse immédiate ────────────────────────────────────────────────
    return IngestResponse(
        document_id=document.id,
        task_id=task.id,
        filename=file.filename,
        status=ProcessingStatus.PENDING,
        message="Document soumis pour traitement. "
                f"Suivez l'avancement via GET /api/v1/ingest/{document.id}/status",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/v1/ingest/{document_id}/status — Statut du traitement
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Statut du traitement d'un document",
    response_description="Statut détaillé du pipeline d'ingestion",
)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retourne le statut actuel du traitement d'un document.

    Statuts possibles :
    - **pending** : en attente dans la queue Celery
    - **parsing** : extraction du texte en cours
    - **embedding** : génération des embeddings en cours
    - **completed** : traitement terminé, prêt pour la recherche RAG
    - **failed** : erreur survenue (voir `error_message`)
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} non trouvé",
        )

    return DocumentStatusResponse.model_validate(document)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/v1/ingest/ — Liste de tous les documents
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="Liste de tous les documents ingérés",
    response_description="Liste paginée des documents avec leur statut",
)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retourne la liste de tous les documents ingérés avec leur statut.

    Supporte la pagination via les paramètres `skip` et `limit`.
    """
    # Compter le total
    count_result = await db.execute(select(func.count(Document.id)))
    total = count_result.scalar() or 0

    # Récupérer les documents paginés
    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = list(result.scalars().all())

    return DocumentListResponse(
        documents=[
            DocumentStatusResponse.model_validate(doc) for doc in documents
        ],
        total=total,
    )

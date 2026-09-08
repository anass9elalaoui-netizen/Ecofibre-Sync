# =============================================================================
# EcoFibre Sync — Tâche Celery d'ingestion (tasks/ingestion.py)
# =============================================================================
# Cette tâche asynchrone orchestre le pipeline complet d'ingestion d'un
# document, depuis le texte brut uploadé jusqu'aux embeddings indexés
# dans pgvector.
#
# Workflow :
#   1. L'API reçoit le fichier → crée un Document en DB → lance cette tâche
#   2. La tâche tourne dans un worker Celery (process séparé)
#   3. Étapes : parsing → embedding → persistance
#   4. Le statut du Document est mis à jour à chaque étape
#
# Pourquoi une tâche Celery plutôt que du background async FastAPI ?
#   - Celery tourne dans un process séparé → pas d'impact sur l'API
#   - Retry automatique avec backoff exponentiel en cas d'erreur
#   - Monitoring via Flower (dashboard temps réel)
#   - Scalabilité horizontale : ajouter des workers pour paralléliser
#   - Persistence : si l'API crash, les tâches en queue ne sont pas perdues
#
# IMPORTANT : les tâches Celery sont SYNCHRONES par défaut.
#   On utilise SQLAlchemy en mode synchrone ici (pas asyncpg) car les
#   workers Celery n'ont pas d'event loop async. On crée une session
#   sync dédiée pour les opérations DB dans le worker.
# =============================================================================

import logging
import uuid

from celery import shared_task
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.document import Document, ProcessingStatus
from app.services.parser import DocumentParser
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session Factory synchrone pour les workers Celery
# ─────────────────────────────────────────────────────────────────────────────
# Les workers Celery tournent dans des threads/processes classiques (pas
# d'event loop async). On utilise donc le driver PostgreSQL synchrone
# (psycopg2 via l'URL sans +asyncpg).
#
# Pourquoi une session factory dédiée plutôt que réutiliser celle de FastAPI ?
#   - L'engine async de FastAPI (asyncpg) n'est pas compatible avec le
#     contexte synchrone de Celery.
#   - Chaque worker crée ses propres connexions, isolées de l'API.
#   - Le pool_size est plus petit (5 vs 20) car les workers font des
#     opérations longues mais peu concurrentes.
# ─────────────────────────────────────────────────────────────────────────────
_sync_url = settings.DATABASE_URL_SYNC.replace("ssl=require", "sslmode=require")

_sync_engine = create_engine(
    _sync_url,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    echo=False,
)

SyncSessionLocal = sessionmaker(
    bind=_sync_engine,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper : mise à jour du statut du document
# ─────────────────────────────────────────────────────────────────────────────
def _update_document_status(
    session: Session,
    document_id: uuid.UUID,
    status: ProcessingStatus,
    error_message: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Met à jour le statut d'un document dans la base de données."""
    values: dict = {"status": status}
    if error_message is not None:
        values["error_message"] = error_message
    if chunk_count is not None:
        values["chunk_count"] = chunk_count

    session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(**values)
    )
    session.commit()

    logger.info(f"📋 Document {document_id} → statut : {status.value}")


# ─────────────────────────────────────────────────────────────────────────────
# Tâche Celery principale : ingestion d'un document
# ─────────────────────────────────────────────────────────────────────────────
@shared_task(
    name="tasks.ingest_document",
    bind=True,
    # ── Retry Configuration ──────────────────────────────────────────────
    # max_retries=3 : on retente 3 fois maximum en cas d'erreur transitoire
    #   (timeout DB, rate limit API embedding, etc.)
    # autoretry_for : liste des exceptions qui déclenchent un retry auto
    # retry_backoff=True : délai exponentiel entre les retries (60s, 120s, 240s)
    # retry_backoff_max=600 : délai max de 10 minutes entre deux retries
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=600,
    # ── Soft/Hard time limits ────────────────────────────────────────────
    # soft_time_limit : envoie un SoftTimeLimitExceeded après 5 min
    #   → on peut catch l'exception et sauvegarder le statut FAILED
    # time_limit : kill dur après 10 min (sécurité contre les boucles infinies)
    soft_time_limit=300,
    time_limit=600,
)
def ingest_document(
    self,
    document_id: str,
    file_content: str,
    filename: str,
    content_type: str | None = None,
) -> dict:
    """
    Tâche Celery d'ingestion complète d'un document.

    Orchestre le pipeline : parsing → embedding → persistance.

    Args:
        self: Référence à l'instance de la tâche Celery (bind=True).
        document_id: UUID du document (str car sérialisé en JSON).
        file_content: Contenu textuel du fichier (déjà décodé).
        filename: Nom du fichier original.
        content_type: Type MIME du fichier.

    Returns:
        Dictionnaire avec le résultat de l'ingestion.
    """
    doc_uuid = uuid.UUID(document_id)
    logger.info(f"🚀 Début de l'ingestion : {filename} (doc_id={document_id})")

    session = SyncSessionLocal()

    try:
        # ── Étape 1 : PARSING ────────────────────────────────────────────
        _update_document_status(session, doc_uuid, ProcessingStatus.PARSING)

        parser = DocumentParser()
        parsed_chunks = parser.parse(
            content=file_content,
            filename=filename,
            content_type=content_type,
        )

        if not parsed_chunks:
            _update_document_status(
                session,
                doc_uuid,
                ProcessingStatus.FAILED,
                error_message="Aucun texte extrait du document",
            )
            return {
                "status": "failed",
                "document_id": document_id,
                "error": "Aucun texte extrait du document",
            }

        logger.info(f"✅ Parsing terminé : {len(parsed_chunks)} chunks")

        # ── Étape 2 : EMBEDDING ──────────────────────────────────────────
        _update_document_status(session, doc_uuid, ProcessingStatus.EMBEDDING)

        vectorstore = VectorStoreService()
        texts = [chunk.content for chunk in parsed_chunks]
        embeddings = vectorstore.generate_embeddings(texts)

        logger.info(f"✅ Embeddings générés : {len(embeddings)} vecteurs")

        # ── Étape 3 : PERSISTANCE ────────────────────────────────────────
        # On crée les DocumentChunk directement avec la session sync
        # car index_chunks est async (conçu pour FastAPI).
        # Ici on fait l'insertion manuellement en sync.
        from app.models.document import DocumentChunk
        from datetime import datetime, timezone

        db_chunks = []
        for parsed_chunk, embedding in zip(parsed_chunks, embeddings):
            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=doc_uuid,
                content=parsed_chunk.content,
                chunk_index=parsed_chunk.chunk_index,
                metadata_extra=str(parsed_chunk.metadata) if parsed_chunk.metadata else None,
                embedding=embedding,
                created_at=datetime.now(timezone.utc),
            )
            db_chunks.append(db_chunk)

        session.add_all(db_chunks)
        session.commit()

        logger.info(f"✅ {len(db_chunks)} chunks persistés en base")

        # ── Étape 4 : MISE À JOUR DU STATUT ─────────────────────────────
        _update_document_status(
            session,
            doc_uuid,
            ProcessingStatus.COMPLETED,
            chunk_count=len(db_chunks),
        )

        result = {
            "status": "completed",
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(db_chunks),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        }

        logger.info(
            f"🎉 Ingestion terminée avec succès : {filename} "
            f"→ {len(db_chunks)} chunks indexés"
        )

        return result

    except Exception as exc:
        # ── Gestion des erreurs ──────────────────────────────────────────
        # On met à jour le statut du document AVANT de relancer l'exception
        # pour que le frontend puisse afficher le diagnostic.
        error_msg = f"{type(exc).__name__}: {exc!s}"
        logger.error(
            f"❌ Erreur d'ingestion pour {filename} : {error_msg}",
            exc_info=True,
        )

        try:
            _update_document_status(
                session,
                doc_uuid,
                ProcessingStatus.FAILED,
                error_message=error_msg[:1000],  # Tronquer les messages trop longs
            )
        except Exception:
            logger.error("❌ Impossible de mettre à jour le statut du document")

        # Relancer l'exception pour que Celery puisse retenter
        raise

    finally:
        # ── Nettoyage ────────────────────────────────────────────────────
        # TOUJOURS fermer la session, même en cas d'erreur.
        # Sans ça, la connexion resterait ouverte et finirait par
        # épuiser le pool PostgreSQL.
        session.close()

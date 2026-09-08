# =============================================================================
# EcoFibre Sync — Modèles ORM pour les documents d'intervention (models/document.py)
# =============================================================================
# Ce module définit deux modèles SQLAlchemy :
#
#   1. Document : métadonnées du rapport d'intervention uploadé
#      (nom du fichier, statut de traitement, timestamps).
#
#   2. DocumentChunk : fragments vectorisés du document.
#      Chaque chunk contient un morceau de texte et son embedding vectoriel
#      stocké dans une colonne pgvector (type Vector).
#
# Relation :
#   Document (1) ──── (N) DocumentChunk
#   Un document est découpé en N chunks, chacun avec son embedding.
#
# Pourquoi pgvector plutôt qu'un vector store externe (Pinecone, Qdrant) ?
#   - Tout dans PostgreSQL = une seule infra à maintenir (MCO simplifié)
#   - Pas de latence réseau supplémentaire pour la recherche vectorielle
#   - Transactions ACID sur les embeddings (cohérence garantie)
#   - L'extension pgvector supporte les index IVFFlat et HNSW pour la perf
# =============================================================================

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enum de statut du traitement
# ─────────────────────────────────────────────────────────────────────────────
# Le document passe par plusieurs étapes après l'upload :
#   PENDING   → reçu, en attente dans la queue Celery
#   PARSING   → le worker Celery extrait le texte (parser.py)
#   EMBEDDING → les chunks sont vectorisés (vectorstore.py)
#   COMPLETED → tout est indexé et prêt pour la recherche RAG
#   FAILED    → une erreur est survenue (consultable via l'API)
#
# Cet enum est stocké comme VARCHAR dans PostgreSQL (pas comme int) pour
# que les requêtes SQL soient lisibles sans table de lookup.
# ─────────────────────────────────────────────────────────────────────────────
class ProcessingStatus(str, enum.Enum):
    """Statut de traitement d'un document dans le pipeline d'ingestion."""
    PENDING = "pending"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modèle Document — Métadonnées du rapport d'intervention
# ─────────────────────────────────────────────────────────────────────────────
# Chaque document uploadé via POST /api/v1/ingest crée une ligne ici.
# Le statut est mis à jour par la tâche Celery au fil du traitement.
#
# Pourquoi UUID plutôt qu'un auto-increment ?
#   - Pas de collision lors d'insertions concurrentes (workers Celery)
#   - Pas d'information exposée (on ne peut pas deviner l'ID suivant)
#   - Compatible avec les systèmes distribués (pas besoin de séquence DB)
# ─────────────────────────────────────────────────────────────────────────────
class Document(Base):
    """Métadonnées d'un rapport d'intervention fibre uploadé."""

    __tablename__ = "documents"

    # ── Clé primaire ─────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Identifiant unique du document (UUID v4)",
    )

    # ── Métadonnées du fichier ───────────────────────────────────────────
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Nom original du fichier uploadé (ex: rapport_ftth_2024.pdf)",
    )

    content_type: Mapped[str] = mapped_column(
        String(128),
        nullable=True,
        comment="Type MIME du fichier (ex: application/pdf)",
    )

    file_size: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Taille du fichier en octets",
    )

    # ── Statut de traitement ─────────────────────────────────────────────
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status", native_enum=False),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True,
        comment="Étape actuelle dans le pipeline d'ingestion",
    )

    # Stocke le message d'erreur si status == FAILED.
    # Permet au frontend d'afficher un diagnostic sans consulter les logs.
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Message d'erreur si le traitement a échoué",
    )

    # Nombre de chunks générés après le parsing.
    # Utile pour le monitoring (un PDF de 100 pages → ~200 chunks attendus).
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Nombre de chunks vectorisés générés",
    )

    # ── Timestamps ───────────────────────────────────────────────────────
    # Tous en UTC pour éviter les problèmes de timezone.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Date d'upload du document",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Dernière mise à jour du statut",
    )

    # ── Relation vers les chunks ─────────────────────────────────────────
    # cascade="all, delete-orphan" : quand un document est supprimé,
    # tous ses chunks sont automatiquement supprimés.
    # lazy="selectin" : charge les chunks via un SELECT IN (efficace pour
    # charger les chunks de plusieurs documents en une seule requête).
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id!s:.8}, filename='{self.filename}', "
            f"status={self.status.value})>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Modèle DocumentChunk — Fragment vectorisé d'un document
# ─────────────────────────────────────────────────────────────────────────────
# Après le parsing, chaque document est découpé en chunks de ~500-1000 tokens.
# Chaque chunk est vectorisé (embedding) et stocké ici avec le texte source.
#
# Architecture de la recherche RAG :
#   1. L'utilisateur pose une question
#   2. La question est vectorisée (même modèle d'embedding)
#   3. pgvector trouve les chunks les plus proches (cosine similarity)
#   4. Les chunks pertinents sont envoyés au LLM comme contexte
#   5. Le LLM génère une réponse basée sur ces chunks
#
# Dimension de l'embedding :
#   3072 dimensions pour les embeddings Google (models/gemini-embedding-001).
#   Ce chiffre DOIT correspondre au modèle d'embedding utilisé.
#   Si on change de modèle, il faut re-vectoriser tous les chunks.
# ─────────────────────────────────────────────────────────────────────────────
# Dimension des embeddings Google Generative AI (models/gemini-embedding-001)
EMBEDDING_DIMENSION = 3072


class DocumentChunk(Base):
    """Fragment vectorisé d'un document d'intervention."""

    __tablename__ = "document_chunks"

    # ── Clé primaire ─────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Identifiant unique du chunk",
    )

    # ── Référence au document parent ─────────────────────────────────────
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="ID du document source",
    )

    # ── Contenu textuel ──────────────────────────────────────────────────
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Texte brut du chunk (portion du document)",
    )

    # ── Métadonnées du chunk ─────────────────────────────────────────────
    # L'index du chunk dans le document (0, 1, 2, ...).
    # Permet de reconstruire l'ordre original si nécessaire.
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position du chunk dans le document (0-indexed)",
    )

    # Métadonnées supplémentaires pour le contexte RAG.
    # Ex: numéro de page, section du rapport, etc.
    metadata_extra: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Métadonnées JSON additionnelles (page, section, etc.)",
    )

    # ── Embedding vectoriel (pgvector) ───────────────────────────────────
    # Le type Vector de pgvector stocke un vecteur de flottants de taille fixe.
    # La dimension (768) DOIT correspondre au modèle d'embedding utilisé.
    #
    # pgvector supporte plusieurs types d'index :
    #   - IVFFlat : rapide à construire, bon pour < 1M vecteurs
    #   - HNSW : plus lent à construire, meilleur recall pour > 1M vecteurs
    # On créera l'index dans une migration Alembic dédiée.
    embedding: Mapped[list | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True,
        comment=f"Embedding vectoriel ({EMBEDDING_DIMENSION} dimensions)",
    )

    # ── Timestamps ───────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Date de création du chunk",
    )

    # ── Relation inverse vers le document ────────────────────────────────
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id!s:.8}, document_id={self.document_id!s:.8}, "
            f"chunk_index={self.chunk_index})>"
        )

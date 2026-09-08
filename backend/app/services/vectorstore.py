# =============================================================================
# EcoFibre Sync — Service Vector Store (services/vectorstore.py)
# =============================================================================
# Ce service gère l'embedding et l'indexation vectorielle des chunks de
# documents dans PostgreSQL via pgvector, en utilisant LangChain.
#
# Architecture :
#   DocumentParser → chunks de texte
#       ↓
#   VectorStoreService → embeddings (Google Generative AI)
#       ↓
#   PostgreSQL (pgvector) → stockage et recherche par similarité
#
# Modèle d'embedding :
#   Google Generative AI (models/embedding-001) via langchain-google-genai.
#   Dimension : 768. C'est un modèle performant et gratuit pour les
#   petits volumes, ce qui est idéal pour le développement.
#
# Pourquoi LangChain pour les embeddings ?
#   - Abstraction uniforme : swapper Google ↔ OpenAI ↔ Ollama en 1 ligne
#   - Gestion automatique du batching et du rate limiting
#   - Intégration native avec PGVector pour la recherche par similarité
# =============================================================================

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import EMBEDDING_DIMENSION, DocumentChunk
from app.services.parser import ParsedChunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Service principal Vector Store
# ─────────────────────────────────────────────────────────────────────────────
class VectorStoreService:
    """
    Service d'embedding et d'indexation vectorielle.

    Ce service :
    1. Génère les embeddings pour une liste de chunks de texte
    2. Persiste les chunks + embeddings dans la table document_chunks
    3. Fournit une méthode de recherche par similarité vectorielle

    L'embedding model est initialisé de manière lazy (au premier appel)
    pour éviter de charger les dépendances Google au démarrage si
    elles ne sont pas nécessaires.
    """

    def __init__(self):
        self._embeddings_model = None

    # ── Initialisation lazy du modèle d'embedding ────────────────────────
    def _get_embeddings_model(self):
        """
        Initialise le modèle d'embedding Google Generative AI.

        Utilise langchain-google-genai pour les embeddings.
        Le modèle est mis en cache après la première initialisation.

        Fallback : si la clé API Google n'est pas configurée, on utilise
        un embedding factice (vecteur aléatoire) pour le développement.
        """
        if self._embeddings_model is not None:
            return self._embeddings_model

        google_api_key = getattr(settings, "GOOGLE_API_KEY", "")

        if google_api_key:
            try:
                # Import conditionnel pour éviter l'erreur si le package
                # n'est pas installé (on reste fonctionnel en mode dev)
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                self._embeddings_model = GoogleGenerativeAIEmbeddings(
                    model=settings.GOOGLE_EMBEDDING_MODEL,
                    google_api_key=google_api_key,
                )
                logger.info(
                    "✅ Modèle d'embedding initialisé : Google Generative AI "
                    f"({settings.GOOGLE_EMBEDDING_MODEL}, dim=768)"
                )
                return self._embeddings_model
            except ImportError:
                logger.warning(
                    "⚠️ langchain-google-genai non installé. "
                    "Utilisation des embeddings factices."
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Erreur d'initialisation Google Embeddings : {e}. "
                    f"Utilisation des embeddings factices."
                )

        # Fallback : embeddings factices pour le développement
        logger.info(
            "🔄 Mode développement : embeddings factices activés "
            f"(dimension={EMBEDDING_DIMENSION})"
        )
        self._embeddings_model = None
        return None

    # ── Génération d'embeddings ──────────────────────────────────────────
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Génère les embeddings pour une liste de textes.

        Args:
            texts: Liste de strings à vectoriser.

        Returns:
            Liste de vecteurs (chaque vecteur = liste de floats).
            La dimension dépend du modèle (768 pour Google embedding-001).

        En mode développement (pas de clé API), retourne des vecteurs
        factices déterministes basés sur le hash du texte. Cela permet
        de tester le pipeline complet sans clé API.
        """
        model = self._get_embeddings_model()

        if model is not None:
            # Production : appel au modèle d'embedding réel
            logger.info(f"🔄 Génération de {len(texts)} embeddings via Google AI...")
            embeddings = model.embed_documents(texts)
            logger.info(f"✅ {len(texts)} embeddings générés")
            return embeddings

        # Développement : embeddings factices déterministes
        logger.info(
            f"🔄 [SIMULATION] Génération de {len(texts)} embeddings factices"
        )
        return [self._generate_fake_embedding(text) for text in texts]

    @staticmethod
    def _generate_fake_embedding(text: str) -> list[float]:
        """
        Génère un embedding factice déterministe basé sur le hash du texte.

        Pourquoi déterministe ?
          Pour que les tests soient reproductibles : le même texte
          produit toujours le même vecteur, ce qui permet de vérifier
          la recherche par similarité même en mode dev.

        La distribution est normalisée entre -1 et 1 pour imiter
        un vrai embedding.
        """
        import hashlib

        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

        # Étendre le hash pour remplir EMBEDDING_DIMENSION valeurs
        # SHA-256 = 32 bytes, on a besoin de 768 floats
        extended = hash_bytes * (EMBEDDING_DIMENSION // len(hash_bytes) + 1)

        embedding = []
        for i in range(EMBEDDING_DIMENSION):
            # Convertir chaque byte en float entre -1 et 1
            value = (extended[i] / 255.0) * 2 - 1
            embedding.append(round(value, 6))

        return embedding

    # ── Persistance des chunks avec embeddings ───────────────────────────
    async def index_chunks(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        parsed_chunks: list[ParsedChunk],
    ) -> list[DocumentChunk]:
        """
        Génère les embeddings et persiste les chunks en base.

        Args:
            session: Session SQLAlchemy async.
            document_id: UUID du document parent.
            parsed_chunks: Chunks issus du parser.

        Returns:
            Liste des DocumentChunk créés et persistés.

        Le processus :
        1. Extraire les textes de tous les chunks
        2. Générer les embeddings en batch (un seul appel API)
        3. Créer les objets ORM avec texte + embedding
        4. Bulk insert dans PostgreSQL
        """
        if not parsed_chunks:
            logger.warning(f"⚠️ Aucun chunk à indexer pour le document {document_id}")
            return []

        # Étape 1 : Extraire les textes
        texts = [chunk.content for chunk in parsed_chunks]

        # Étape 2 : Générer les embeddings en batch
        # Le batching est crucial : un seul appel API pour N textes
        # au lieu de N appels. Réduit la latence de ~Nx à ~1x.
        embeddings = self.generate_embeddings(texts)

        # Étape 3 : Créer les objets ORM
        db_chunks: list[DocumentChunk] = []
        for parsed_chunk, embedding in zip(parsed_chunks, embeddings):
            db_chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=document_id,
                content=parsed_chunk.content,
                chunk_index=parsed_chunk.chunk_index,
                metadata_extra=str(parsed_chunk.metadata) if parsed_chunk.metadata else None,
                embedding=embedding,
                created_at=datetime.now(timezone.utc),
            )
            db_chunks.append(db_chunk)

        # Étape 4 : Bulk insert
        session.add_all(db_chunks)
        await session.flush()

        logger.info(
            f"✅ {len(db_chunks)} chunks indexés pour le document {document_id}"
        )

        return db_chunks

    # ── Recherche par similarité vectorielle ─────────────────────────────
    async def similarity_search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 5,
        document_id: uuid.UUID | None = None,
    ) -> list[DocumentChunk]:
        """
        Recherche les chunks les plus similaires à une requête.

        Utilise la distance cosinus de pgvector pour trouver les chunks
        dont l'embedding est le plus proche de celui de la requête.

        Args:
            session: Session SQLAlchemy async.
            query: Texte de la requête utilisateur.
            top_k: Nombre de résultats à retourner.
            document_id: Optionnel, restreint la recherche à un document.

        Returns:
            Liste des DocumentChunk les plus pertinents, triés par
            similarité décroissante.
        """
        # Vectoriser la requête avec le même modèle
        query_embedding = self.generate_embeddings([query])[0]

        # Construction de la requête pgvector
        # L'opérateur <=> calcule la distance cosinus.
        # Plus la distance est petite, plus les vecteurs sont similaires.
        stmt = (
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        # Filtrer par document si spécifié
        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        result = await session.execute(stmt)
        chunks = list(result.scalars().all())

        logger.info(
            f"🔍 Recherche similarité : '{query[:50]}...' → {len(chunks)} résultats"
        )

        return chunks

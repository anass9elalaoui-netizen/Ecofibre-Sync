# =============================================================================
# EcoFibre Sync — Database Configuration (database.py)
# =============================================================================
# Ce module configure la couche base de données :
#   1. Engine async SQLAlchemy 2.0 (via asyncpg)
#   2. Session factory async (pour les requêtes DB dans FastAPI)
#   3. Classe de base pour tous les modèles ORM
#   4. Fonction d'initialisation de pgvector
#
# Architecture des sessions :
#   FastAPI utilise l'injection de dépendances pour fournir une session
#   à chaque endpoint. La session est créée au début de la requête et
#   fermée à la fin, garantissant qu'on ne laisse jamais de connexion
#   ouverte en cas d'erreur.
# =============================================================================

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# 1. Engine Async
# ─────────────────────────────────────────────────────────────────────────────
# L'engine est le point d'entrée vers la base de données. Il gère le pool
# de connexions et traduit les requêtes SQLAlchemy en SQL PostgreSQL.
#
# Paramètres du pool :
#   - pool_size=20 : nombre de connexions maintenues en permanence.
#     20 est un bon défaut pour une API web (correspond à ~20 requêtes
#     concurrentes avant de devoir attendre).
#   - max_overflow=10 : connexions supplémentaires temporaires au-delà
#     du pool_size en cas de pic de charge. Elles sont fermées après usage.
#   - pool_pre_ping=True : vérifie que la connexion est vivante avant de
#     l'utiliser. Évite les erreurs "connection closed" après un timeout
#     PostgreSQL ou un redémarrage du container.
#   - echo=False : ne log pas chaque requête SQL en production.
#     En dev, on peut passer echo=True via APP_DEBUG pour debugger.
# ─────────────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=settings.APP_DEBUG,  # Log SQL en mode debug uniquement
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Session Factory Async
# ─────────────────────────────────────────────────────────────────────────────
# async_sessionmaker crée un "patron" (factory) pour instancier des sessions.
# Chaque appel à AsyncSessionLocal() crée une nouvelle session liée à l'engine.
#
# Paramètres :
#   - bind=engine : la session utilise notre engine async.
#   - class_=AsyncSession : type de session retourné (async).
#   - expire_on_commit=False : après un commit, les objets ORM restent
#     accessibles en mémoire sans re-query. C'est important en async car
#     un await supplémentaire pour recharger les attributs serait coûteux.
#     Sans ça, accéder à obj.name après commit lèverait une erreur.
# ─────────────────────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Base ORM
# ─────────────────────────────────────────────────────────────────────────────
# Tous les modèles SQLAlchemy du projet hériteront de cette classe.
# DeclarativeBase (SQLAlchemy 2.0) remplace l'ancien declarative_base().
#
# Exemple d'utilisation dans app/models/ :
#   class Intervention(Base):
#       __tablename__ = "interventions"
#       id = mapped_column(Integer, primary_key=True)
#       ...
# ─────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM du projet."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dependency Injection pour FastAPI
# ─────────────────────────────────────────────────────────────────────────────
# Cette fonction est un générateur async utilisé comme Depends() dans FastAPI.
#
# Workflow :
#   1. Crée une session au début de la requête HTTP
#   2. yield la session → le endpoint l'utilise pour ses queries
#   3. Après le yield (que ce soit succès ou erreur), ferme la session
#
# Le try/finally garantit que la session est TOUJOURS fermée, même si
# le endpoint lève une exception. Sans ça, on aurait des fuites de
# connexions qui finiraient par épuiser le pool.
#
# Usage dans un endpoint :
#   @router.get("/items")
#   async def get_items(db: AsyncSession = Depends(get_db_session)):
#       result = await db.execute(select(Item))
#       return result.scalars().all()
# ─────────────────────────────────────────────────────────────────────────────
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Fournit une session DB async, auto-fermée après usage."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Initialisation de la base de données
# ─────────────────────────────────────────────────────────────────────────────
# Cette fonction est appelée une seule fois au démarrage de l'application
# (dans le lifespan de FastAPI). Elle :
#   1. Active l'extension pgvector dans PostgreSQL
#   2. Crée toutes les tables ORM qui n'existent pas encore
#
# Pourquoi CREATE EXTENSION IF NOT EXISTS ?
#   L'extension vector doit être activée AVANT de créer des tables avec
#   des colonnes de type Vector. IF NOT EXISTS évite les erreurs si
#   l'extension est déjà activée (idempotent).
#
# Pourquoi run_sync(Base.metadata.create_all) ?
#   SQLAlchemy 2.0 async n'a pas de create_all() async natif.
#   run_sync() exécute une fonction sync dans le contexte async de façon
#   thread-safe. C'est la méthode recommandée par la doc SQLAlchemy.
#
# ⚠️ En production, on utilisera Alembic pour les migrations au lieu de
#    create_all(). Cette fonction reste utile pour le dev et les tests.
# ─────────────────────────────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Initialise la base de données :
    - Active l'extension pgvector
    - Crée les tables manquantes
    """
    async with engine.begin() as conn:
        # Étape 1 : Activer pgvector
        # L'image ankane/pgvector inclut l'extension, mais elle doit être
        # explicitement activée dans chaque base de données.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Étape 2 : Créer les tables ORM
        # Inspecte Base.metadata (tous les modèles importés) et crée
        # les tables qui n'existent pas encore dans PostgreSQL.
        await conn.run_sync(Base.metadata.create_all)

# =============================================================================
# EcoFibre Sync — Configuration centralisée (config.py)
# =============================================================================
# Ce module utilise pydantic-settings pour charger et valider TOUTES les
# variables d'environnement au démarrage de l'application.
#
# Pourquoi pydantic-settings plutôt que os.getenv() ?
#   - Validation de type au démarrage (crash immédiat si une variable
#     manque ou a un mauvais format, au lieu d'un crash en runtime).
#   - Autocomplétion IDE sur settings.POSTGRES_HOST, etc.
#   - Valeurs par défaut centralisées en un seul endroit.
#   - Support natif des fichiers .env (pas besoin de charger manuellement).
#
# Usage dans le reste du code :
#   from app.core.config import settings
#   print(settings.DATABASE_URL)
# =============================================================================

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Remonte de backend/app/core/ → backend/app/ → backend/ → racine du projet
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """
    Configuration globale de l'application.

    Les valeurs sont lues dans cet ordre de priorité (du plus prioritaire
    au moins prioritaire) :
      1. Variables d'environnement système
      2. Fichier .env à la racine du projet
      3. Valeurs par défaut définies ci-dessous
    """

    # ── Informations Projet ──────────────────────────────────────────────
    # Métadonnées utilisées dans les logs, les traces Langfuse, et la
    # documentation OpenAPI auto-générée par FastAPI.
    PROJECT_NAME: str = "EcoFibre Sync"
    APP_ENV: str = "development"       # development | staging | production
    APP_DEBUG: bool = True             # Active les logs détaillés en dev
    APP_LOG_LEVEL: str = "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # ── PostgreSQL ───────────────────────────────────────────────────────
    # On stocke les composants individuels ET l'URL complète.
    # Les composants servent pour des cas spécifiques (healthcheck, Alembic).
    # L'URL complète est utilisée par SQLAlchemy.
    #
    # Format : postgresql+asyncpg://user:password@host:port/dbname
    #   - postgresql+asyncpg:// → indique à SQLAlchemy d'utiliser le
    #     driver asyncpg (async) au lieu de psycopg2 (sync).
    POSTGRES_USER: str = "ecofibre"
    POSTGRES_PASSWORD: str = "ecofibre_dev"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ecofibre_sync"
    DATABASE_URL: str = "postgresql+asyncpg://ecofibre:ecofibre_dev@localhost:5432/ecofibre_sync"

    # URL synchrone pour Alembic (les migrations ne supportent pas encore
    # bien l'async dans tous les cas). On utilise le driver psycopg2 classique.
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """URL synchrone pour Alembic migrations."""
        return self.DATABASE_URL.replace("+asyncpg", "")

    # ── Redis ────────────────────────────────────────────────────────────
    # Une seule instance Redis, mais on utilise des bases différentes
    # (numérotées 0-15) pour séparer les usages :
    #   - /0 → Celery broker (file d'attente des tâches)
    #   - /1 → Celery result backend (résultats des tâches)
    # Cela permet de flush une base sans impacter l'autre.
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ── OpenAI / LLM ────────────────────────────────────────────────────
    # Clé API pour le provider LLM. On utilise OpenAI par défaut, mais
    # l'architecture LangChain permet de swapper facilement vers
    # Anthropic, Mistral, ou Ollama (local) sans changer la logique métier.
    #
    # OPENAI_EMBEDDING_MODEL : modèle d'embeddings pour vectoriser les
    #   rapports d'intervention. text-embedding-3-small offre un bon
    #   ratio qualité/coût (1536 dimensions, ~$0.02/1M tokens).
    #
    # OPENAI_CHAT_MODEL : modèle de chat pour les agents LangGraph.
    #   gpt-4o est le plus performant pour le raisonnement multi-étapes.
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"

    # ── Google Generative AI (Embeddings) ────────────────────────────────
    # Clé API pour les embeddings Google Generative AI.
    # Le modèle models/embedding-001 produit des vecteurs de 768 dimensions.
    # Gratuit pour les petits volumes — idéal pour le développement.
    # En production, on pourra basculer vers text-embedding-3-small d'OpenAI
    # ou un modèle local via Ollama.
    GOOGLE_API_KEY: str = ""
    GOOGLE_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GOOGLE_CHAT_MODEL: str = "gemini-3.6-flash"

    # ── Langfuse (Observabilité LLM) ────────────────────────────────────
    # Langfuse trace chaque appel LLM pour le monitoring et le debugging.
    # Les clés sont optionnelles en dev (Langfuse est désactivé si vides).
    # En production, elles sont obligatoires pour le MCO.
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── Configuration Pydantic Settings ──────────────────────────────────
    # model_config remplace l'ancienne classe Meta/Config.
    # C'est la façon Pydantic v2 de configurer le comportement du modèle.
    model_config = SettingsConfigDict(
        # Chemin absolu vers .env à la racine du projet.
        # Utilise __file__ pour être indépendant du répertoire de lancement.
        env_file=str(_PROJECT_ROOT / ".env"),

        # Si True, les variables .env écrasent les valeurs par défaut
        # mais sont écrasées par les variables d'environnement système.
        env_file_encoding="utf-8",

        # Sensibilité à la casse : désactivée pour être tolérant
        # (DATABASE_URL = database_url = Database_Url).
        case_sensitive=False,

        # Autoriser les champs supplémentaires dans le .env sans erreur.
        # Utile quand le .env contient des variables pour d'autres services.
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton des settings, mis en cache par lru_cache.

    Pourquoi lru_cache ?
      Le fichier .env est lu et parsé une seule fois au premier appel.
      Les appels suivants retournent l'instance en cache, sans I/O disque.
      C'est important car get_settings() est appelé dans les Depends()
      de FastAPI à chaque requête.
    """
    return Settings()


# Instance globale pour import direct : `from app.core.config import settings`
# C'est un raccourci pratique pour ne pas appeler get_settings() partout.
settings = get_settings()

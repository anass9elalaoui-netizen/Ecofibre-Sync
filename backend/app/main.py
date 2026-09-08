# =============================================================================
# EcoFibre Sync — Point d'entrée FastAPI (main.py)
# =============================================================================
# Ce module configure l'application FastAPI avec :
#   1. Lifespan : connexion/déconnexion DB au démarrage/arrêt
#   2. Middleware CORS : autorise le frontend React à appeler l'API
#   3. Endpoint /health : diagnostic de santé (DB + Redis)
#
# Lancement :
#   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# Documentation auto-générée :
#   - Swagger UI : http://localhost:8000/docs
#   - ReDoc      : http://localhost:8000/redoc
# =============================================================================

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# IMPORTANT : Charger la configuration Celery (Eager mode)
import celery_app  # noqa: F401

from app.core.config import settings
from app.core.database import engine, init_db

# Import des modèles ORM pour que Base.metadata.create_all() les détecte.
# Sans cet import, les tables documents et document_chunks ne seraient pas
# créées au démarrage car SQLAlchemy ne scan pas les fichiers automatiquement.
from app.models.document import Document, DocumentChunk  # noqa: F401

# Import du router d'ingestion (API v1)
from app.api.v1.ingestion import router as ingestion_router

# Import du router de résolution de pannes (API v1 — Agents LangGraph)
from app.api.v1.resolution import router as resolution_router


# ── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lifespan — Cycle de vie de l'application
# ─────────────────────────────────────────────────────────────────────────────
# Le lifespan remplace les anciens événements @app.on_event("startup") et
# @app.on_event("shutdown") (dépréciés depuis FastAPI 0.100+).
#
# Pourquoi un lifespan ?
#   Il garantit que les ressources (DB, Redis) sont correctement :
#   - Initialisées AVANT que l'API n'accepte des requêtes
#   - Nettoyées APRÈS que la dernière requête soit traitée
#
# Fonctionnement :
#   1. Code AVANT le yield → exécuté au DÉMARRAGE
#   2. yield → l'API tourne et accepte les requêtes
#   3. Code APRÈS le yield → exécuté à l'ARRÊT (Ctrl+C, SIGTERM, etc.)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application FastAPI."""

    # ── DÉMARRAGE ────────────────────────────────────────────────────────
    logger.info("🚀 Démarrage EcoFibre Sync...")

    # Initialise la DB : active pgvector + crée les tables manquantes
    try:
        await init_db()
        logger.info("✅ Base de données initialisée (pgvector activé)")
    except Exception as e:
        # On log l'erreur mais on ne crash pas l'API.
        # L'endpoint /health permettra de diagnostiquer le problème.
        logger.error(f"❌ Erreur d'initialisation DB : {e}")

    # L'API est prête à recevoir des requêtes
    logger.info(f"✅ {settings.PROJECT_NAME} démarré en mode {settings.APP_ENV}")

    yield  # ← L'application tourne ici

    # ── ARRÊT ────────────────────────────────────────────────────────────
    logger.info("🛑 Arrêt EcoFibre Sync...")

    # Ferme proprement le pool de connexions SQLAlchemy.
    # Sans ça, des connexions pourraient rester ouvertes côté PostgreSQL
    # et atteindre la limite max_connections.
    await engine.dispose()
    logger.info("✅ Pool de connexions DB fermé")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Création de l'application FastAPI
# ─────────────────────────────────────────────────────────────────────────────
# Paramètres de l'application :
#   - title/description/version : apparaissent dans la doc Swagger/ReDoc
#   - lifespan : gestionnaire de cycle de vie (défini ci-dessus)
#   - docs_url/redoc_url : URLs de la documentation interactive
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Système RAG asynchrone pour la Direction Déploiement et Exploitation. "
        "Parse les rapports d'intervention fibre et recommande des procédures "
        "de réparation via des agents autonomes (Numérique Responsable)."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc (documentation alternative)
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Middleware CORS
# ─────────────────────────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) contrôle quels domaines peuvent
# appeler notre API depuis un navigateur web.
#
# Problème résolu :
#   Le frontend React tourne sur http://localhost:3000 (ou 5173 avec Vite)
#   mais l'API tourne sur http://localhost:8000. Sans CORS, le navigateur
#   bloquerait TOUTES les requêtes du frontend vers l'API (Same-Origin Policy).
#
# Configuration actuelle (permissive pour le développement) :
#   - allow_origins=["*"] : accepte les requêtes de TOUT domaine.
#     ⚠️ En production, on restreindra aux domaines autorisés :
#     allow_origins=["https://ecofibre.telecom-operator.com"]
#   - allow_methods=["*"] : autorise GET, POST, PUT, DELETE, etc.
#   - allow_headers=["*"] : autorise tous les headers (Authorization, etc.)
#   - allow_credentials=True : autorise l'envoi de cookies/tokens.
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite dev server
        "http://localhost:3000",       # React dev server (alt)
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://ecofibresync-frontend.onrender.com",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Enregistrement des Routers API
# ─────────────────────────────────────────────────────────────────────────────
# Chaque domaine fonctionnel a son propre router dans app/api/v1/.
# On les inclut ici pour les rendre accessibles via l'URL préfixée.
#
# Routers enregistrés :
#   Ingestion  : POST /api/v1/ingest/            → Upload de documents
#                GET  /api/v1/ingest/{id}/status → Statut du traitement
#                GET  /api/v1/ingest/            → Liste des documents
#
#   Résolution : POST /api/v1/resolution/solve   → Résolution de panne IA
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(ingestion_router)
app.include_router(resolution_router)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Endpoint /health — Diagnostic de santé
# ─────────────────────────────────────────────────────────────────────────────
# Cet endpoint est essentiel pour :
#   - Les healthchecks Docker/Kubernetes (readiness/liveness probes)
#   - Le monitoring Ops (Datadog, Grafana, etc.)
#   - Le debugging rapide : "est-ce que la DB est up ?"
#
# Il vérifie indépendamment chaque composant et retourne :
#   - status : "healthy" si tout est OK, "degraded" si un composant est down
#   - Détail par composant (database, redis) avec latence
#
# Pourquoi vérifier chaque composant séparément ?
#   En cas de panne partielle (ex: Redis down mais DB OK), le endpoint
#   retourne "degraded" avec le détail, plutôt qu'un simple "unhealthy".
#   Cela aide l'équipe Ops à cibler le problème rapidement.
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Infrastructure"],
    summary="Vérification de santé du système",
    response_description="État de santé de chaque composant",
)
async def health_check():
    """
    Vérifie la connectivité avec PostgreSQL et Redis.
    Retourne le statut détaillé de chaque composant.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": settings.PROJECT_NAME,
        "environment": settings.APP_ENV,
        "components": {},
    }

    # ── Check PostgreSQL ─────────────────────────────────────────────────
    try:
        # On utilise une connexion brute (pas une session ORM) pour
        # minimiser l'overhead. "SELECT 1" est la requête la plus simple
        # possible — elle vérifie que le driver, le réseau, l'auth, et
        # le moteur SQL fonctionnent tous correctement.
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.scalar()

            # Vérifie aussi que pgvector est bien activé
            ext_result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            pgvector_active = ext_result.scalar() is not None

        health_status["components"]["database"] = {
            "status": "healthy",
            "type": "PostgreSQL + pgvector",
            "pgvector_enabled": pgvector_active,
            "connection": "ok" if row == 1 else "unexpected",
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "type": "PostgreSQL + pgvector",
            "error": str(e),
        }
        logger.error(f"❌ Healthcheck DB échoué : {e}")

    # ── Check Redis ──────────────────────────────────────────────────────
    try:
        # Connexion async à Redis via redis.asyncio (ex aioredis).
        # On crée une connexion éphémère juste pour le healthcheck.
        # En production, on pourrait utiliser un pool partagé.
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        # PING retourne "PONG" si Redis est vivant.
        pong = await redis_client.ping()

        # INFO server retourne les métadonnées du serveur Redis
        # (version, uptime, mémoire utilisée, etc.)
        info = await redis_client.info("server")
        await redis_client.aclose()

        health_status["components"]["redis"] = {
            "status": "healthy",
            "type": "Redis",
            "ping": "PONG" if pong else "failed",
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "type": "Redis",
            "error": str(e),
        }
        logger.error(f"❌ Healthcheck Redis échoué : {e}")

    return health_status


# ─────────────────────────────────────────────────────────────────────────────
# 5. Endpoint racine — Information de bienvenue
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Infrastructure"],
    summary="Information de l'API",
)
async def root():
    """Retourne les informations de base de l'API."""
    return {
        "project": settings.PROJECT_NAME,
        "version": "0.1.0",
        "description": "Système RAG pour rapports d'intervention fibre",
        "docs": "/docs",
        "health": "/health",
    }

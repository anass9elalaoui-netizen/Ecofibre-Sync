# =============================================================================
# EcoFibre Sync — Configuration Celery (celery_app.py)
# =============================================================================
# Ce module configure l'instance Celery qui gère les tâches asynchrones.
#
# Pourquoi Celery ?
#   Certaines opérations sont trop lourdes pour être exécutées dans le
#   cycle requête/réponse HTTP (timeout de 30s en général) :
#     - Parsing d'un PDF de 200 pages (~10-30s)
#     - Génération d'embeddings pour des centaines de chunks (~20-60s)
#     - Exécution d'un agent LangGraph multi-étapes (~30-120s)
#
#   Celery résout ce problème : l'API crée une tâche, retourne un task_id
#   immédiatement (< 100ms), et un worker Celery exécute le travail en
#   arrière-plan. Le frontend peut ensuite poller le statut via task_id.
#
# Architecture :
#   [FastAPI API] --(crée tâche)--> [Redis Broker] --(consomme)--> [Celery Worker]
#                                                                       |
#                                   [Redis Backend] <--(stocke résultat)--
#                                       |
#   [FastAPI API] --(lit résultat)------
#
# Lancement du worker :
#   celery -A celery_app.celery worker --loglevel=info
#
# Lancement du monitoring (optionnel) :
#   celery -A celery_app.celery flower --port=5555
# =============================================================================

from celery import Celery
from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# 1. Création de l'instance Celery
# ─────────────────────────────────────────────────────────────────────────────
# Le premier argument "ecofibre_sync" est le nom du projet Celery.
# Il est utilisé comme préfixe dans les logs et dans la clé Redis.
#
# broker : URL Redis pour la FILE D'ATTENTE des tâches (queue).
#   Les tâches sont sérialisées en JSON et poussées dans cette queue.
#
# backend : URL Redis pour le STOCKAGE DES RÉSULTATS.
#   On utilise une base Redis différente (/1 vs /0) pour pouvoir
#   flusher les résultats expirés sans toucher à la queue de tâches.
# ─────────────────────────────────────────────────────────────────────────────
celery = Celery(
    "ecofibre_sync",
    broker="memory://",  # Remplace settings.CELERY_BROKER_URL pour éviter Redis en dev local
    backend="cache+memory://",  # Remplace settings.CELERY_RESULT_BACKEND
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configuration détaillée
# ─────────────────────────────────────────────────────────────────────────────
celery.conf.update(
    # ── Exécution Synchrone (Pas besoin de Redis pour le POC) ────────────
    # Exécute la tâche immédiatement de manière synchrone dans le même processus.
    # Permet de se passer de Redis sans avoir à retirer Celery du code.
    task_always_eager=True,
    task_eager_propagates=True,

    # ── Sérialisation ────────────────────────────────────────────────────
    # JSON est choisi plutôt que pickle pour la sécurité :
    #   - pickle peut exécuter du code arbitraire à la désérialisation
    #     (risque d'injection si le broker est compromis).
    #   - JSON est inoffensif et lisible (debugging facile avec redis-cli).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],  # Refuse tout format autre que JSON

    # ── Timezone ─────────────────────────────────────────────────────────
    # UTC pour éviter les problèmes de DST (heure d'été/hiver).
    # Les conversions locales se font au niveau du frontend.
    timezone="UTC",
    enable_utc=True,

    # ── Résultats ────────────────────────────────────────────────────────
    # result_expires : durée de rétention des résultats dans Redis (en sec).
    #   24h est suffisant pour notre cas d'usage. Après, les résultats
    #   sont automatiquement supprimés pour libérer la mémoire Redis.
    result_expires=86400,  # 24 heures

    # ── Fiabilité ────────────────────────────────────────────────────────
    # task_acks_late : le worker n'acquitte (ACK) la tâche qu'APRÈS
    #   l'avoir terminée avec succès. Si le worker crash en pleine
    #   exécution, la tâche retourne dans la queue et est retentée
    #   par un autre worker. Sans ça, la tâche serait perdue.
    task_acks_late=True,

    # worker_prefetch_multiplier : nombre de tâches pré-chargées par worker.
    #   1 = le worker ne prend qu'une tâche à la fois.
    #   Important pour les tâches longues (RAG, agents) : on ne veut pas
    #   qu'un worker bloque 10 tâches pendant qu'il traite la première.
    worker_prefetch_multiplier=1,

    # task_reject_on_worker_lost : si le worker est tué (OOM, kill -9),
    #   la tâche est rejetée et retourne dans la queue au lieu d'être
    #   marquée comme échouée. Assure la résilience.
    task_reject_on_worker_lost=True,

    # ── Retry ────────────────────────────────────────────────────────────
    # Politique de retry par défaut pour toutes les tâches.
    # Les tâches individuelles peuvent surcharger ces valeurs.
    task_default_retry_delay=60,    # Délai initial avant retry : 60s
    task_max_retries=3,             # 3 tentatives max

    # ── Tracking ─────────────────────────────────────────────────────────
    # task_track_started : enregistre l'état "STARTED" dans le backend.
    #   Par défaut, Celery ne track que PENDING → SUCCESS/FAILURE.
    #   Avec ça, on peut montrer au frontend que la tâche est "en cours"
    #   plutôt que simplement "en attente".
    task_track_started=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Auto-découverte des tâches
# ─────────────────────────────────────────────────────────────────────────────
# autodiscover_tasks scanne les modules listés pour trouver les fonctions
# décorées avec @celery.task ou @shared_task.
#
# Cela permet de définir les tâches dans app/tasks/ sans avoir à les
# importer manuellement ici. Chaque nouveau fichier dans app/tasks/
# sera automatiquement détecté au démarrage du worker.
#
# Exemple : app/tasks/parsing.py contient @celery.task def parse_report(...)
#   → Celery la découvre automatiquement et l'enregistre sous le nom
#     "app.tasks.parsing.parse_report"
# ─────────────────────────────────────────────────────────────────────────────
celery.autodiscover_tasks(["app.tasks"])

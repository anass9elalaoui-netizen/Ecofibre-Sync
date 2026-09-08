# =============================================================================
# EcoFibre Sync — Service de parsing de documents (services/parser.py)
# =============================================================================
# Ce service gère l'extraction de texte et le découpage en chunks des
# rapports d'intervention fibre uploadés.
#
# Pipeline de parsing :
#   1. Extraction du texte brut depuis le fichier (PDF, DOCX, TXT)
#   2. Nettoyage du texte (espaces multiples, caractères spéciaux)
#   3. Découpage en chunks de taille uniforme avec chevauchement (overlap)
#
# Pourquoi un overlap entre les chunks ?
#   Sans overlap, une phrase coupée entre deux chunks perd son sens.
#   Avec un overlap de ~100 caractères, le contexte est préservé aux
#   frontières des chunks, ce qui améliore la qualité du RAG.
#
# Mode actuel : SIMULATION
#   Pour garder le développement rapide à cette étape, l'extraction de
#   texte est simulée. En production, on utilisera pypdf, python-docx,
#   et unstructured pour l'extraction réelle. La structure du code est
#   déjà prête pour cette transition.
# =============================================================================

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration du chunking
# ─────────────────────────────────────────────────────────────────────────────
# Ces valeurs sont calibrées pour les rapports d'intervention fibre :
#   - CHUNK_SIZE=1000 : ~200-250 mots par chunk, assez pour un paragraphe
#     technique complet (description d'un défaut, procédure de réparation)
#   - CHUNK_OVERLAP=200 : 20% de recouvrement, préserve le contexte aux
#     frontières sans trop dupliquer de données
#
# En production, on pourrait rendre ces valeurs configurables via .env
# pour ajuster selon le type de document.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 1000       # Taille cible d'un chunk en caractères
DEFAULT_CHUNK_OVERLAP = 200     # Chevauchement entre chunks consécutifs


@dataclass
class ParsedChunk:
    """
    Représente un fragment de texte extrait d'un document.

    Attributes:
        content: Texte brut du chunk.
        chunk_index: Position du chunk dans le document (0-indexed).
        metadata: Métadonnées additionnelles (page, section, etc.).
    """
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Service principal de parsing
# ─────────────────────────────────────────────────────────────────────────────
class DocumentParser:
    """
    Service d'extraction de texte et de découpage en chunks.

    Ce service est conçu pour être facilement extensible :
    - Ajouter un nouveau format → implémenter une méthode _extract_from_*
    - Changer la stratégie de chunking → modifier _split_into_chunks
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Point d'entrée principal ─────────────────────────────────────────
    def parse(
        self,
        content: bytes | str,
        filename: str,
        content_type: str | None = None,
    ) -> list[ParsedChunk]:
        """
        Parse un document et retourne une liste de chunks.

        Args:
            content: Contenu brut du fichier (bytes) ou texte déjà extrait.
            filename: Nom du fichier original (pour détecter le format).
            content_type: Type MIME optionnel.

        Returns:
            Liste de ParsedChunk prêts pour l'embedding.
        """
        logger.info(f"📄 Début du parsing : {filename}")

        # Étape 1 : Extraction du texte brut
        raw_text = self._extract_text(content, filename, content_type)

        if not raw_text or not raw_text.strip():
            logger.warning(f"⚠️ Aucun texte extrait de {filename}")
            return []

        # Étape 2 : Nettoyage du texte
        cleaned_text = self._clean_text(raw_text)

        # Étape 3 : Découpage en chunks
        chunks = self._split_into_chunks(cleaned_text, filename)

        logger.info(
            f"✅ Parsing terminé : {filename} → {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks

    # ── Extraction de texte ──────────────────────────────────────────────
    def _extract_text(
        self,
        content: bytes | str,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        """
        Extrait le texte brut d'un fichier.

        Mode actuel : SIMULATION
        En production, cette méthode appellera pypdf, python-docx, ou
        unstructured selon le format détecté.
        """
        # Si le contenu est déjà du texte, on le retourne directement
        if isinstance(content, str):
            return content

        # Détection du format par extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "txt" or content_type == "text/plain":
            return self._extract_from_txt(content)
        elif ext == "pdf" or content_type == "application/pdf":
            return self._extract_from_pdf_simulated(content, filename)
        elif ext in ("docx", "doc"):
            return self._extract_from_docx_simulated(content, filename)
        else:
            # Fallback : tenter de décoder en UTF-8
            logger.warning(
                f"⚠️ Format non reconnu pour {filename} (ext={ext}), "
                f"tentative de décodage UTF-8"
            )
            return self._extract_from_txt(content)

    # ── Extracteurs par format ───────────────────────────────────────────

    @staticmethod
    def _extract_from_txt(content: bytes) -> str:
        """Extraction directe depuis un fichier texte."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="replace")

    @staticmethod
    def _extract_from_pdf_simulated(content: bytes, filename: str) -> str:
        """
        SIMULATION : extraction de texte depuis un PDF.

        TODO (Étape production) : Remplacer par :
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            return "\\n".join(page.extract_text() for page in reader.pages)
        """
        logger.info(f"🔄 [SIMULATION] Extraction PDF pour {filename}")
        return (
            f"[Contenu simulé du PDF : {filename}]\n\n"
            "RAPPORT D'INTERVENTION FIBRE OPTIQUE\n"
            "========================================\n\n"
            "1. CONTEXTE DE L'INTERVENTION\n"
            "Date : 2024-03-15\n"
            "Technicien : Jean Dupont (ID: TECH-042)\n"
            "Zone : NRO Orléans-Nord — PM-045 — PBO-12\n\n"
            "2. DIAGNOSTIC\n"
            "Affaiblissement anormal détecté sur la fibre G.657.A2 entre "
            "le PM-045 et le PBO-12 (tronçon de 340m). Mesure OTDR : "
            "atténuation de 1.2 dB/km au lieu de 0.35 dB/km nominal. "
            "Suspicion de macro-courbure au niveau du passage en gaine "
            "technique de l'immeuble (rayon de courbure < 15mm constaté).\n\n"
            "3. ACTIONS RÉALISÉES\n"
            "- Repositionnement de la fibre dans la gaine technique avec "
            "respect du rayon de courbure minimal (30mm pour G.657.A2)\n"
            "- Remplacement du manchon de protection endommagé\n"
            "- Nouvelle soudure par fusion au point de raccordement PBO-12 "
            "(perte mesurée : 0.02 dB, conforme au seuil de 0.1 dB)\n\n"
            "4. MESURES POST-INTERVENTION\n"
            "- Atténuation mesurée : 0.38 dB/km (conforme)\n"
            "- Réflectance : -55 dB (conforme, seuil : -40 dB)\n"
            "- Débit client restauré : 940 Mbps DL / 480 Mbps UL\n\n"
            "5. RECOMMANDATIONS\n"
            "Prévoir une inspection des passages en gaine technique pour "
            "les 12 autres fibres du même câble. Risque de macro-courbure "
            "similaire sur les fibres adjacentes.\n"
        )

    @staticmethod
    def _extract_from_docx_simulated(content: bytes, filename: str) -> str:
        """
        SIMULATION : extraction de texte depuis un DOCX.

        TODO (Étape production) : Remplacer par :
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\\n".join(para.text for para in doc.paragraphs)
        """
        logger.info(f"🔄 [SIMULATION] Extraction DOCX pour {filename}")
        return (
            f"[Contenu simulé du DOCX : {filename}]\n\n"
            "COMPTE-RENDU D'INTERVENTION — Maintenance Préventive\n"
            "=====================================================\n\n"
            "Objet : Inspection et maintenance préventive du réseau FTTH\n"
            "Secteur : Agglomération d'Orléans — Quartier Madeleine\n\n"
            "Points de contrôle effectués :\n"
            "- État des boîtiers PBO (12 inspectés, 2 nécessitent nettoyage)\n"
            "- Niveau de signal optique (tous conformes, marge > 3 dB)\n"
            "- État des câbles en façade (1 câble exposé à signaler)\n"
        )

    # ── Nettoyage du texte ───────────────────────────────────────────────
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Nettoie le texte extrait avant le chunking.

        Opérations :
        - Supprime les espaces multiples (courant dans les exports PDF)
        - Normalise les sauts de ligne
        - Supprime les lignes vides consécutives
        - Trim les espaces en début/fin
        """
        # Remplacer les tabulations par des espaces
        text = text.replace("\t", " ")

        # Normaliser les retours chariot Windows (CRLF → LF)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Supprimer les espaces multiples (garder un seul)
        text = re.sub(r" {2,}", " ", text)

        # Supprimer les lignes vides consécutives (garder max 2 newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ── Découpage en chunks ──────────────────────────────────────────────
    def _split_into_chunks(
        self,
        text: str,
        filename: str,
    ) -> list[ParsedChunk]:
        """
        Découpe le texte en chunks de taille uniforme avec overlap.

        Stratégie : découpage par paragraphes d'abord, puis par taille.
        On essaie de couper aux limites de paragraphes pour préserver
        la cohérence sémantique. Si un paragraphe est trop long, on
        coupe aux limites de phrases.
        """
        if len(text) <= self.chunk_size:
            # Document court : un seul chunk
            return [
                ParsedChunk(
                    content=text,
                    chunk_index=0,
                    metadata={"source": filename},
                )
            ]

        chunks: list[ParsedChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Fin du chunk candidat
            end = start + self.chunk_size

            if end >= len(text):
                # Dernier chunk : on prend tout le reste
                chunk_text = text[start:]
            else:
                # Chercher une frontière naturelle de coupure
                # Priorité : double newline > newline > point + espace > espace
                cut_point = self._find_best_cut_point(text, start, end)
                chunk_text = text[start:cut_point]
                end = cut_point

            # Ne pas créer de chunks vides ou trop courts
            if chunk_text.strip():
                chunks.append(
                    ParsedChunk(
                        content=chunk_text.strip(),
                        chunk_index=chunk_index,
                        metadata={"source": filename, "char_start": start},
                    )
                )
                chunk_index += 1

            # Avancer avec overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)

            # Sécurité : éviter les boucles infinies
            if start <= (end - self.chunk_size) and end < len(text):
                start = end

        return chunks

    def _find_best_cut_point(self, text: str, start: int, end: int) -> int:
        """
        Trouve le meilleur point de coupure dans la zone [end-200, end].

        Cherche dans l'ordre de préférence :
        1. Double saut de ligne (fin de paragraphe)
        2. Saut de ligne simple (fin de ligne)
        3. Point suivi d'un espace (fin de phrase)
        4. Espace (fin de mot)
        5. Position exacte (fallback)
        """
        # Zone de recherche : les 200 derniers caractères avant `end`
        search_start = max(start, end - 200)
        search_zone = text[search_start:end]

        # 1. Double newline (meilleur : frontière de paragraphe)
        idx = search_zone.rfind("\n\n")
        if idx != -1:
            return search_start + idx + 2

        # 2. Newline simple
        idx = search_zone.rfind("\n")
        if idx != -1:
            return search_start + idx + 1

        # 3. Point + espace (fin de phrase)
        idx = search_zone.rfind(". ")
        if idx != -1:
            return search_start + idx + 2

        # 4. Espace (fin de mot)
        idx = search_zone.rfind(" ")
        if idx != -1:
            return search_start + idx + 1

        # 5. Fallback : coupure exacte
        return end

# Vigilant Compliance Requirement Finder - hardened v11
# Adds deterministic extraction of exact normative requirements for broad standards queries.

import argparse
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher

from qdrant_client import QdrantClient


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_URL = "http://127.0.0.1:11434"

KNOWLEDGE_BASE_NAME = "Vigilant Compliance Knowledge Base"
KNOWLEDGE_BASE_ID = "vigilant_compliance_kb"

EMBED_MODEL = "embeddinggemma:latest"
CHAT_MODEL = "llama3.2:3b"

EXPECTED_EMBEDDING_DIMENSIONS = 768

CANDIDATE_K = 20
PREFERRED_CANDIDATE_K = 80
PREFERRED_SUBQUERY_K = 32
MAX_DECOMPOSED_SUBQUERIES = 5
MAX_SOURCES = 3
MAX_DETERMINISTIC_REQUIREMENTS = 3
MIN_SCORE = 0.55
PREFERRED_MIN_SCORE = 0.45

# A broad standards question (for example, "What are the NIST SP 800-171
# requirements ...?") is decomposed into a small deterministic set of
# retrieval probes. Only an explicitly named/preferred collection receives
# the extra probes; all other collections are still searched with the original
# question so multi-collection discovery remains intact.

OLLAMA_TIMEOUT_SECONDS = 600

# This host has only ~3.3 GiB of RAM. Keep the embedding and chat
# models from remaining resident at the same time. The embedding model
# unloads immediately after vector creation. Chat warm-up stays resident
# only long enough for the real generation call, which then unloads it.
EMBED_KEEP_ALIVE = 0
CHAT_WARM_KEEP_ALIVE = "5m"
CHAT_GENERATE_KEEP_ALIVE = 0

# Keep structured output compact on CPU-only inference.
NUM_PREDICT = 768
MAX_CHARS_PER_SOURCE = 2500

# Requirements tolerate small PDF/LLM wording differences.
MIN_REQUIREMENT_TOKEN_COVERAGE = 0.80
MIN_REQUIREMENT_SEQUENCE_RATIO = 0.75

# Role attribution stays strict.
MIN_RESPONSIBILITY_KEYWORD_OVERLAP = 2

# A generated requirement must be substantially grounded in
# its verified evidence, not merely share a few keywords.
MIN_REQUIREMENT_EVIDENCE_COVERAGE = 0.70

# If a generated requirement is partly grounded but broader than its
# evidence, preserve the supported portion as an evidence-backed finding
# instead of discarding the entire result. Claims below this floor are
# too weakly grounded to salvage.
MIN_REQUIREMENT_SALVAGE_COVERAGE = 0.35
MAX_SANITIZED_CLAIM_CHARS = 600

# Deterministic retrieval fallback. If the LLM fails to produce a usable
# requirement even though Qdrant returned clearly relevant passages, Python
# can preserve a conservative exact-source finding instead of returning a
# false "insufficient" result.
MIN_FALLBACK_QUESTION_OVERLAP = 3
MAX_FALLBACK_FINDINGS = 2
MAX_FALLBACK_FINDING_CHARS = 600

# Normative/actionability gate. A validated requirement must be phrased as
# an actionable obligation, or its verified evidence must contain explicit
# normative language. Descriptive topics/headings remain evidence-backed
# findings instead of being upgraded into requirements.
NORMATIVE_MODAL_PATTERNS = (
    r"\bmust\b",
    r"\bshall\b",
    r"\brequires?\b",
    r"\brequired\s+to\b",
    r"\bis\s+required\b",
    r"\bare\s+required\b",
    r"\bneed(?:s)?\s+to\b",
)

NORMATIVE_ACTION_VERBS = {
    "apply", "assess", "assign", "authorize", "configure", "conduct",
    "control", "define", "designate", "develop", "disable", "document",
    "encrypt", "enforce", "ensure", "establish", "identify", "implement",
    "limit", "maintain", "manage", "monitor", "perform", "prevent",
    "prohibit", "protect", "provide", "remove", "report", "restrict",
    "retain", "review", "secure", "share", "test", "train", "update",
    "use", "validate", "verify",
}

# Guard against generated requirements that introduce scope
# concepts not explicitly present in the verified evidence.
REQUIREMENT_SCOPE_TERMS = {
    "identity": {
        "identity", "identities", "credential", "credentials",
        "authentication", "authorization",
    },
    "device": {
        "device", "devices", "endpoint", "endpoints",
    },
    "network": {
        "network", "networks", "networking",
    },
    "application": {
        "application", "applications", "app", "apps", "software",
    },
}

REQUIREMENT_DEDUPE_SIMILARITY = 0.82


# ============================================================
# STRUCTURED OLLAMA OUTPUT SCHEMA
# ============================================================

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "direct_answer": {
            "type": "string"
        },
        "evidence_status": {
            "type": "string",
            "enum": [
                "supported",
                "partial",
                "insufficient"
            ]
        },
        "dod_requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string"
                    },
                    "explicit_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string"
                                },
                                "quote": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "source",
                                "quote"
                            ]
                        }
                    }
                },
                "required": [
                    "requirement",
                    "explicit_evidence"
                ]
            }
        },
        "roles_responsibilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string"
                    },
                    "responsibility": {
                        "type": "string"
                    },
                    "explicit_linkage_evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string"
                                },
                                "quote": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "source",
                                "quote"
                            ]
                        }
                    }
                },
                "required": [
                    "role",
                    "responsibility",
                    "explicit_linkage_evidence"
                ]
            }
        },
        "recommended_company_actions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "limitations": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "direct_answer",
        "evidence_status",
        "dod_requirements",
        "roles_responsibilities",
        "recommended_company_actions",
        "limitations"
    ]
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "will",
    "must",
    "shall",
    "should",
    "may"
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Vigilant RMF Requirement Finder - "
            "grounded multi-source DoD/NIST compliance analysis"
        )
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="RMF question to analyze"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Return machine-readable JSON only"
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip Ollama chat-model warm-up"
    )

    return parser.parse_args()


# ============================================================
# OLLAMA
# ============================================================

def ollama_post(endpoint, payload):
    url = f"{OLLAMA_URL}{endpoint}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT_SECONDS
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Ollama HTTP error {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to connect to Ollama at {url}: {exc}"
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama timed out after "
            f"{OLLAMA_TIMEOUT_SECONDS} seconds."
        ) from exc


def warm_chat_model(quiet=False):
    if not quiet:
        print(f"Loading {CHAT_MODEL}...")

    started = time.time()

    result = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": CHAT_WARM_KEEP_ALIVE,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly READY."
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 5
            }
        }
    )

    elapsed = time.time() - started

    content = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not quiet:
        print(
            f"Warm-up completed in "
            f"{elapsed:.1f} seconds."
        )

        if content:
            print(
                f"Warm-up response: {content}"
            )


def create_embeddings(texts):
    """Create one or more embeddings in a single Ollama request.

    Batching the deterministic retrieval probes avoids repeatedly loading the
    embedding model on this memory-constrained CPU host.
    """
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(text).strip() for text in texts if str(text).strip()]
    if not texts:
        raise RuntimeError("No embedding input was supplied.")

    result = ollama_post(
        "/api/embed",
        {
            "model": EMBED_MODEL,
            "input": texts,
            "keep_alive": EMBED_KEEP_ALIVE
        }
    )

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected embedding count. "
            f"Expected {len(texts)}, received {len(embeddings)}."
        )

    for vector in embeddings:
        if len(vector) != EXPECTED_EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Embedding dimension mismatch. "
                f"Expected {EXPECTED_EMBEDDING_DIMENSIONS}, "
                f"received {len(vector)}."
            )

    return embeddings


def create_embedding(text):
    return create_embeddings([text])[0]


# ============================================================
# REQUIREMENT-AWARE RETRIEVAL RERANKING
# ============================================================

REQUIREMENT_QUERY_TERMS = {
    "requirement", "requirements", "required", "require",
    "control", "controls", "must", "shall", "compliance",
    "protect", "protecting", "implement", "implementation"
}

IMPERATIVE_REQUIREMENT_VERBS = (
    "allow", "apply", "authenticate", "authorize", "configure",
    "control", "disable", "encrypt", "enforce", "ensure",
    "establish", "identify", "implement", "limit", "maintain",
    "manage", "monitor", "prevent", "protect", "provide",
    "restrict", "review", "sanitize", "terminate", "use", "verify"
)

META_EVIDENCE_PATTERNS = (
    r"\bsection\s+\d+(?:\.\d+)*\s+(?:lists|contains|describes|provides)\b",
    r"\bsecurity requirements? (?:are|is) organized\b",
    r"\bthis section (?:lists|contains|describes|provides)\b",
    r"\btable\s+\d+\b",
    r"\billustrated in table\b",
    r"\bof paramount importance\b",
    r"\borganization of (?:this|the) (?:document|publication)\b",
    r"\bpurpose of (?:this|the) (?:document|publication|section)\b",
)


def question_requests_requirements(question):
    words = set(re.findall(r"[a-z0-9]+", normalize_text(question)))
    return bool(words.intersection(REQUIREMENT_QUERY_TERMS))


def preferred_collections_for_question(question, collection_names):
    """Resolve an explicitly named standard/document to matching collections.

    This is intentionally conservative: it activates only when the question
    contains a recognizable document identifier or a distinctive document
    title. Supporting collections remain searchable, but when the named
    collection has enough qualified evidence it owns the final LLM source set.
    """
    q = normalize_text(question or "")
    q_compact = re.sub(r"[^a-z0-9]+", " ", q).strip()

    signatures = []

    for match in re.finditer(r"\bnist\s+(?:sp\s+)?(\d{3})[-_\s]+(\d{2,3})\b", q_compact):
        signatures.append(("nist", match.group(1), match.group(2)))

    for match in re.finditer(r"\b(?:dodi?|dod\s+instruction|dod\s+manual)?\s*(\d{4})[._\s-]+(\d{2})\b", q_compact):
        signatures.append(("dod-number", match.group(1), match.group(2)))

    if "zero trust strategy" in q_compact:
        signatures.append(("title", "zero trust strategy"))

    preferred = []

    for name in collection_names:
        n = normalize_text(name or "")
        n_compact = re.sub(r"[^a-z0-9]+", " ", n).strip()

        matched = False
        for signature in signatures:
            if signature[0] == "nist":
                _, series, publication = signature
                if (
                    "nist" in n_compact
                    and series in n_compact.split()
                    and publication in n_compact.split()
                ):
                    matched = True
                    break
            elif signature[0] == "dod-number":
                _, major, minor = signature
                tokens = n_compact.split()
                if major in tokens and minor in tokens:
                    matched = True
                    break
            elif signature[0] == "title":
                if signature[1] in n_compact:
                    matched = True
                    break

        if matched:
            preferred.append(name)

    return preferred


BROAD_REQUIREMENT_PATTERNS = (
    r"\bwhat\s+are\s+the\b.*\brequirements?\b",
    r"\blist\b.*\brequirements?\b",
    r"\bsummar(?:ize|ise)\b.*\brequirements?\b",
    r"\brequirements?\s+for\b",
    r"\bsecurity\s+requirements?\b",
)


def is_broad_requirement_question(question):
    """Detect a broad request for a set of requirements/controls.

    Specific control IDs and narrow "does X require Y" questions stay on the
    normal single-query path.
    """
    normalized = normalize_text(question or "")

    # A precise control identifier usually means the user wants one narrow
    # item rather than a document-wide requirement sweep.
    if re.search(r"\b(?:ac|au|at|cm|cp|ia|ir|ma|mp|pe|pl|ps|ra|ca|sc|si|sa|sr)[-_. ]?\d+(?:\.\d+)*\b", normalized):
        return False
    if re.search(r"\b3\.\d+(?:\.\d+){1,3}\b", normalized):
        return False

    return any(re.search(pattern, normalized) for pattern in BROAD_REQUIREMENT_PATTERNS)


def _question_topic(question):
    """Keep the user's subject while removing conversational filler."""
    normalized = normalize_text(question or "")
    normalized = re.sub(r"\b(?:what|which|please|tell|show|list|summarize|summarise)\b", " ", normalized)
    normalized = re.sub(r"\b(?:are|is|the|a|an|of|for|about|regarding|requirements?|controls?)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ?.,;:")
    return normalized or normalize_text(question or "")


def build_retrieval_subqueries(question, preferred_collections):
    """Build low-cost deterministic probes for broad standards questions.

    The probes are intentionally generic rather than hard-coding a particular
    framework's family names. They look for distinct kinds of normative
    security-control language while keeping the user's subject in every query.
    """
    original = str(question).strip()
    if not preferred_collections or not is_broad_requirement_question(original):
        return [original]

    topic = _question_topic(original)
    preferred_hint = ", ".join(name.strip() for name in preferred_collections[:2])

    probes = [
        original,
        f"{preferred_hint} exact mandatory security control statements for {topic}",
        f"{preferred_hint} must shall required implement restrict protect ensure requirements for {topic}",
        f"{preferred_hint} access authentication authorization account protection requirements for {topic}",
        f"{preferred_hint} audit monitoring configuration integrity incident response requirements for {topic}",
    ]

    deduped = []
    seen = set()
    for probe in probes:
        key = normalize_text(probe)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(probe)
        if len(deduped) >= MAX_DECOMPOSED_SUBQUERIES:
            break

    return deduped


def _point_line_start(point):
    payload = point.payload or {}
    metadata = payload.get("metadata", {})
    return (metadata.get("loc", {}).get("lines", {}) or {}).get("from")


def select_diverse_points(points, limit):
    """Choose high-ranked evidence while avoiding adjacent duplicate chunks.

    The first pass prefers chunks with actual normative signals. A second pass
    fills any remaining slots with the best-ranked evidence. Adjacent chunks
    from the same document are skipped when possible so broad questions expose
    more than one control statement to the LLM.
    """
    if limit <= 0:
        return []

    selected = []
    selected_ids = set()

    def far_enough(candidate):
        collection = str((candidate.payload or {}).get("_vigilant_collection", "")).strip()
        line_start = _point_line_start(candidate)
        for existing in selected:
            existing_collection = str((existing.payload or {}).get("_vigilant_collection", "")).strip()
            if collection != existing_collection:
                continue
            existing_start = _point_line_start(existing)
            if isinstance(line_start, int) and isinstance(existing_start, int):
                if abs(line_start - existing_start) < 40:
                    return False
        return True

    passes = [
        [p for p in points if int((p.payload or {}).get("_vigilant_normative_hits", 0) or 0) > 0],
        points,
    ]

    for pool in passes:
        for point in pool:
            key = (str((point.payload or {}).get("_vigilant_collection", "")), str(point.id))
            if key in selected_ids:
                continue
            if selected and not far_enough(point):
                continue
            selected.append(point)
            selected_ids.add(key)
            if len(selected) >= limit:
                return selected

    # If proximity filtering was too strict, fill from the global ranking.
    for point in points:
        key = (str((point.payload or {}).get("_vigilant_collection", "")), str(point.id))
        if key in selected_ids:
            continue
        selected.append(point)
        selected_ids.add(key)
        if len(selected) >= limit:
            break

    return selected


def requirement_rerank_components(content):
    """Return conservative ranking bonuses/penalties for source text.

    This does not validate a requirement. It only helps actual normative
    passages outrank introductory/meta passages before the LLM sees them.
    """
    normalized = normalize_text(content)

    normative_hits = 0
    normative_hits += len(re.findall(r"\b(?:must|shall)\b", normalized)) * 2
    normative_hits += len(re.findall(r"\b(?:is|are) required to\b", normalized)) * 2
    normative_hits += len(re.findall(r"\brequires?\b", normalized))

    imperative_pattern = (
        r"(?:^|\n)\s*(?:[0-9]+(?:\.[0-9]+){1,4}\s+)?(?:"
        + "|".join(IMPERATIVE_REQUIREMENT_VERBS)
        + r")\b"
    )
    imperative_hits = len(
        re.findall(imperative_pattern, (content or "").lower(), flags=re.MULTILINE)
    )
    normative_hits += imperative_hits * 3

    meta_hits = sum(
        len(re.findall(pattern, normalized))
        for pattern in META_EVIDENCE_PATTERNS
    )

    # Keep semantic similarity dominant. Heuristics may move a real control
    # above nearby front matter, but cannot rescue a clearly irrelevant chunk.
    requirement_bonus = min(0.12, normative_hits * 0.025)
    meta_penalty = min(0.10, meta_hits * 0.04)

    return requirement_bonus, meta_penalty, normative_hits, meta_hits


def rerank_qdrant_points(points, question):
    requirement_mode = question_requests_requirements(question)

    for point in points:
        payload = dict(point.payload or {})
        content = str(payload.get("content", ""))
        semantic_score = float(point.score or 0.0)

        if requirement_mode:
            bonus, penalty, normative_hits, meta_hits = (
                requirement_rerank_components(content)
            )
        else:
            bonus = penalty = 0.0
            normative_hits = meta_hits = 0

        query_match_count = int(payload.get("_vigilant_query_match_count", 1) or 1)
        multi_query_bonus = min(0.045, max(0, query_match_count - 1) * 0.015)
        rerank_score = semantic_score + bonus + multi_query_bonus - penalty

        payload["_vigilant_rerank_score"] = round(rerank_score, 6)
        payload["_vigilant_requirement_bonus"] = round(bonus, 6)
        payload["_vigilant_multi_query_bonus"] = round(multi_query_bonus, 6)
        payload["_vigilant_meta_penalty"] = round(penalty, 6)
        payload["_vigilant_normative_hits"] = normative_hits
        payload["_vigilant_meta_hits"] = meta_hits
        point.payload = payload

    points.sort(
        key=lambda point: (
            float((point.payload or {}).get("_vigilant_rerank_score", point.score or 0.0)),
            float(point.score or 0.0),
        ),
        reverse=True,
    )

    return points, ("requirement_aware" if requirement_mode else "semantic")


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def search_qdrant(question):
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=30
    )

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    preferred_collections = preferred_collections_for_question(
        question,
        collection_names
    )
    preferred_normalized = {
        name.strip().lower()
        for name in preferred_collections
    }

    subqueries = build_retrieval_subqueries(
        question,
        preferred_collections
    )
    query_vectors = create_embeddings(subqueries)
    original_vector = query_vectors[0]
    decomposed_mode = len(subqueries) > 1

    # Deduplicate the same Qdrant point returned by multiple probes while
    # preserving the best semantic score and the number of probes that found it.
    point_map = {}

    searched_collections = []
    skipped_collections = []
    failed_collections = []
    qdrant_queries_executed = 0

    for collection in collections:
        collection_name = collection.name

        try:
            info = client.get_collection(
                collection_name=collection_name
            )
            vectors = info.config.params.vectors
            query_args = {}

            if isinstance(vectors, dict):
                vector_names = list(vectors.keys())
                if len(vector_names) != 1:
                    skipped_collections.append(collection_name)
                    continue
                vector_name = vector_names[0]
                vector_config = vectors[vector_name]
                query_args["using"] = vector_name
            else:
                vector_config = vectors

            if vector_config.size != len(original_vector):
                skipped_collections.append(collection_name)
                continue

            searched_collections.append(collection_name)
            is_preferred = (
                collection_name.strip().lower()
                in preferred_normalized
            )

            # Every compatible collection receives the original user query.
            # Only an explicitly named collection receives decomposition probes.
            vector_plan = [
                (
                    0,
                    subqueries[0],
                    original_vector,
                    PREFERRED_CANDIDATE_K if is_preferred else CANDIDATE_K,
                    PREFERRED_MIN_SCORE if is_preferred else MIN_SCORE,
                )
            ]

            if is_preferred and decomposed_mode:
                for query_index, (subquery, vector) in enumerate(
                    zip(subqueries[1:], query_vectors[1:]),
                    start=1
                ):
                    vector_plan.append(
                        (
                            query_index,
                            subquery,
                            vector,
                            PREFERRED_SUBQUERY_K,
                            PREFERRED_MIN_SCORE,
                        )
                    )

            for query_index, subquery, vector, limit, min_score in vector_plan:
                response = client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    **query_args
                )
                qdrant_queries_executed += 1

                for point in response.points:
                    if point.score is None or point.score < min_score:
                        continue

                    payload = dict(point.payload or {})
                    payload["_vigilant_collection"] = collection_name

                    key = (collection_name, str(point.id))
                    existing = point_map.get(key)

                    if existing is None:
                        payload["_vigilant_query_indexes"] = [query_index]
                        payload["_vigilant_query_texts"] = [subquery]
                        payload["_vigilant_query_match_count"] = 1
                        point.payload = payload
                        point_map[key] = point
                        continue

                    existing_payload = dict(existing.payload or {})
                    indexes = list(existing_payload.get("_vigilant_query_indexes", []))
                    texts = list(existing_payload.get("_vigilant_query_texts", []))
                    if query_index not in indexes:
                        indexes.append(query_index)
                        texts.append(subquery)
                    existing_payload["_vigilant_query_indexes"] = indexes
                    existing_payload["_vigilant_query_texts"] = texts
                    existing_payload["_vigilant_query_match_count"] = len(indexes)

                    # Retain the strongest semantic score observed for this point.
                    if float(point.score or 0.0) > float(existing.score or 0.0):
                        existing.score = point.score
                    existing.payload = existing_payload

        except Exception as exc:
            failed_collections.append(collection_name)
            print(
                "Qdrant collection warning: "
                f"{collection_name!r}: {exc}",
                file=sys.stderr
            )

    all_points = list(point_map.values())
    all_points, ranking_mode = rerank_qdrant_points(
        all_points,
        question
    )

    qualified_collections = sorted({
        str((point.payload or {}).get("_vigilant_collection", "")).strip()
        for point in all_points
        if str((point.payload or {}).get("_vigilant_collection", "")).strip()
    })

    preferred_points = [
        point
        for point in all_points
        if str((point.payload or {}).get("_vigilant_collection", "")).strip().lower()
        in preferred_normalized
    ]

    if preferred_collections and len(preferred_points) >= MAX_SOURCES:
        if decomposed_mode:
            selected_points = select_diverse_points(
                preferred_points,
                MAX_SOURCES
            )
            ranking_mode = "decomposed_document_aware_" + ranking_mode
        else:
            selected_points = preferred_points[:MAX_SOURCES]
            ranking_mode = "document_aware_" + ranking_mode
    elif preferred_collections and preferred_points:
        selected_points = list(preferred_points)
        selected_ids = {
            (str((point.payload or {}).get("_vigilant_collection", "")), str(point.id))
            for point in selected_points
        }
        for point in all_points:
            key = (
                str((point.payload or {}).get("_vigilant_collection", "")),
                str(point.id)
            )
            if key in selected_ids:
                continue
            selected_points.append(point)
            selected_ids.add(key)
            if len(selected_points) >= MAX_SOURCES:
                break
        ranking_mode = "document_aware_mixed_" + ranking_mode
    else:
        selected_points = all_points[:MAX_SOURCES]

    represented_collections = sorted({
        str((point.payload or {}).get("_vigilant_collection", "")).strip()
        for point in selected_points
        if str((point.payload or {}).get("_vigilant_collection", "")).strip()
    })

    retrieval_stats = {
        "query_mode": "decomposed" if decomposed_mode else "single",
        "subqueries_executed": len(subqueries),
        "subqueries": subqueries,
        "qdrant_queries_executed": qdrant_queries_executed,
        "ranking_mode": ranking_mode,
        "rerank_candidates": len(all_points),
        "llm_source_limit": MAX_SOURCES,
        "preferred_collections": [
            name.strip()
            for name in preferred_collections
        ],
        "preferred_candidates": len(preferred_points),
        "preferred_candidate_limit": PREFERRED_CANDIDATE_K,
        "preferred_subquery_candidate_limit": PREFERRED_SUBQUERY_K,
        "preferred_minimum_score": PREFERRED_MIN_SCORE,
        "collections_discovered": len(collections),
        "collections_searched": len(searched_collections),
        "collections_with_qualified_hits": len(qualified_collections),
        "collections_represented": represented_collections,
        "collections_skipped": len(skipped_collections),
        "collections_failed": len(failed_collections),
    }

    return selected_points, retrieval_stats


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_source(point, number):
    payload = point.payload or {}

    metadata = payload.get(
        "metadata",
        {}
    )

    lines = (
        metadata
        .get("loc", {})
        .get("lines", {})
    )

    vigilant = payload.get(
        "vigilant",
        {}
    )

    collection_name = str(
        payload.get(
            "_vigilant_collection",
            ""
        )
    ).strip()

    pdf_title = (
        metadata
        .get("pdf", {})
        .get("info", {})
        .get("Title")
    )

    document = (
        vigilant.get("document")
        or pdf_title
        or collection_name
        or KNOWLEDGE_BASE_NAME
    )

    document_id = (
        vigilant.get("document_id")
        or collection_name
        or KNOWLEDGE_BASE_ID
    )

    return {
        "source_number": number,
        "label": f"SOURCE {number}",
        "collection": collection_name,
        "document": document,
        "document_id": document_id,
        "point_id": str(point.id),
        "score": round(
            float(point.score),
            4
        ),
        "retrieval_rank_score": round(
            float(payload.get("_vigilant_rerank_score", point.score or 0.0)),
            4
        ),
        "requirement_signal_bonus": round(
            float(payload.get("_vigilant_requirement_bonus", 0.0)),
            4
        ),
        "multi_query_bonus": round(
            float(payload.get("_vigilant_multi_query_bonus", 0.0)),
            4
        ),
        "query_match_count": int(
            payload.get("_vigilant_query_match_count", 1) or 1
        ),
        "query_indexes": list(
            payload.get("_vigilant_query_indexes", [0])
        ),
        "meta_evidence_penalty": round(
            float(payload.get("_vigilant_meta_penalty", 0.0)),
            4
        ),
        "line_start": lines.get("from"),
        "line_end": lines.get("to"),
        "content": payload.get(
            "content",
            ""
        )
    }

def build_sources(points):
    return [
        extract_source(
            point,
            number
        )
        for number, point in enumerate(
            points,
            start=1
        )
    ]

def build_llm_context(sources):
    parts = []

    for source in sources:
        content = (
            source["content"]
            [:MAX_CHARS_PER_SOURCE]
        )

        parts.append(
            f"[{source['label']}]\n"
            f"Document: {source['document']}\n"
            f"Collection: {source['collection']}\n"
            f"Similarity Score: "
            f"{source['score']}\n\n"
            f"{content}"
        )

    return "\n\n".join(parts)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = text or ""

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-"
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    # Repair common PDF line-break hyphenation.
    text = re.sub(
        r"-\s+",
        "",
        text
    )

    return (
        re.sub(
            r"\s+",
            " ",
            text
        )
        .strip()
        .lower()
    )


def simple_stem(word):
    word = word.lower()

    if (
        len(word) > 6
        and word.endswith("ing")
    ):
        return word[:-3]

    if (
        len(word) > 5
        and word.endswith("ed")
    ):
        return word[:-2]

    if (
        len(word) > 4
        and word.endswith("es")
    ):
        return word[:-2]

    if (
        len(word) > 3
        and word.endswith("s")
    ):
        return word[:-1]

    return word


def significant_tokens(text):
    words = re.findall(
        r"[a-z0-9]+",
        normalize_text(text)
    )

    return {
        simple_stem(word)
        for word in words
        if (
            word not in STOPWORDS
            and len(word) >= 3
        )
    }


def keyword_overlap(
    left_text,
    right_text
):
    return len(
        significant_tokens(
            left_text
        ).intersection(
            significant_tokens(
                right_text
            )
        )
    )


# ============================================================
# SOURCE VERIFICATION
# ============================================================

def source_map(sources):
    return {
        source["label"]: source
        for source in sources
    }


def quote_exists_in_source(
    quote,
    source_label,
    sources
):
    """
    Strict normalized substring check.

    This remains the standard for role attribution.
    """

    source_lookup = source_map(
        sources
    )

    source = source_lookup.get(
        source_label
    )

    if source is None:
        return False

    normalized_quote = normalize_text(
        quote
    )

    normalized_source = normalize_text(
        source["content"]
    )

    if not normalized_quote:
        return False

    return (
        normalized_quote
        in normalized_source
    )


def role_appears_in_evidence(
    role,
    quote
):
    """
    Require the claimed role to appear explicitly
    in the evidence quote.
    """

    normalized_role = normalize_text(
        role
    )

    normalized_quote = normalize_text(
        quote
    )

    if not normalized_role:
        return False

    if not normalized_quote:
        return False

    return (
        normalized_role
        in normalized_quote
    )


# ============================================================
# FUZZY REQUIREMENT MATCHING
# ============================================================

def token_coverage(
    quote,
    candidate_text
):
    quote_tokens = significant_tokens(
        quote
    )

    candidate_tokens = significant_tokens(
        candidate_text
    )

    if not quote_tokens:
        return 0.0

    matched = quote_tokens.intersection(
        candidate_tokens
    )

    return (
        len(matched)
        / len(quote_tokens)
    )


def candidate_windows(
    source_text,
    quote_text
):
    """
    Build local source windows approximately the
    same size as the proposed evidence quote.
    """

    normalized_source = normalize_text(
        source_text
    )

    normalized_quote = normalize_text(
        quote_text
    )

    source_words = normalized_source.split()
    quote_words = normalized_quote.split()

    if not source_words or not quote_words:
        return []

    quote_size = len(quote_words)

    sizes = sorted(
        {
            max(
                4,
                int(
                    quote_size
                    * 0.80
                )
            ),
            quote_size,
            max(
                4,
                int(
                    quote_size
                    * 1.20
                )
            )
        }
    )

    windows = []

    for size in sizes:
        if size >= len(source_words):
            windows.append(
                " ".join(
                    source_words
                )
            )
            continue

        step = max(
            1,
            quote_size // 5
        )

        last_start = (
            len(source_words)
            - size
        )

        starts = list(
            range(
                0,
                last_start + 1,
                step
            )
        )

        if (
            starts
            and starts[-1]
            != last_start
        ):
            starts.append(
                last_start
            )

        for start in starts:
            window = " ".join(
                source_words[
                    start:start + size
                ]
            )

            windows.append(
                window
            )

    return windows


def fuzzy_quote_match(
    quote,
    source_label,
    sources
):
    """
    Requirement evidence validation.

    Exact normalized matches pass immediately.

    Otherwise, Python finds the best local source
    window and requires both:
      - token coverage >= 0.80
      - sequence similarity >= 0.75
    """

    lookup = source_map(
        sources
    )

    source = lookup.get(
        source_label
    )

    if source is None:
        return {
            "matched": False,
            "validation_method": (
                "source_not_found"
            ),
            "token_coverage": 0.0,
            "sequence_similarity": 0.0,
            "matched_text": ""
        }

    normalized_quote = normalize_text(
        quote
    )

    normalized_source = normalize_text(
        source["content"]
    )

    if not normalized_quote:
        return {
            "matched": False,
            "validation_method": (
                "empty_quote"
            ),
            "token_coverage": 0.0,
            "sequence_similarity": 0.0,
            "matched_text": ""
        }

    if (
        normalized_quote
        in normalized_source
    ):
        return {
            "matched": True,
            "validation_method": (
                "exact_normalized_match"
            ),
            "token_coverage": 1.0,
            "sequence_similarity": 1.0,
            "matched_text": quote
        }

    best_window = ""
    best_ratio = 0.0
    best_coverage = 0.0

    for window in candidate_windows(
        source["content"],
        quote
    ):
        coverage = token_coverage(
            quote,
            window
        )

        ratio = SequenceMatcher(
            None,
            normalized_quote,
            window
        ).ratio()

        if (
            coverage + ratio
            > best_coverage + best_ratio
        ):
            best_window = window
            best_ratio = ratio
            best_coverage = coverage

    matched = (
        best_coverage
        >= MIN_REQUIREMENT_TOKEN_COVERAGE
        and best_ratio
        >= MIN_REQUIREMENT_SEQUENCE_RATIO
    )

    return {
        "matched": matched,
        "validation_method": (
            "fuzzy_source_match"
            if matched
            else "no_sufficient_match"
        ),
        "token_coverage": round(
            best_coverage,
            4
        ),
        "sequence_similarity": round(
            best_ratio,
            4
        ),
        "matched_text": best_window
    }


# ============================================================
# REQUIREMENT CLAIM HARDENING
# ============================================================

def normalized_word_tokens(text):
    return set(
        re.findall(
            r"[a-z0-9]+",
            normalize_text(text)
        )
    )


def requirement_evidence_coverage(
    requirement,
    evidence_text
):
    """
    Measure how much of the generated requirement is
    explicitly represented in its verified evidence.

    This is intentionally directional: requirement -> evidence.
    A short evidence quote cannot validate a much broader claim
    merely because a few keywords overlap.
    """

    requirement_tokens = significant_tokens(
        requirement
    )

    evidence_tokens = significant_tokens(
        evidence_text
    )

    if not requirement_tokens:
        return (
            0.0,
            [],
            []
        )

    matched_tokens = sorted(
        requirement_tokens.intersection(
            evidence_tokens
        )
    )

    unsupported_tokens = sorted(
        requirement_tokens.difference(
            evidence_tokens
        )
    )

    coverage = (
        len(matched_tokens)
        / len(requirement_tokens)
    )

    return (
        round(coverage, 4),
        matched_tokens,
        unsupported_tokens
    )


def unsupported_requirement_scopes(
    requirement,
    evidence_text
):
    requirement_tokens = normalized_word_tokens(
        requirement
    )

    evidence_tokens = normalized_word_tokens(
        evidence_text
    )

    unsupported = []

    for scope, terms in (
        REQUIREMENT_SCOPE_TERMS.items()
    ):
        if not requirement_tokens.intersection(
            terms
        ):
            continue

        if not evidence_tokens.intersection(
            terms
        ):
            unsupported.append(
                scope
            )

    return unsupported


def starts_with_normative_action(text):
    """Return True when a claim begins with an imperative-style action verb."""

    match = re.match(
        r"^\s*([a-zA-Z]+)",
        text or ""
    )

    if not match:
        return False

    return (
        match.group(1).lower()
        in NORMATIVE_ACTION_VERBS
    )


def contains_normative_modal(text):
    normalized = normalize_text(
        text or ""
    )

    return any(
        re.search(pattern, normalized)
        for pattern in NORMATIVE_MODAL_PATTERNS
    )


def is_normative_requirement(
    requirement,
    verified_evidence
):
    """Determine whether verified source text establishes a formal requirement.

    V7 deliberately does NOT let generated wording create normativity. A model
    can paraphrase descriptive source text as an imperative (for example,
    "Implement controls...") even when the source only says that a section
    lists requirements. Formal requirement status therefore requires the
    VERIFIED SOURCE EVIDENCE itself to contain normative/modal language or an
    imperative-style action statement. Otherwise the claim is retained only as
    an evidence-backed finding.
    """

    del requirement  # Normativity must come from evidence, not model wording.

    for evidence in verified_evidence:
        evidence_text = (
            evidence.get(
                "matched_source_text",
                ""
            )
            or evidence.get(
                "quote",
                ""
            )
        )

        if contains_normative_modal(
            evidence_text
        ):
            return True

        if starts_with_normative_action(
            evidence_text
        ):
            return True

    return False


def build_non_normative_finding(
    original_requirement,
    verified_evidence
):
    """Downgrade a descriptive claim to a traceable evidence-backed finding."""

    best = max(
        verified_evidence,
        key=lambda evidence: (
            float(
                evidence.get(
                    "requirement_evidence_coverage",
                    0.0
                )
                or 0.0
            ),
            int(
                evidence.get(
                    "requirement_keyword_overlap",
                    0
                )
                or 0
            )
        )
    )

    evidence_text = (
        best.get(
            "matched_source_text",
            ""
        )
        or best.get(
            "quote",
            ""
        )
    )

    sanitized = build_sanitized_requirement(
        original_requirement=(
            original_requirement
        ),
        evidence_text=evidence_text,
        source_label=best.get(
            "source",
            ""
        ),
        coverage=float(
            best.get(
                "requirement_evidence_coverage",
                0.0
            )
            or 0.0
        ),
        unsupported_scopes=[],
        unsupported_terms=[]
    )

    sanitized[
        "validation_state"
    ] = "non_normative_evidence_finding"

    sanitized[
        "finding_reason"
    ] = (
        "The source supports this topic/finding, "
        "but the verified text does not establish "
        "a normative or actionable requirement."
    )

    sanitized["sources"] = list(
        dict.fromkeys(
            evidence.get(
                "source",
                ""
            )
            for evidence in verified_evidence
            if evidence.get(
                "source",
                ""
            )
        )
    )

    finding_evidence = []

    for evidence in verified_evidence:
        record = dict(evidence)
        record[
            "validation_method"
        ] = (
            str(
                record.get(
                    "validation_method",
                    ""
                )
            )
            + "_non_normative_finding"
        )
        finding_evidence.append(record)

    sanitized[
        "verified_evidence"
    ] = finding_evidence

    return sanitized


def requirement_evidence_signature(item):
    evidence = item.get(
        "verified_evidence",
        []
    )

    parts = []

    for entry in evidence:
        source = str(
            entry.get(
                "source",
                ""
            )
        ).strip()

        matched_text = (
            entry.get(
                "matched_source_text",
                ""
            )
            or entry.get(
                "quote",
                ""
            )
        )

        parts.append(
            f"{source}|{normalize_text(matched_text)}"
        )

    return tuple(sorted(parts))


def requirement_similarity(left, right):
    return SequenceMatcher(
        None,
        normalize_text(left),
        normalize_text(right)
    ).ratio()


def deduplicate_requirements(requirements):
    deduplicated = []
    duplicates = []

    for candidate in requirements:
        candidate_signature = (
            requirement_evidence_signature(
                candidate
            )
        )

        duplicate_of = None

        for existing in deduplicated:
            same_evidence = bool(
                candidate_signature
            ) and (
                candidate_signature
                == requirement_evidence_signature(
                    existing
                )
            )

            similar_claim = (
                requirement_similarity(
                    candidate.get(
                        "requirement",
                        ""
                    ),
                    existing.get(
                        "requirement",
                        ""
                    )
                )
                >= REQUIREMENT_DEDUPE_SIMILARITY
            )

            if same_evidence and similar_claim:
                duplicate_of = existing
                break

        if duplicate_of is None:
            deduplicated.append(candidate)
        else:
            duplicates.append({
                "requirement": candidate.get(
                    "requirement",
                    ""
                ),
                "reason": (
                    "Duplicate requirement claim "
                    "supported by the same evidence "
                    "as another validated claim."
                ),
            })

    return (
        deduplicated,
        duplicates
    )


def build_validated_direct_answer(
    requirements,
    roles
):
    """Build the headline answer only from Python-validated claims.

    A sanitized requirement is intentionally labeled as an evidence-backed
    finding so the final answer never upgrades descriptive source language
    into a stronger implementation mandate.
    """

    if requirements:
        claims = []
        for item in requirements:
            claim = str(item.get("requirement", "")).strip()
            if not claim:
                continue
            claim = re.sub(
                r"^Evidence-backed finding:\s*",
                "",
                claim,
                flags=re.IGNORECASE
            ).strip()
            claims.append(claim)

        if len(claims) == 1:
            return claims[0]

        if claims:
            if all(
                item.get("validation_state")
                in {
                    "sanitized_from_evidence",
                    "non_normative_evidence_finding",
                    "fallback_from_retrieved_evidence",
                }
                for item in requirements
            ):
                return (
                    "Evidence-backed findings: "
                    + "; ".join(claims)
                )

            return (
                "Validated findings: "
                + "; ".join(claims)
            )

    if roles:
        statements = []

        for item in roles:
            role = str(
                item.get(
                    "role",
                    ""
                )
            ).strip()

            responsibility = str(
                item.get(
                    "responsibility",
                    ""
                )
            ).strip()

            if role and responsibility:
                statements.append(
                    f"{role}: {responsibility}"
                )

        if len(statements) == 1:
            return statements[0]

        if statements:
            return (
                "Validated roles and responsibilities: "
                + "; ".join(statements)
            )

    return (
        "The retrieved evidence is insufficient "
        "to establish a validated answer to this "
        "question."
    )


def build_sanitized_requirement(
    original_requirement,
    evidence_text,
    source_label,
    coverage,
    unsupported_scopes=None,
    unsupported_terms=None
):
    """Create a conservative finding directly from verified evidence.

    This deliberately avoids trying to repair the LLM's grammar by deleting
    individual words. The verified source text becomes the claim, which is
    safer and fully traceable.
    """

    clean_text = re.sub(
        r"\s+",
        " ",
        evidence_text or ""
    ).strip()

    if len(clean_text) > MAX_SANITIZED_CLAIM_CHARS:
        truncated = clean_text[
            :MAX_SANITIZED_CLAIM_CHARS
        ]
        if " " in truncated:
            truncated = truncated.rsplit(
                " ",
                1
            )[0]
        clean_text = truncated + "..."

    return {
        "requirement": (
            "Evidence-backed finding: "
            + clean_text
        ),
        "original_requirement": (
            original_requirement
        ),
        "validation_state": (
            "sanitized_from_evidence"
        ),
        "sanitized_from_source": (
            source_label
        ),
        "sanitization": {
            "requirement_evidence_coverage": (
                coverage
            ),
            "unsupported_scopes_removed": sorted(
                set(unsupported_scopes or [])
            ),
            "unsupported_terms_removed": sorted(
                set(unsupported_terms or [])
            )[:20]
        }
    }


# ============================================================
# REQUIREMENT VALIDATION
# ============================================================

def validate_requirements(
    requirements,
    sources
):
    validated = []
    rejected = []

    if not isinstance(
        requirements,
        list
    ):
        return (
            validated,
            rejected
        )

    for item in requirements:
        if not isinstance(
            item,
            dict
        ):
            continue

        requirement = (
            item
            .get(
                "requirement",
                ""
            )
            .strip()
        )

        evidence_items = item.get(
            "explicit_evidence",
            []
        )

        if (
            not requirement
            or not isinstance(
                evidence_items,
                list
            )
        ):
            rejected.append({
                "requirement": requirement,
                "reason": (
                    "Missing requirement text "
                    "or explicit evidence."
                )
            })
            continue

        verified_evidence = []
        salvage_candidates = []
        unsupported_scopes = set()
        low_coverage_evidence = []

        for evidence in evidence_items:
            if not isinstance(
                evidence,
                dict
            ):
                continue

            label = (
                evidence
                .get(
                    "source",
                    ""
                )
                .strip()
            )

            quote = (
                evidence
                .get(
                    "quote",
                    ""
                )
                .strip()
            )

            if not label or not quote:
                continue

            match = fuzzy_quote_match(
                quote,
                label,
                sources
            )

            if not match["matched"]:
                continue

            matched_text = (
                match["matched_text"]
                or quote
            )

            requirement_overlap = keyword_overlap(
                requirement,
                matched_text
            )

            if requirement_overlap < 2:
                continue

            (
                claim_coverage,
                matched_claim_tokens,
                unsupported_claim_tokens
            ) = requirement_evidence_coverage(
                requirement,
                matched_text
            )

            scope_failures = (
                unsupported_requirement_scopes(
                    requirement,
                    matched_text
                )
            )

            evidence_record = {
                "source": label,
                "quote": quote,
                "validation_method": (
                    match["validation_method"]
                ),
                "token_coverage": (
                    match["token_coverage"]
                ),
                "sequence_similarity": (
                    match["sequence_similarity"]
                ),
                "requirement_keyword_overlap": (
                    requirement_overlap
                ),
                "requirement_evidence_coverage": (
                    claim_coverage
                ),
                "matched_requirement_tokens": (
                    matched_claim_tokens
                ),
                "unsupported_requirement_tokens": (
                    unsupported_claim_tokens
                ),
                "matched_source_text": (
                    match["matched_text"]
                )
            }

            if (
                not scope_failures
                and claim_coverage
                >= MIN_REQUIREMENT_EVIDENCE_COVERAGE
            ):
                verified_evidence.append(
                    evidence_record
                )
                continue

            if scope_failures:
                unsupported_scopes.update(
                    scope_failures
                )

            if (
                claim_coverage
                >= MIN_REQUIREMENT_SALVAGE_COVERAGE
            ):
                salvage_candidates.append({
                    "source": label,
                    "coverage": claim_coverage,
                    "scope_failures": scope_failures,
                    "unsupported_terms": (
                        unsupported_claim_tokens
                    ),
                    "matched_text": matched_text,
                    "evidence_record": evidence_record
                })
            else:
                low_coverage_evidence.append({
                    "source": label,
                    "coverage": claim_coverage,
                    "unsupported_terms": (
                        unsupported_claim_tokens
                    )
                })

        if verified_evidence:
            if is_normative_requirement(
                requirement,
                verified_evidence
            ):
                validated.append({
                    "requirement": requirement,
                    "validation_state": (
                        "validated"
                    ),
                    "sources": list(
                        dict.fromkeys(
                            evidence["source"]
                            for evidence
                            in verified_evidence
                        )
                    ),
                    "verified_evidence": (
                        verified_evidence
                    )
                })
            else:
                validated.append(
                    build_non_normative_finding(
                        requirement,
                        verified_evidence
                    )
                )
            continue

        # V4 salvage path: if the generated requirement was broader than
        # the evidence but still had meaningful verified overlap, preserve
        # the evidence itself as a conservative finding rather than losing
        # the entire answer.
        if salvage_candidates:
            best = max(
                salvage_candidates,
                key=lambda candidate: (
                    candidate["coverage"],
                    candidate["evidence_record"][
                        "requirement_keyword_overlap"
                    ]
                )
            )

            sanitized = build_sanitized_requirement(
                original_requirement=requirement,
                evidence_text=best["matched_text"],
                source_label=best["source"],
                coverage=best["coverage"],
                unsupported_scopes=(
                    best["scope_failures"]
                ),
                unsupported_terms=(
                    best["unsupported_terms"]
                )
            )

            sanitized["sources"] = [
                best["source"]
            ]

            sanitized_evidence = dict(
                best["evidence_record"]
            )
            sanitized_evidence[
                "validation_method"
            ] = (
                sanitized_evidence[
                    "validation_method"
                ]
                + "_sanitized"
            )
            sanitized_evidence[
                "sanitized_from_original_requirement"
            ] = True

            sanitized["verified_evidence"] = [
                sanitized_evidence
            ]

            validated.append(
                sanitized
            )
            continue

        if unsupported_scopes:
            reason = (
                "The requirement introduced "
                "scope term(s) not explicitly "
                "supported by its verified "
                "evidence: "
                + ", ".join(
                    sorted(unsupported_scopes)
                )
                + "."
            )
        elif low_coverage_evidence:
            best = max(
                low_coverage_evidence,
                key=lambda candidate: (
                    candidate["coverage"]
                )
            )

            unsupported_terms = best.get(
                "unsupported_terms",
                []
            )

            reason = (
                "The requirement was broader than "
                "the verified evidence. Best "
                "requirement-to-evidence coverage "
                f"was {best['coverage']:.2f}; "
                f"minimum for full validation is "
                f"{MIN_REQUIREMENT_EVIDENCE_COVERAGE:.2f} "
                f"and minimum for evidence salvage is "
                f"{MIN_REQUIREMENT_SALVAGE_COVERAGE:.2f}."
            )

            if unsupported_terms:
                reason += (
                    " Unsupported requirement "
                    "term(s): "
                    + ", ".join(
                        unsupported_terms[:12]
                    )
                    + "."
                )
        else:
            reason = (
                "No source evidence passed "
                "exact or fuzzy validation "
                "with sufficient requirement "
                "keyword overlap."
            )

        rejected.append({
            "requirement": requirement,
            "reason": reason
        })

    (
        validated,
        duplicate_rejections
    ) = deduplicate_requirements(
        validated
    )

    rejected.extend(
        duplicate_rejections
    )

    return (
        validated,
        rejected
    )


# ============================================================
# ROLE VALIDATION - STRICT
# ============================================================

def validate_roles(
    roles,
    sources
):
    validated = []
    rejected = []

    if not isinstance(
        roles,
        list
    ):
        return (
            validated,
            rejected
        )

    for item in roles:
        if not isinstance(
            item,
            dict
        ):
            continue

        role = (
            item
            .get(
                "role",
                ""
            )
            .strip()
        )

        responsibility = (
            item
            .get(
                "responsibility",
                ""
            )
            .strip()
        )

        evidence_items = item.get(
            "explicit_linkage_evidence",
            []
        )

        if (
            not role
            or not responsibility
            or not isinstance(
                evidence_items,
                list
            )
        ):
            rejected.append(
                {
                    "role": role,
                    "reason": (
                        "Missing role, responsibility, "
                        "or explicit linkage evidence."
                    )
                }
            )
            continue

        verified_evidence = []

        for evidence in evidence_items:
            if not isinstance(
                evidence,
                dict
            ):
                continue

            label = (
                evidence
                .get(
                    "source",
                    ""
                )
                .strip()
            )

            quote = (
                evidence
                .get(
                    "quote",
                    ""
                )
                .strip()
            )

            if not label or not quote:
                continue

            quote_verified = (
                quote_exists_in_source(
                    quote,
                    label,
                    sources
                )
            )

            role_verified = (
                role_appears_in_evidence(
                    role,
                    quote
                )
            )

            if (
                not quote_verified
                or not role_verified
            ):
                continue

            overlap = keyword_overlap(
                responsibility,
                quote
            )

            if (
                overlap
                < MIN_RESPONSIBILITY_KEYWORD_OVERLAP
            ):
                continue

            verified_evidence.append(
                {
                    "source": label,
                    "quote": quote,
                    "validation_method": (
                        "strict_role_match"
                    ),
                    "responsibility_keyword_overlap": (
                        overlap
                    )
                }
            )

        if not verified_evidence:
            rejected.append(
                {
                    "role": role,
                    "reason": (
                        "The claimed role/responsibility "
                        "could not be explicitly verified. "
                        "The evidence must exist in the "
                        "retrieved source, explicitly name "
                        "the claimed role, and support the "
                        "claimed responsibility."
                    )
                }
            )
            continue

        validated.append(
            {
                "role": role,
                "responsibility": responsibility,
                "sources": list(
                    dict.fromkeys(
                        evidence["source"]
                        for evidence
                        in verified_evidence
                    )
                ),
                "verified_linkage_evidence": (
                    verified_evidence
                )
            }
        )

    return (
        validated,
        rejected
    )


# ============================================================
# RETRIEVAL FALLBACK
# ============================================================

def evidence_sentences(text):
    """Return conservative sentence-like units from retrieved source text."""

    cleaned = re.sub(
        r"[\t\r]+",
        " ",
        text or ""
    )

    # Preserve PDF line boundaries as potential finding boundaries while also
    # honoring ordinary sentence punctuation.
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        cleaned
    )

    sentences = []

    for part in parts:
        sentence = re.sub(
            r"\s+",
            " ",
            part
        ).strip(" -\u2022\t")

        if len(sentence) < 35:
            continue

        if len(sentence) > MAX_FALLBACK_FINDING_CHARS:
            truncated = sentence[:MAX_FALLBACK_FINDING_CHARS]
            if " " in truncated:
                truncated = truncated.rsplit(" ", 1)[0]
            sentence = truncated + "..."

        sentences.append(sentence)

    return sentences


def _strip_requirement_prefix(text):
    """Remove bullets/section numbers only for normativity detection."""
    value = str(text or "").strip()
    value = re.sub(r"^\s*[\-\u2022*]+\s*", "", value)
    value = re.sub(r"^\s*\(?[a-z0-9]+\)?[.)]\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*[0-9]+(?:\.[0-9]+){1,6}\s+", "", value)
    return value.strip()


def _is_complete_evidence_sentence(text):
    normalized = normalize_text(text or "")
    if len(normalized) < 30:
        return False
    if re.search(r"\b(?:for|to|and|or|of|in|with|that|which|the|a|an)\s*$", normalized):
        return False
    if str(text).rstrip().endswith((":", ";", ",", "(")):
        return False
    return True


def _looks_like_meta_evidence(text):
    normalized = normalize_text(text or "")
    return any(re.search(pattern, normalized) for pattern in META_EVIDENCE_PATTERNS)


def build_deterministic_normative_requirements(question, sources, limit=MAX_DETERMINISTIC_REQUIREMENTS):
    """Extract exact normative source statements for broad requirement questions.

    This path does not ask the LLM to paraphrase. It preserves complete,
    normative source sentences from the already-selected evidence chunks.
    It is intentionally limited to broad requirement questions and returns
    at most a few representative controls, so the final status remains PARTIAL.
    """
    if not question or not is_broad_requirement_question(question):
        return []
    if not question_requests_requirements(question):
        return []

    question_tokens = significant_tokens(question)
    candidates = []

    for source in sources:
        label = source.get("label", "")
        content = source.get("content", "")
        source_score = float(source.get("retrieval_rank_score", source.get("score", 0.0)) or 0.0)
        query_match_count = int(source.get("query_match_count", 1) or 1)

        for sentence in evidence_sentences(content):
            if not _is_complete_evidence_sentence(sentence):
                continue
            if _looks_like_meta_evidence(sentence):
                continue

            action_text = _strip_requirement_prefix(sentence)
            normative = contains_normative_modal(sentence) or starts_with_normative_action(action_text)
            if not normative:
                continue

            sentence_tokens = significant_tokens(sentence)
            overlap_tokens = sorted(question_tokens.intersection(sentence_tokens))
            overlap = len(overlap_tokens)

            candidates.append({
                "source": label,
                "sentence": sentence,
                "source_score": source_score,
                "query_match_count": query_match_count,
                "overlap": overlap,
                "overlap_tokens": overlap_tokens,
            })

    candidates.sort(
        key=lambda item: (
            item["query_match_count"],
            item["source_score"],
            item["overlap"],
            -len(item["sentence"]),
        ),
        reverse=True,
    )

    selected = []
    for candidate in candidates:
        normalized = normalize_text(candidate["sentence"])
        duplicate = False
        for existing in selected:
            ratio = SequenceMatcher(None, normalized, normalize_text(existing["sentence"])).ratio()
            if ratio >= 0.88:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break

    requirements = []
    for candidate in selected:
        sentence = candidate["sentence"]
        label = candidate["source"]
        requirements.append({
            "requirement": sentence,
            "validation_state": "deterministic_normative_source",
            "deterministic_from_source": label,
            "sources": [label],
            "verified_evidence": [{
                "source": label,
                "quote": sentence,
                "validation_method": "deterministic_exact_normative_source",
                "token_coverage": 1.0,
                "sequence_similarity": 1.0,
                "requirement_keyword_overlap": candidate["overlap"],
                "question_overlap_tokens": candidate["overlap_tokens"],
                "matched_source_text": sentence,
            }],
        })

    return requirements


def merge_requirement_items(existing, additions, limit=MAX_DETERMINISTIC_REQUIREMENTS):
    """Merge validated and deterministic requirements without near-duplicates."""
    merged = list(existing or [])
    for item in additions or []:
        text = normalize_text(item.get("requirement", ""))
        if not text:
            continue
        duplicate = False
        for current in merged:
            current_text = normalize_text(current.get("requirement", ""))
            if not current_text:
                continue
            if SequenceMatcher(None, text, current_text).ratio() >= REQUIREMENT_DEDUPE_SIMILARITY:
                duplicate = True
                break
        if duplicate:
            continue
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def build_retrieval_fallback_requirements(
    question,
    sources
):
    """Build exact-source findings when the LLM yields no usable claims.

    This is intentionally conservative. It never invents a requirement. It
    selects sentence-like source text with strong lexical overlap to the
    user's question and marks the result as a retrieval fallback so the final
    status can remain PARTIAL rather than falsely SUPPORTED.
    """

    question_tokens = significant_tokens(
        question or ""
    )

    if not question_tokens:
        return []

    candidates = []

    for source in sources:
        label = source.get("label", "")
        content = source.get("content", "")
        source_score = float(
            source.get("score", 0.0) or 0.0
        )

        for sentence in evidence_sentences(content):
            normalized_sentence = normalize_text(sentence)

            # Do not manufacture a fallback from a document title, heading,
            # or a chunk-boundary fragment. Fallback text must stand on its
            # own as complete evidence.
            if re.match(
                r"^nist\s+sp\s+800[-_\s]*\d+\w*\s+protecting\b",
                normalized_sentence
            ) and len(sentence) < 120:
                continue

            if re.search(
                r"\b(?:for|to|and|or|of|in|with|that|which|the|a|an)\s*$",
                normalized_sentence
            ):
                continue

            if sentence.rstrip().endswith((":", ";", ",", "(")):
                continue

            sentence_tokens = significant_tokens(
                sentence
            )

            overlap_tokens = sorted(
                question_tokens.intersection(
                    sentence_tokens
                )
            )

            overlap = len(overlap_tokens)

            if overlap < MIN_FALLBACK_QUESTION_OVERLAP:
                continue

            normative = (
                contains_normative_modal(sentence)
                or starts_with_normative_action(sentence)
            )

            candidates.append({
                "source": label,
                "sentence": sentence,
                "overlap": overlap,
                "overlap_tokens": overlap_tokens,
                "source_score": source_score,
                "normative": normative,
            })

    requirement_mode = question_requests_requirements(question)

    candidates.sort(
        key=lambda item: (
            1 if (requirement_mode and item.get("normative")) else 0,
            item["overlap"],
            item["source_score"],
            len(item["sentence"]),
        ),
        reverse=True
    )

    if requirement_mode and any(item.get("normative") for item in candidates):
        candidates = [item for item in candidates if item.get("normative")]

    selected = []

    for candidate in candidates:
        duplicate = False

        for existing in selected:
            ratio = SequenceMatcher(
                None,
                normalize_text(candidate["sentence"]),
                normalize_text(existing["sentence"])
            ).ratio()

            if ratio >= 0.88:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(candidate)

        if len(selected) >= MAX_FALLBACK_FINDINGS:
            break

    findings = []

    for candidate in selected:
        sentence = candidate["sentence"]
        label = candidate["source"]

        findings.append({
            "requirement": (
                "Evidence-backed finding: "
                + sentence
            ),
            "validation_state": (
                "fallback_from_retrieved_evidence"
            ),
            "fallback_from_source": label,
            "sources": [label],
            "verified_evidence": [
                {
                    "source": label,
                    "quote": sentence,
                    "validation_method": (
                        "retrieval_fallback_exact_source"
                    ),
                    "token_coverage": 1.0,
                    "sequence_similarity": 1.0,
                    "requirement_keyword_overlap": (
                        candidate["overlap"]
                    ),
                    "question_overlap_tokens": (
                        candidate["overlap_tokens"]
                    ),
                    "matched_source_text": sentence,
                }
            ],
        })

    return findings


# ============================================================
# LLM ANALYSIS
# ============================================================

def generate_analysis(
    question,
    sources
):
    labels = [
        source["label"]
        for source in sources
    ]

    context = build_llm_context(
        sources
    )

    schema_text = json.dumps(
        ANALYSIS_SCHEMA,
        indent=2
    )

    system_prompt = f"""
You are the Vigilant Compliance Requirement Finder.

Analyze ONLY the retrieved passages supplied from the
{KNOWLEDGE_BASE_NAME}.

The retrieved passages may come from multiple authoritative DoD,
NIST, and defense-related documents. Treat each SOURCE label and
its associated document identity independently.

Do not attribute a requirement, control, responsibility, or policy
statement to a document unless the retrieved passage from that
document supports the attribution. NIST guidance must not be
represented as DoD policy unless the supplied evidence establishes
that DoD applicability.

When the user's question explicitly names a source such as NIST SP 800-171
and retrieved passages from that source directly address the question, treat
that source as in-scope. Do not reject NIST evidence merely because it is not
DoD policy. Answer from the retrieved NIST passage and identify it as NIST.

Do not copy scope terms from the user question into a
requirement unless those concepts are explicitly present in the
cited source passage. Do not turn one general passage into separate
identity, device, network, or application requirements unless each
scope is directly supported by the cited evidence.

Requirements must closely restate what the cited passage actually
says. Do not introduce architectural mechanisms, implementation
techniques, technologies, or control concepts that are absent from
the supporting evidence.

If a retrieved passage is descriptive rather than prescriptive, keep
the finding conservative and close to the source wording. Do not
convert descriptive language into an "Implement ..." mandate unless
the passage itself establishes that requirement.

A formal requirement must be normative/actionable IN THE CITED SOURCE TEXT.
Prefer source language such as must, shall, required to, requires, implement,
restrict, protect, limit, ensure, establish, maintain, or comparable
obligation/action language. Generated imperative wording does not make a
descriptive passage normative. In particular, statements such as "Section 3
lists the security requirements...", "this section describes...", or
"protection of CUI is important" are descriptive/meta statements, not the
underlying requirements. Do not rewrite those statements as "Implement..."
requirements. Treat them only as evidence-backed findings unless an actual
requirement statement is present in the cited passage.

Do not use outside knowledge as evidence.

STRICT RULES:

1. Use only the supplied passages.

2. Never invent requirements, responsibilities,
   page numbers, section numbers, paragraph numbers,
   control identifiers, citations, or role assignments.

3. Evidence labels may ONLY be:
   {", ".join(labels)}

4. Every requirement or evidence-backed finding must include at least one
   short source excerpt in explicit_evidence. This applies equally to DoD,
   NIST, and other retrieved authoritative sources. Copy the wording as
   closely as possible.

5. Every role/responsibility must include at least one
   exact source excerpt that BOTH names the role and
   connects that role to the responsibility.

6. Do not infer responsibility from nearby text.

7. If a role is not explicitly named in the supporting
   excerpt, omit that role.

8. Recommended company actions are advisory and must remain separate
   from formal source requirements or evidence-backed findings.

9. If the passages do not answer the question, set
   evidence_status to "insufficient".

10. Do not generate point IDs, line numbers,
    similarity scores, or traceability metadata.
    Python generates those from Qdrant.

For broad questions asking for multiple requirements, prefer distinct
requirement/control statements from different supplied passages rather than
repeating one topic. Do not claim completeness unless the supplied evidence
actually supports a complete enumeration.

Keep the response compact for local CPU inference:
- At most 3 requirement/finding objects.
- At most 2 role/responsibility objects.
- At most 2 recommended company actions.
- Use one short evidence excerpt per claim unless a second excerpt is essential.
- Keep direct_answer to one or two concise sentences.

11. Return only JSON conforming to this schema:

{schema_text}
"""

    user_prompt = f"""
QUESTION:

{question}

RETRIEVED EVIDENCE:

{context}
"""

    started = time.time()

    result = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": CHAT_GENERATE_KEEP_ALIVE,
            "format": ANALYSIS_SCHEMA,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": NUM_PREDICT
            }
        }
    )

    elapsed = (
        time.time()
        - started
    )

    raw = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not raw:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    try:
        parsed = json.loads(
            raw
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama did not return valid JSON.\n\n"
            f"Raw response:\n{raw}"
        ) from exc

    if not isinstance(
        parsed,
        dict
    ):
        raise RuntimeError(
            "Ollama output was not a JSON object."
        )

    return (
        parsed,
        elapsed
    )


# ============================================================
# POST-LLM VALIDATION
# ============================================================

def validate_analysis(
    analysis,
    sources,
    question=None
):
    (
        requirements,
        rejected_requirements
    ) = validate_requirements(
        analysis.get(
            "dod_requirements",
            []
        ),
        sources
    )

    (
        roles,
        rejected_roles
    ) = validate_roles(
        analysis.get(
            "roles_responsibilities",
            []
        ),
        sources
    )

    actions = analysis.get(
        "recommended_company_actions",
        []
    )

    if not isinstance(
        actions,
        list
    ):
        actions = []

    actions = [
        str(action).strip()
        for action in actions
        if str(action).strip()
    ]

    # Do not pass free-form LLM limitations through unvalidated. They can
    # contradict the retrieved source (for example, claiming a NIST document
    # does not address CUI when the retrieved passage explicitly does).
    # Limitations returned to callers are generated deterministically below.
    limitations = []

    fallback_requirement_count = 0
    deterministic_requirement_count = 0
    broad_requirement_query = bool(
        question
        and is_broad_requirement_question(question)
        and question_requests_requirements(question)
    )

    # For broad standards questions, do not depend on the small chat model to
    # restate every control correctly. Extract a few exact normative source
    # statements from the selected evidence and merge them with any claims that
    # survived ordinary validation. These are representative controls, not a
    # claim of exhaustive framework coverage.
    if broad_requirement_query and question:
        deterministic_requirements = build_deterministic_normative_requirements(
            question,
            sources,
            MAX_DETERMINISTIC_REQUIREMENTS
        )
        before_count = len(requirements)
        requirements = merge_requirement_items(
            requirements,
            deterministic_requirements,
            MAX_DETERMINISTIC_REQUIREMENTS
        )
        deterministic_requirement_count = max(0, len(requirements) - before_count)
        if deterministic_requirement_count:
            limitations.append(
                f"{deterministic_requirement_count} exact normative requirement(s) "
                "were extracted directly from retrieved source text to improve "
                "coverage of this broad standards question. The returned controls "
                "are representative, not exhaustive."
            )

    # If the LLM produced no usable claims but retrieval itself is clearly
    # relevant, preserve exact source findings rather than returning a false
    # insufficient result. This remains PARTIAL by design.
    if not requirements and not roles and question:
        fallback_requirements = (
            build_retrieval_fallback_requirements(
                question,
                sources
            )
        )

        if fallback_requirements:
            requirements = fallback_requirements
            fallback_requirement_count = len(
                fallback_requirements
            )

            limitations.append(
                f"{fallback_requirement_count} conservative "
                "finding(s) were extracted directly from "
                "retrieved source text because the LLM did "
                "not produce a usable validated claim."
            )

    sanitized_requirement_count = sum(
        1
        for item in requirements
        if item.get("validation_state")
        == "sanitized_from_evidence"
    )

    non_normative_finding_count = sum(
        1
        for item in requirements
        if item.get("validation_state")
        == "non_normative_evidence_finding"
    )

    if non_normative_finding_count:
        limitations.append(
            f"{non_normative_finding_count} descriptive claim(s) "
            "were retained as evidence-backed findings rather than "
            "formal requirements because the verified source text "
            "was not normative/actionable."
        )

    if sanitized_requirement_count:
        limitations.append(
            f"{sanitized_requirement_count} "
            "requirement claim(s) were narrowed to "
            "conservative evidence-backed findings "
            "because the generated wording was broader "
            "than the verified source text."
        )

    if rejected_requirements:
        limitations.append(
            f"{len(rejected_requirements)} "
            "requirement claim(s) were removed "
            "because Python could not verify "
            "sufficient source evidence."
        )

    if rejected_roles:
        limitations.append(
            f"{len(rejected_roles)} "
            "role/responsibility claim(s) were "
            "removed because Python could not "
            "verify an explicit role-to-"
            "responsibility linkage in the "
            "retrieved source text."
        )

    validated_claim_count = (
        len(requirements)
        + len(roles)
    )

    raw_status = analysis.get(
        "evidence_status",
        "insufficient"
    )

    # Determine support from the claims that matter most to the answer.
    # Validated requirements can remain fully supported even if an
    # unrelated generated role claim was rejected. A roles-only answer
    # is fully supported only when no requirement or role claims were
    # rejected. This prevents a mixed Zero Trust result from appearing
    # fully supported while preserving strong NIST/RMF requirement hits.
    if validated_claim_count == 0:
        evidence_status = (
            "insufficient"
        )
    elif requirements:
        evidence_status = (
            "supported"
            if (
                raw_status == "supported"
                and not rejected_requirements
                and sanitized_requirement_count == 0
                and non_normative_finding_count == 0
                and fallback_requirement_count == 0
                and deterministic_requirement_count == 0
                and not broad_requirement_query
            )
            else "partial"
        )
    else:
        evidence_status = (
            "supported"
            if (
                raw_status == "supported"
                and not rejected_requirements
                and not rejected_roles
            )
            else "partial"
        )

    # Never pass the LLM's free-form direct answer through
    # unchanged. Rebuild it exclusively from claims that
    # survived Python evidence validation.
    direct_answer = (
        build_validated_direct_answer(
            requirements,
            roles
        )
    )

    return {
        "direct_answer": direct_answer,
        "evidence_status": (
            evidence_status
        ),
        "dod_requirements": (
            requirements
        ),
        "roles_responsibilities": (
            roles
        ),
        "recommended_company_actions": (
            actions
        ),
        "limitations": (
            limitations
        ),
        "validation": {
            "requirements_validated": sum(
                1
                for item in requirements
                if item.get("validation_state") == "validated"
            ),
            "requirements_retained": len(requirements),
            "requirements_sanitized": (
                sanitized_requirement_count
            ),
            "evidence_findings": (
                non_normative_finding_count
            ),
            "requirements_fallback": (
                fallback_requirement_count
            ),
            "requirements_deterministic": (
                deterministic_requirement_count
            ),
            "broad_requirement_query": broad_requirement_query,
            "roles_validated": len(roles),
            "requirements_rejected": (
                len(
                    rejected_requirements
                )
            ),
            "role_claims_rejected": (
                len(
                    rejected_roles
                )
            ),
            "rejected_requirements": (
                rejected_requirements
            ),
            "rejected_roles": (
                rejected_roles
            )
        }
    }


# ============================================================
# INSUFFICIENT RESULT
# ============================================================

def insufficient_result(
    question,
    retrieval_stats=None
):
    return {
        "question": question,
        "document": KNOWLEDGE_BASE_NAME,
        "document_id": KNOWLEDGE_BASE_ID,
        "status": "insufficient",
        "retrieval": {
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "minimum_score": MIN_SCORE,
            "candidate_count": CANDIDATE_K,
            "candidate_limit_per_collection": CANDIDATE_K,
            "sources_used": 0,
            **(retrieval_stats or {})
        },
        "analysis": {
            "direct_answer": (
                "The retrieved evidence is "
                "insufficient to establish "
                "this requirement."
            ),
            "evidence_status": (
                "insufficient"
            ),
            "dod_requirements": [],
            "roles_responsibilities": [],
            "recommended_company_actions": [],
            "limitations": [
                (
                    "No retrieved passage met "
                    f"the {MIN_SCORE} similarity "
                    "threshold."
                )
            ],
            "validation": {
                "requirements_rejected": 0,
                "role_claims_rejected": 0,
                "rejected_requirements": [],
                "rejected_roles": []
            }
        },
        "sources": [],
        "generation_seconds": 0
    }


# ============================================================
# REPORT
# ============================================================

def build_report(
    question,
    analysis,
    sources,
    generation_seconds,
    retrieval_stats=None
):
    traceability = [
        {
            "source": source[
                "label"
            ],
            "document": source[
                "document"
            ],
            "document_id": source[
                "document_id"
            ],
            "collection": source.get(
                "collection",
                ""
            ),
            "point_id": source[
                "point_id"
            ],
            "line_start": source[
                "line_start"
            ],
            "line_end": source[
                "line_end"
            ],
            "similarity_score": source[
                "score"
            ],
            "retrieval_rank_score": source.get(
                "retrieval_rank_score",
                source["score"]
            ),
            "requirement_signal_bonus": source.get(
                "requirement_signal_bonus",
                0.0
            ),
            "meta_evidence_penalty": source.get(
                "meta_evidence_penalty",
                0.0
            )
        }
        for source in sources
    ]

    return {
        "question": question,
        "document": KNOWLEDGE_BASE_NAME,
        "document_id": KNOWLEDGE_BASE_ID,
        "status": analysis.get(
            "evidence_status",
            "unknown"
        ),
        "retrieval": {
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "minimum_score": MIN_SCORE,
            "candidate_count": CANDIDATE_K,
            "candidate_limit_per_collection": CANDIDATE_K,
            "sources_used": len(sources),
            **(retrieval_stats or {})
        },
        "analysis": analysis,
        "sources": traceability,
        "generation_seconds": round(
            generation_seconds,
            1
        )
    }


# ============================================================
# HUMAN OUTPUT
# ============================================================

def print_human_report(report):
    analysis = report[
        "analysis"
    ]

    print()
    print("=" * 70)
    print("VIGILANT RMF REQUIREMENT FINDER")
    print("=" * 70)

    print("\nQUESTION")
    print(report["question"])

    print("\nDIRECT ANSWER")
    print(
        analysis.get(
            "direct_answer",
            "No answer returned."
        )
    )

    print("\nEVIDENCE STATUS")
    print(
        analysis.get(
            "evidence_status",
            "unknown"
        ).upper()
    )

    print("\nDOD REQUIREMENTS")

    requirements = analysis.get(
        "dod_requirements",
        []
    )

    if requirements:
        for item in requirements:
            print(
                f"- {item['requirement']}"
            )

            for evidence in item.get(
                "verified_evidence",
                []
            ):
                print(
                    f"  {evidence['source']}: "
                    f"\"{evidence['quote']}\""
                )

                print(
                    "    Validation: "
                    f"{evidence['validation_method']}; "
                    "token coverage="
                    f"{evidence['token_coverage']:.2f}; "
                    "sequence similarity="
                    f"{evidence['sequence_similarity']:.2f}"
                )

    else:
        print(
            "- No validated requirement "
            "extracted."
        )

    print(
        "\nROLES / RESPONSIBILITIES"
    )

    roles = analysis.get(
        "roles_responsibilities",
        []
    )

    if roles:
        for item in roles:
            print(
                f"- {item['role']}: "
                f"{item['responsibility']}"
            )

            for evidence in item.get(
                "verified_linkage_evidence",
                []
            ):
                print(
                    f"  {evidence['source']}: "
                    f"\"{evidence['quote']}\""
                )

    else:
        print(
            "- No explicitly verified "
            "role/responsibility linkage "
            "found."
        )

    print(
        "\nRECOMMENDED COMPANY ACTIONS"
    )

    actions = analysis.get(
        "recommended_company_actions",
        []
    )

    if actions:
        for action in actions:
            print(
                f"- {action}"
            )

    else:
        print(
            "- No recommendation generated."
        )

    limitations = analysis.get(
        "limitations",
        []
    )

    if limitations:
        print(
            "\nLIMITATIONS"
        )

        for limitation in limitations:
            print(
                f"- {limitation}"
            )

    validation = analysis.get(
        "validation",
        {}
    )

    print(
        "\nVALIDATION"
    )

    print(
        "Rejected requirement claims: "
        f"{validation.get('requirements_rejected', 0)}"
    )

    print(
        "Rejected role/responsibility claims: "
        f"{validation.get('role_claims_rejected', 0)}"
    )

    print(
        "\nSOURCE TRACEABILITY"
    )

    sources = report.get(
        "sources",
        []
    )

    if not sources:
        print(
            "No qualifying sources."
        )

    for source in sources:
        print(
            f"\n{source['source']}"
        )

        print(
            f"  Document: "
            f"{source['document']}"
        )

        if source.get("collection"):
            print(
                f"  Collection: "
                f"{source['collection']}"
            )

        print(
            f"  Point ID: "
            f"{source['point_id']}"
        )

        if (
            source["line_start"]
            is not None
            and source["line_end"]
            is not None
        ):
            print(
                f"  Lines: "
                f"{source['line_start']}-"
                f"{source['line_end']}"
            )
        else:
            print(
                "  Lines: unavailable"
            )

        print(
            "  Similarity: "
            f"{source['similarity_score']:.4f}"
        )

    print()
    print("=" * 70)

    print(
        f"Sources used: "
        f"{report['retrieval']['sources_used']}"
    )

    print(
        "Minimum similarity: "
        f"{report['retrieval']['minimum_score']}"
    )

    print(
        "LLM generation time: "
        f"{report['generation_seconds']} seconds"
    )

    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    question = " ".join(
        args.question
    ).strip()

    quiet = args.json_output

    try:
        if not quiet:
            print(
                f"\nSearching "
                f"{KNOWLEDGE_BASE_NAME}..."
            )

            print(
                f"Similarity threshold: "
                f"{MIN_SCORE}"
            )

        (
            points,
            retrieval_stats
        ) = search_qdrant(
            question
        )

        if not points:
            report = (
                insufficient_result(
                    question,
                    retrieval_stats
                )
            )

            if args.json_output:
                print(
                    json.dumps(
                        report,
                        indent=2
                    )
                )
            else:
                print_human_report(
                    report
                )

            return

        sources = build_sources(
            points
        )

        # The embedding request has already completed and uses keep_alive=0,
        # so it is unloaded before the larger chat model is warmed/used.
        if not args.no_warmup:
            warm_chat_model(
                quiet=quiet
            )

        if not quiet:
            print(
                f"Qualified sources: "
                f"{len(sources)}"
            )

            for source in sources:
                print(
                    f"  {source['label']}: "
                    f"{source['score']:.4f}"
                )

            print(
                f"\nGenerating validated "
                f"analysis with "
                f"{CHAT_MODEL}..."
            )

        (
            raw_analysis,
            elapsed
        ) = generate_analysis(
            question,
            sources
        )

        validated_analysis = (
            validate_analysis(
                raw_analysis,
                sources,
                question=question
            )
        )

        report = build_report(
            question,
            validated_analysis,
            sources,
            elapsed,
            retrieval_stats
        )

        if args.json_output:
            print(
                json.dumps(
                    report,
                    indent=2
                )
            )
        else:
            print_human_report(
                report
            )

    except KeyboardInterrupt:
        if not args.json_output:
            print(
                "\nOperation cancelled."
            )

        sys.exit(130)

    except Exception as exc:
        error = {
            "status": "error",
            "error": str(exc)
        }

        if args.json_output:
            print(
                json.dumps(
                    error,
                    indent=2
                )
            )
        else:
            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)
            print(str(exc))

            print(
                "\nDiagnostics:"
            )

            print(
                "  docker exec "
                "ollama ollama ps"
            )

            print(
                "  docker exec "
                "ollama ollama list"
            )

            print(
                "  curl "
                "http://127.0.0.1:11434/api/tags"
            )

            print(
                "  curl "
                "http://127.0.0.1:6333/collections"
            )

            print()

        sys.exit(1)


if __name__ == "__main__":
    main()

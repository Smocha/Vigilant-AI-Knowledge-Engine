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

CANDIDATE_K = 8
MAX_SOURCES = 3
MIN_SCORE = 0.55

OLLAMA_TIMEOUT_SECONDS = 600
KEEP_ALIVE = "30m"
NUM_PREDICT = 2048
MAX_CHARS_PER_SOURCE = 3500

# Requirements tolerate small PDF/LLM wording differences.
MIN_REQUIREMENT_TOKEN_COVERAGE = 0.80
MIN_REQUIREMENT_SEQUENCE_RATIO = 0.75

# Role attribution stays strict.
MIN_RESPONSIBILITY_KEYWORD_OVERLAP = 2

# A generated requirement must be substantially grounded in
# its verified evidence, not merely share a few keywords.
MIN_REQUIREMENT_EVIDENCE_COVERAGE = 0.70

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
            "keep_alive": KEEP_ALIVE,
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


def create_embedding(text):
    result = ollama_post(
        "/api/embed",
        {
            "model": EMBED_MODEL,
            "input": text,
            "keep_alive": KEEP_ALIVE
        }
    )

    embeddings = result.get(
        "embeddings",
        []
    )

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no embedding."
        )

    vector = embeddings[0]

    if len(vector) != EXPECTED_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Embedding dimension mismatch. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSIONS}, "
            f"received {len(vector)}."
        )

    return vector


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def search_qdrant(question):
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=30
    )

    query_vector = create_embedding(
        question
    )

    all_points = []

    collections = (
        client
        .get_collections()
        .collections
    )

    searched_collections = []
    skipped_collections = []
    failed_collections = []

    for collection in collections:
        collection_name = collection.name

        try:
            info = client.get_collection(
                collection_name=collection_name
            )

            vectors = (
                info
                .config
                .params
                .vectors
            )

            query_args = {}

            # Handle named-vector collections,
            # including the NIST collections.
            if isinstance(vectors, dict):
                vector_names = list(
                    vectors.keys()
                )

                if len(vector_names) != 1:
                    skipped_collections.append(
                        collection_name
                    )
                    continue

                vector_name = vector_names[0]
                vector_config = vectors[
                    vector_name
                ]

                query_args["using"] = (
                    vector_name
                )

            else:
                vector_config = vectors

            # Skip incompatible vector sizes.
            if (
                vector_config.size
                != len(query_vector)
            ):
                skipped_collections.append(
                    collection_name
                )
                continue

            searched_collections.append(
                collection_name
            )

            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=CANDIDATE_K,
                with_payload=True,
                with_vectors=False,
                **query_args
            )

            for point in response.points:
                if (
                    point.score is not None
                    and point.score >= MIN_SCORE
                ):
                    payload = dict(
                        point.payload or {}
                    )

                    payload[
                        "_vigilant_collection"
                    ] = collection_name

                    point.payload = payload

                    all_points.append(
                        point
                    )

        except Exception as exc:
            failed_collections.append(
                collection_name
            )

            print(
                "Qdrant collection warning: "
                f"{collection_name!r}: "
                f"{exc}",
                file=sys.stderr
            )

    all_points.sort(
        key=lambda point: (
            float(point.score)
            if point.score is not None
            else 0.0
        ),
        reverse=True
    )

    qualified_collections = sorted({
        str(
            (point.payload or {}).get(
                "_vigilant_collection",
                ""
            )
        ).strip()
        for point in all_points
        if str(
            (point.payload or {}).get(
                "_vigilant_collection",
                ""
            )
        ).strip()
    })

    selected_points = (
        all_points[:MAX_SOURCES]
    )

    represented_collections = sorted({
        str(
            (point.payload or {}).get(
                "_vigilant_collection",
                ""
            )
        ).strip()
        for point in selected_points
        if str(
            (point.payload or {}).get(
                "_vigilant_collection",
                ""
            )
        ).strip()
    })

    retrieval_stats = {
        "collections_discovered": len(collections),
        "collections_searched": len(
            searched_collections
        ),
        "collections_with_qualified_hits": len(
            qualified_collections
        ),
        "collections_represented": (
            represented_collections
        ),
        "collections_skipped": len(
            skipped_collections
        ),
        "collections_failed": len(
            failed_collections
        ),
    }

    return (
        selected_points,
        retrieval_stats
    )


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
    if requirements:
        claims = [
            item.get(
                "requirement",
                ""
            ).strip()
            for item in requirements
            if item.get(
                "requirement",
                ""
            ).strip()
        ]

        if len(claims) == 1:
            return claims[0]

        if claims:
            return (
                "Validated requirements: "
                + "; ".join(claims)
            )

    if roles:
        return (
            "Validated role and responsibility "
            "evidence was found in the retrieved "
            "sources."
        )

    return (
        "The retrieved evidence is insufficient "
        "to establish a validated answer to this "
        "question."
    )


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

            scope_failures = (
                unsupported_requirement_scopes(
                    requirement,
                    matched_text
                )
            )

            if scope_failures:
                unsupported_scopes.update(
                    scope_failures
                )
                continue

            (
                claim_coverage,
                matched_claim_tokens,
                unsupported_claim_tokens
            ) = requirement_evidence_coverage(
                requirement,
                matched_text
            )

            if (
                claim_coverage
                < MIN_REQUIREMENT_EVIDENCE_COVERAGE
            ):
                low_coverage_evidence.append({
                    "source": label,
                    "coverage": claim_coverage,
                    "unsupported_terms": (
                        unsupported_claim_tokens
                    )
                })
                continue

            verified_evidence.append({
                "source": label,
                "quote": quote,
                "validation_method": (
                    match[
                        "validation_method"
                    ]
                ),
                "token_coverage": (
                    match[
                        "token_coverage"
                    ]
                ),
                "sequence_similarity": (
                    match[
                        "sequence_similarity"
                    ]
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
                    match[
                        "matched_text"
                    ]
                )
            })

        if not verified_evidence:
            if unsupported_scopes:
                reason = (
                    "The requirement introduced "
                    "scope term(s) not explicitly "
                    "supported by its verified "
                    "evidence: "
                    + ", ".join(
                        sorted(
                            unsupported_scopes
                        )
                    )
                    + "."
                )
            elif low_coverage_evidence:
                best = max(
                    low_coverage_evidence,
                    key=lambda item: item["coverage"]
                )

                unsupported_terms = (
                    best.get(
                        "unsupported_terms",
                        []
                    )
                )

                reason = (
                    "The requirement was broader than "
                    "the verified evidence. Best "
                    "requirement-to-evidence coverage "
                    f"was {best['coverage']:.2f}; "
                    f"minimum is "
                    f"{MIN_REQUIREMENT_EVIDENCE_COVERAGE:.2f}."
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
            continue

        validated.append({
            "requirement": requirement,
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
You are the Vigilant DoD RMF Requirement Finder.

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

Do not copy scope terms from the user question into a
requirement unless those concepts are explicitly present in the
cited source passage. Do not turn one general passage into separate
identity, device, network, or application requirements unless each
scope is directly supported by the cited evidence.

Requirements must closely restate what the cited passage actually
says. Do not introduce architectural mechanisms, implementation
techniques, technologies, or control concepts that are absent from
the supporting evidence.

Do not use outside knowledge as evidence.

STRICT RULES:

1. Use only the supplied passages.

2. Never invent requirements, responsibilities,
   page numbers, section numbers, paragraph numbers,
   control identifiers, citations, or role assignments.

3. Evidence labels may ONLY be:
   {", ".join(labels)}

4. Every DoD requirement must include at least one
   short source excerpt in explicit_evidence.
   Copy the wording as closely as possible.

5. Every role/responsibility must include at least one
   exact source excerpt that BOTH names the role and
   connects that role to the responsibility.

6. Do not infer responsibility from nearby text.

7. If a role is not explicitly named in the supporting
   excerpt, omit that role.

8. Recommended company actions are advisory and must
   remain separate from formal DoD requirements.

9. If the passages do not answer the question, set
   evidence_status to "insufficient".

10. Do not generate point IDs, line numbers,
    similarity scores, or traceability metadata.
    Python generates those from Qdrant.

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
            "keep_alive": KEEP_ALIVE,
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
    sources
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

    limitations = analysis.get(
        "limitations",
        []
    )

    if not isinstance(
        limitations,
        list
    ):
        limitations = []

    limitations = [
        str(item).strip()
        for item in limitations
        if str(item).strip()
    ]

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

    if validated_claim_count == 0:
        evidence_status = (
            "insufficient"
        )
    else:
        evidence_status = (
            "supported"
            if raw_status == "supported"
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
            ]
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
        if not args.no_warmup:
            warm_chat_model(
                quiet=quiet
            )

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
                sources
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

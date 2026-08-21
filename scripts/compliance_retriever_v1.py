#!/usr/bin/env python3
"""
Production-oriented read-only retriever for Vigilant Compliance v1.

Retrieval:
- active normative NIST SP 800-171r3 records only
- dense: Ollama embeddinggemma:latest
- sparse: FastEmbed Qdrant/bm25
- fusion: Qdrant native RRF
- returns structured authoritative payloads; no LLM generation

Examples:
  python scripts/compliance_retriever_v1.py \
    "How should media containing CUI be protected during transport?"

  python scripts/compliance_retriever_v1.py \
    "03.08.05 media transport" --limit 5 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"

COLLECTION = "vigilant_compliance_v1"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

DENSE_MODEL = "embeddinggemma:latest"
SPARSE_MODEL = "Qdrant/bm25"
EXPECTED_DENSE_DIM = 768

DOCUMENT_ID = "nist_sp_800_171_r3"
DEFAULT_LIMIT = 5
PREFETCH_LIMIT = 20


def active_nist_filter() -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document.document_id",
                match=models.MatchValue(value=DOCUMENT_ID),
            ),
            models.FieldCondition(
                key="record_type",
                match=models.MatchValue(value="requirement"),
            ),
            models.FieldCondition(
                key="normative",
                match=models.MatchValue(value=True),
            ),
        ]
    )


def embed_dense(query: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": DENSE_MODEL,
            "input": query,
            "keep_alive": "0",
        },
        timeout=300,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise RuntimeError("Ollama returned an unexpected embeddings response.")

    vector = embeddings[0]
    if len(vector) != EXPECTED_DENSE_DIM:
        raise RuntimeError(
            f"Dense vector has {len(vector)} dimensions; "
            f"expected {EXPECTED_DENSE_DIM}."
        )

    return [float(value) for value in vector]


def embed_sparse_query(
    model: SparseTextEmbedding,
    query: str,
) -> models.SparseVector:
    embeddings = list(model.query_embed([query]))

    if len(embeddings) != 1:
        raise RuntimeError(
            f"FastEmbed returned {len(embeddings)} sparse query vectors; expected 1."
        )

    embedding = embeddings[0]

    return models.SparseVector(
        indices=[int(value) for value in embedding.indices.tolist()],
        values=[float(value) for value in embedding.values.tolist()],
    )


def normalize_result(point: Any, rank: int) -> dict[str, Any]:
    payload = point.payload or {}
    requirement = payload.get("requirement") or {}
    family = payload.get("family") or {}
    document = payload.get("document") or {}
    source = payload.get("source") or {}
    references = payload.get("references") or {}

    return {
        "rank": rank,
        "score": float(point.score),
        "record_id": payload.get("record_id"),
        "document": {
            "document_id": document.get("document_id"),
            "document_number": document.get("document_number"),
            "revision": document.get("revision"),
            "authority": document.get("authority"),
        },
        "family": {
            "family_id": family.get("family_id"),
            "family_name": family.get("family_name"),
        },
        "requirement": {
            "requirement_id": requirement.get("requirement_id"),
            "title": requirement.get("title"),
            "clause": requirement.get("clause"),
            "text": requirement.get("text"),
        },
        "references": {
            "source_controls": references.get("source_controls") or [],
            "supporting_publications": references.get("supporting_publications") or [],
        },
        "source": {
            "source_file": source.get("source_file"),
            "line_start": source.get("line_start"),
            "line_end": source.get("line_end"),
            "pdf_page": source.get("pdf_page"),
        },
    }


def retrieve(query: str, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if limit < 1 or limit > 20:
        raise ValueError("limit must be between 1 and 20.")

    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(COLLECTION)
    if int(info.points_count or 0) != 255:
        raise RuntimeError(
            f"Collection safety check failed: expected 255 points, "
            f"found {int(info.points_count or 0)}."
        )

    dense_query = embed_dense(query)

    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_query = embed_sparse_query(sparse_model, query)

    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR_NAME,
                filter=active_nist_filter(),
                limit=max(PREFETCH_LIMIT, limit),
            ),
            models.Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                filter=active_nist_filter(),
                limit=max(PREFETCH_LIMIT, limit),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return [
        normalize_result(point, rank)
        for rank, point in enumerate(response.points, 1)
    ]


def print_human(query: str, results: list[dict[str, Any]]) -> None:
    print("=" * 88)
    print("VIGILANT COMPLIANCE RETRIEVER v1")
    print("=" * 88)
    print(f"Query: {query}")
    print(f"Collection: {COLLECTION}")
    print(f"Results: {len(results)}")
    print()

    for result in results:
        req = result["requirement"]
        family = result["family"]
        source = result["source"]

        clause = req["clause"]
        clause_suffix = "" if clause == "base" else f".{clause}"

        print(
            f"{result['rank']}. "
            f"{req['requirement_id']}{clause_suffix} "
            f"{req['title']} "
            f"(score={result['score']:.6f})"
        )
        print(
            f"   Family: {family['family_id']} {family['family_name']}"
        )
        print(f"   {req['text']}")
        print(
            f"   Source lines: "
            f"{source['line_start']}-{source['line_end']}"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve active NIST SP 800-171r3 requirements."
    )
    parser.add_argument("query", help="Natural-language compliance query")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of results (default: {DEFAULT_LIMIT}, max: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    args = parser.parse_args()

    try:
        results = retrieve(args.query, args.limit)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "collection": COLLECTION,
                    "retrieval": {
                        "dense_model": DENSE_MODEL,
                        "sparse_model": SPARSE_MODEL,
                        "fusion": "rrf",
                        "document_id": DOCUMENT_ID,
                        "record_type": "requirement",
                        "normative": True,
                    },
                    "results": results,
                },
                indent=2,
            )
        )
        return

    print_human(args.query, results)


if __name__ == "__main__":
    main()

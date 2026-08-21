#!/usr/bin/env python3
"""
Read-only retrieval test for the 3-point Vigilant Compliance pilot.

Tests each query three ways:
1. dense-only (Ollama embeddinggemma)
2. BM25 sparse-only (FastEmbed Qdrant/bm25)
3. hybrid dense + BM25 using Qdrant RRF

This script DOES NOT write, update, or delete any Qdrant points.
"""

from __future__ import annotations

import sys
from typing import Iterable

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

TESTS = [
    {
        "query": "What authentication mechanism prevents replay attacks for account access?",
        "expected": ("03.05.04", "base"),
    },
    {
        "query": "What is required for secure nonlocal maintenance and diagnostic sessions?",
        "expected": ("03.07.05", "b"),
    },
    {
        "query": "How must system media containing CUI be protected during transport outside controlled areas?",
        "expected": ("03.08.05", "a"),
    },
]


def active_nist_filter() -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document.document_id",
                match=models.MatchValue(value="nist_sp_800_171_r3"),
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


def embed_dense(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": DENSE_MODEL,
            "input": texts,
            "keep_alive": "0",
        },
        timeout=300,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama response did not contain an embeddings list.")

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} queries."
        )

    vectors: list[list[float]] = []
    for index, embedding in enumerate(embeddings, 1):
        if len(embedding) != EXPECTED_DENSE_DIM:
            raise RuntimeError(
                f"Dense query vector {index} has {len(embedding)} dimensions; "
                f"expected {EXPECTED_DENSE_DIM}."
            )
        vectors.append([float(value) for value in embedding])

    return vectors


def to_sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(value) for value in embedding.indices.tolist()],
        values=[float(value) for value in embedding.values.tolist()],
    )


def record_key(point) -> tuple[str, str]:
    payload = point.payload or {}
    req = payload.get("requirement", {})
    return str(req.get("requirement_id", "")), str(req.get("clause", ""))


def result_label(point) -> str:
    payload = point.payload or {}
    req = payload.get("requirement", {})
    return (
        f"{req.get('requirement_id', '?')} | "
        f"{req.get('clause', '?')} | "
        f"{req.get('title', '?')}"
    )


def result_text(point) -> str:
    payload = point.payload or {}
    req = payload.get("requirement", {})
    return str(req.get("text", ""))


def print_results(
    mode: str,
    points: Iterable,
    expected: tuple[str, str],
) -> int | None:
    points = list(points)

    print(f"\n  {mode}")
    if not points:
        print("    NO RESULTS")
        return None

    expected_rank = None

    for rank, point in enumerate(points, 1):
        key = record_key(point)
        if key == expected and expected_rank is None:
            expected_rank = rank

        marker = " <-- expected" if key == expected else ""
        print(
            f"    {rank}. score={point.score:.6f} "
            f"{result_label(point)}{marker}"
        )
        print(f"       {result_text(point)}")

    if expected_rank is None:
        print("    Expected record was NOT returned.")
    else:
        print(f"    Expected record rank: {expected_rank}")

    return expected_rank


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(COLLECTION)
    point_count = int(info.points_count or 0)

    if point_count != 3:
        raise SystemExit(
            f"Safety check failed: expected exactly 3 pilot points in "
            f"{COLLECTION}, found {point_count}."
        )

    print("=" * 88)
    print("VIGILANT COMPLIANCE PILOT RETRIEVAL TEST")
    print("=" * 88)
    print(f"Collection: {COLLECTION}")
    print(f"points_count: {point_count}")
    print(f"Dense model: {DENSE_MODEL}")
    print(f"Sparse model: {SPARSE_MODEL}")
    print("Modes: dense-only | BM25-only | hybrid RRF")

    queries = [test["query"] for test in TESTS]

    print("\nGenerating dense query embeddings in one Ollama batch...")
    dense_queries = embed_dense(queries)
    print(f"Dense query embeddings: PASS ({len(dense_queries)} x 768)")

    print("\nGenerating BM25 query vectors...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_queries = [
        to_sparse_vector(embedding)
        for embedding in sparse_model.query_embed(queries)
    ]
    print(f"BM25 query embeddings: PASS ({len(sparse_queries)})")

    query_filter = active_nist_filter()

    hybrid_top1_passes = 0
    dense_top1_passes = 0
    sparse_top1_passes = 0

    for index, test in enumerate(TESTS):
        query = test["query"]
        expected = test["expected"]
        dense_query = dense_queries[index]
        sparse_query = sparse_queries[index]

        print("\n" + "=" * 88)
        print(f"TEST {index + 1}")
        print(f"Query: {query}")
        print(f"Expected: {expected[0]} | {expected[1]}")

        dense_response = client.query_points(
            collection_name=COLLECTION,
            query=dense_query,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )

        sparse_response = client.query_points(
            collection_name=COLLECTION,
            query=sparse_query,
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )

        hybrid_response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=3,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=3,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=3,
            with_payload=True,
            with_vectors=False,
        )

        dense_rank = print_results(
            "DENSE ONLY",
            dense_response.points,
            expected,
        )
        sparse_rank = print_results(
            "BM25 ONLY",
            sparse_response.points,
            expected,
        )
        hybrid_rank = print_results(
            "HYBRID RRF",
            hybrid_response.points,
            expected,
        )

        if dense_rank == 1:
            dense_top1_passes += 1
        if sparse_rank == 1:
            sparse_top1_passes += 1
        if hybrid_rank == 1:
            hybrid_top1_passes += 1

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Dense top-1 expected:  {dense_top1_passes}/{len(TESTS)}")
    print(f"BM25 top-1 expected:   {sparse_top1_passes}/{len(TESTS)}")
    print(f"Hybrid top-1 expected: {hybrid_top1_passes}/{len(TESTS)}")

    if hybrid_top1_passes == len(TESTS):
        print("\nPILOT RETRIEVAL RESULT: PASS")
        print("Hybrid RRF ranked the expected requirement first for all 3 tests.")
        return

    print("\nPILOT RETRIEVAL RESULT: REVIEW")
    print(
        "At least one hybrid query did not rank its expected pilot requirement first. "
        "Do not bulk ingest yet."
    )
    raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

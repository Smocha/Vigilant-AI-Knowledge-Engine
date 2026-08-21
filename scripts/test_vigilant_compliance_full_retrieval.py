#!/usr/bin/env python3
"""
Full-corpus read-only retrieval regression test for vigilant_compliance_v1.

Tests hybrid retrieval against all active NIST SP 800-171r3 requirement records.

Modes:
- dense-only
- BM25-only
- hybrid RRF

Success criteria:
- every expected requirement must appear in hybrid top-3
- report hybrid top-1 rate separately
- no writes, updates, or deletes
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

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

EXPECTED_TOTAL_POINTS = 255
EXPECTED_ACTIVE_POINTS = 222
TOP_K = 5


@dataclass(frozen=True)
class TestCase:
    query: str
    expected_requirement_id: str
    expected_clause: str
    label: str


TESTS = [
    TestCase(
        query="How should logical access to CUI and system resources be enforced?",
        expected_requirement_id="03.01.02",
        expected_clause="base",
        label="Access Enforcement",
    ),
    TestCase(
        query="How should the flow of CUI within and between connected systems be controlled?",
        expected_requirement_id="03.01.03",
        expected_clause="base",
        label="Information Flow Enforcement",
    ),
    TestCase(
        query="What authentication mechanism prevents replay attacks for privileged and non-privileged account access?",
        expected_requirement_id="03.05.04",
        expected_clause="base",
        label="Replay-Resistant Authentication",
    ),
    TestCase(
        query="When should a user session be automatically terminated?",
        expected_requirement_id="03.01.11",
        expected_clause="base",
        label="Session Termination",
    ),
    TestCase(
        query="What is required for secure nonlocal maintenance and diagnostic sessions?",
        expected_requirement_id="03.07.05",
        expected_clause="b",
        label="Nonlocal Maintenance",
    ),
    TestCase(
        query="How must system media containing CUI be protected during transport outside controlled areas?",
        expected_requirement_id="03.08.05",
        expected_clause="a",
        label="Media Transport",
    ),
    TestCase(
        query="How should system media containing CUI be physically controlled and securely stored?",
        expected_requirement_id="03.08.01",
        expected_clause="base",
        label="Media Storage",
    ),
    TestCase(
        query="Who should be allowed to access CUI stored on system media?",
        expected_requirement_id="03.08.02",
        expected_clause="base",
        label="Media Access",
    ),
    TestCase(
        query="What should an incident-handling capability include?",
        expected_requirement_id="03.06.01",
        expected_clause="base",
        label="Incident Handling",
    ),
    TestCase(
        query="What control requires multi-factor authentication for privileged and non-privileged accounts?",
        expected_requirement_id="03.05.03",
        expected_clause="base",
        label="Multi-Factor Authentication",
    ),
    TestCase(
        query="03.08.05 media transport",
        expected_requirement_id="03.08.05",
        expected_clause="a",
        label="Exact-ID Media Transport",
    ),
    TestCase(
        query="03.05.04 replay-resistant authentication",
        expected_requirement_id="03.05.04",
        expected_clause="base",
        label="Exact-ID Replay Resistance",
    ),
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
        timeout=600,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama response did not contain an embeddings list.")

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} queries."
        )

    result: list[list[float]] = []

    for index, embedding in enumerate(embeddings, 1):
        if len(embedding) != EXPECTED_DENSE_DIM:
            raise RuntimeError(
                f"Dense query vector {index} has {len(embedding)} dimensions; "
                f"expected {EXPECTED_DENSE_DIM}."
            )
        result.append([float(value) for value in embedding])

    return result


def to_sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(value) for value in embedding.indices.tolist()],
        values=[float(value) for value in embedding.values.tolist()],
    )


def point_key(point) -> tuple[str, str]:
    payload = point.payload or {}
    req = payload.get("requirement") or {}
    return (
        str(req.get("requirement_id", "")),
        str(req.get("clause", "")),
    )


def point_label(point) -> str:
    payload = point.payload or {}
    req = payload.get("requirement") or {}
    return (
        f"{req.get('requirement_id', '?')} | "
        f"{req.get('clause', '?')} | "
        f"{req.get('title', '?')}"
    )


def expected_rank(points, expected_key: tuple[str, str]) -> int | None:
    for rank, point in enumerate(points, 1):
        if point_key(point) == expected_key:
            return rank
    return None


def print_top(points, expected_key: tuple[str, str], limit: int = TOP_K) -> None:
    for rank, point in enumerate(points[:limit], 1):
        marker = " <-- expected" if point_key(point) == expected_key else ""
        print(
            f"      {rank}. score={point.score:.6f} "
            f"{point_label(point)}{marker}"
        )


def count_active_points(client: QdrantClient) -> int:
    result = client.count(
        collection_name=COLLECTION,
        count_filter=active_nist_filter(),
        exact=True,
    )
    return int(result.count)


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(COLLECTION)
    total_points = int(info.points_count or 0)
    active_points = count_active_points(client)

    if total_points != EXPECTED_TOTAL_POINTS:
        raise SystemExit(
            f"Safety check failed: expected {EXPECTED_TOTAL_POINTS} total points, "
            f"found {total_points}."
        )

    if active_points != EXPECTED_ACTIVE_POINTS:
        raise SystemExit(
            f"Safety check failed: expected {EXPECTED_ACTIVE_POINTS} active points, "
            f"found {active_points}."
        )

    print("=" * 96)
    print("VIGILANT COMPLIANCE FULL-CORPUS RETRIEVAL REGRESSION")
    print("=" * 96)
    print(f"Collection: {COLLECTION}")
    print(f"Total points: {total_points}")
    print(f"Active normative NIST requirements: {active_points}")
    print(f"Test cases: {len(TESTS)}")
    print(f"Result depth: top-{TOP_K}")
    print("Success rule: expected requirement must appear in HYBRID top-3 for every test")

    queries = [test.query for test in TESTS]

    print("\nGenerating dense query embeddings...")
    dense_queries = embed_dense(queries)
    print(f"Dense query embeddings: PASS ({len(dense_queries)} x {EXPECTED_DENSE_DIM})")

    print("\nGenerating BM25 query embeddings...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_queries = [
        to_sparse_vector(embedding)
        for embedding in sparse_model.query_embed(queries)
    ]
    print(f"BM25 query embeddings: PASS ({len(sparse_queries)})")

    query_filter = active_nist_filter()

    dense_top1 = 0
    dense_top3 = 0
    sparse_top1 = 0
    sparse_top3 = 0
    hybrid_top1 = 0
    hybrid_top3 = 0

    failed_hybrid: list[tuple[TestCase, int | None]] = []

    for index, test in enumerate(TESTS):
        expected_key = (
            test.expected_requirement_id,
            test.expected_clause,
        )

        dense_query = dense_queries[index]
        sparse_query = sparse_queries[index]

        dense_response = client.query_points(
            collection_name=COLLECTION,
            query=dense_query,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        )

        sparse_response = client.query_points(
            collection_name=COLLECTION,
            query=sparse_query,
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=TOP_K,
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
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        )

        dense_rank = expected_rank(dense_response.points, expected_key)
        sparse_rank = expected_rank(sparse_response.points, expected_key)
        hybrid_rank = expected_rank(hybrid_response.points, expected_key)

        dense_top1 += int(dense_rank == 1)
        dense_top3 += int(dense_rank is not None and dense_rank <= 3)
        sparse_top1 += int(sparse_rank == 1)
        sparse_top3 += int(sparse_rank is not None and sparse_rank <= 3)
        hybrid_top1 += int(hybrid_rank == 1)
        hybrid_top3 += int(hybrid_rank is not None and hybrid_rank <= 3)

        if hybrid_rank is None or hybrid_rank > 3:
            failed_hybrid.append((test, hybrid_rank))

        print("\n" + "-" * 96)
        print(f"TEST {index + 1}: {test.label}")
        print(f"Query: {test.query}")
        print(
            f"Expected: {test.expected_requirement_id} | "
            f"{test.expected_clause}"
        )
        print(
            f"Ranks: dense={dense_rank or '-'} | "
            f"bm25={sparse_rank or '-'} | "
            f"hybrid={hybrid_rank or '-'}"
        )
        print("  HYBRID TOP RESULTS:")
        print_top(hybrid_response.points, expected_key)

    total = len(TESTS)

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"Dense top-1:   {dense_top1}/{total}")
    print(f"Dense top-3:   {dense_top3}/{total}")
    print(f"BM25 top-1:    {sparse_top1}/{total}")
    print(f"BM25 top-3:    {sparse_top3}/{total}")
    print(f"Hybrid top-1:  {hybrid_top1}/{total}")
    print(f"Hybrid top-3:  {hybrid_top3}/{total}")

    if failed_hybrid:
        print("\nFULL-CORPUS RETRIEVAL RESULT: REVIEW")
        print("Hybrid failed the top-3 criterion for:")
        for test, rank in failed_hybrid:
            print(
                f"  {test.expected_requirement_id}|{test.expected_clause} "
                f"{test.label}: rank={rank or 'not in top-5'}"
            )
        raise SystemExit(2)

    print("\nFULL-CORPUS RETRIEVAL RESULT: PASS")
    print("Every expected requirement appeared in hybrid top-3.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

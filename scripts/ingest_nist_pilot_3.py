#!/usr/bin/env python3
"""
Pilot ingest for exactly three audited NIST SP 800-171r3 records.

Writes ONLY these records to Qdrant collection vigilant_compliance_v1:
- 03.05.04 | base | Replay-Resistant Authentication
- 03.07.05 | b    | Nonlocal Maintenance
- 03.08.05 | a    | Media Transport

Dense vector: Ollama embeddinggemma:latest
Sparse vector: FastEmbed Qdrant/bm25
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
from pathlib import Path

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

JSONL_PATH = Path(
    "/home/vigilantceo/vigilant-automation/data/structured/"
    "nist_sp_800_171_r3_requirements.jsonl"
)

TARGET_RECORD_IDS = [
    "nist_sp_800_171_r3|03.05.04|base",
    "nist_sp_800_171_r3|03.07.05|b",
    "nist_sp_800_171_r3|03.08.05|a",
]


def load_target_records() -> list[dict]:
    if not JSONL_PATH.exists():
        raise SystemExit(f"Missing JSONL: {JSONL_PATH}")

    wanted = set(TARGET_RECORD_IDS)
    found: dict[str, dict] = {}

    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON at {JSONL_PATH}:{line_no}: {exc}"
                ) from exc

            record_id = record.get("record_id")
            if record_id in wanted:
                found[record_id] = record

    missing = [record_id for record_id in TARGET_RECORD_IDS if record_id not in found]
    if missing:
        raise SystemExit(
            "Missing expected pilot record(s):\n  " + "\n  ".join(missing)
        )

    records = [found[record_id] for record_id in TARGET_RECORD_IDS]

    for record in records:
        if record.get("record_type") != "requirement":
            raise SystemExit(
                f"Pilot record is not an active requirement: {record['record_id']}"
            )
        if record.get("normative") is not True:
            raise SystemExit(
                f"Pilot record is not normative: {record['record_id']}"
            )

    return records


def build_search_text(record: dict) -> str:
    document = record["document"]
    family = record["family"]
    requirement = record["requirement"]

    return "\n".join(
        [
            f"Document: NIST SP {document['document_number']} {document['revision']}",
            f"Family: {family['family_id']} {family['family_name']}",
            (
                f"Requirement: {requirement['requirement_id']} "
                f"{requirement['title']}"
            ),
            f"Clause: {requirement['clause']}",
            f"Text: {requirement['text']}",
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

    body = response.json()
    embeddings = body.get("embeddings")

    if not isinstance(embeddings, list):
        raise SystemExit("Ollama response did not contain an embeddings list.")

    if len(embeddings) != len(texts):
        raise SystemExit(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts."
        )

    for idx, vector in enumerate(embeddings, 1):
        if len(vector) != EXPECTED_DENSE_DIM:
            raise SystemExit(
                f"Dense vector {idx} has {len(vector)} dimensions; "
                f"expected {EXPECTED_DENSE_DIM}."
            )

    return [[float(value) for value in vector] for vector in embeddings]


def embed_sparse_documents(texts: list[str]) -> list[models.SparseVector]:
    model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    # BM25 document-side encoding. Query-time encoding will use query_embed()
    # in the search test, because BM25 weights documents and queries differently.
    embeddings = list(model.embed(texts))

    if len(embeddings) != len(texts):
        raise SystemExit(
            f"FastEmbed returned {len(embeddings)} sparse vectors "
            f"for {len(texts)} texts."
        )

    vectors: list[models.SparseVector] = []
    for embedding in embeddings:
        vectors.append(
            models.SparseVector(
                indices=[int(value) for value in embedding.indices.tolist()],
                values=[float(value) for value in embedding.values.tolist()],
            )
        )

    return vectors


def qdrant_point_id(record_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"vigilant-compliance:{record_id}",
        )
    )


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(COLLECTION)
    existing_count = int(info.points_count or 0)

    if existing_count != 0:
        raise SystemExit(
            f"SAFETY STOP: {COLLECTION} already contains "
            f"{existing_count} point(s). Expected 0 for the pilot ingest."
        )

    records = load_target_records()
    search_texts = [build_search_text(record) for record in records]

    print("Pilot records selected:")
    for record in records:
        req = record["requirement"]
        print(
            f"  {req['requirement_id']} | {req['clause']} | "
            f"{req['title']}"
        )

    print("\nGenerating 3 dense embeddings with Ollama...")
    dense_vectors = embed_dense(search_texts)
    print(
        f"Dense embeddings: PASS "
        f"({len(dense_vectors)} x {EXPECTED_DENSE_DIM})"
    )

    print("\nGenerating 3 BM25 document sparse vectors...")
    sparse_vectors = embed_sparse_documents(search_texts)
    print("Sparse embeddings: PASS")
    for record, sparse in zip(records, sparse_vectors):
        print(
            f"  {record['record_id']}: "
            f"{len(sparse.indices)} non-zero dimensions"
        )

    points: list[models.PointStruct] = []

    for record, search_text, dense, sparse in zip(
        records,
        search_texts,
        dense_vectors,
        sparse_vectors,
    ):
        payload = copy.deepcopy(record)
        payload["search"] = {
            "text": search_text,
            "dense_model": DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
        }
        payload["ingestion"] = {
            "pipeline": "vigilant_compliance_v1",
            "pilot": True,
        }

        points.append(
            models.PointStruct(
                id=qdrant_point_id(record["record_id"]),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: sparse,
                },
                payload=payload,
            )
        )

    print("\nUpserting exactly 3 pilot points...")
    result = client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )
    print(f"Qdrant upsert status: {result.status}")

    info = client.get_collection(COLLECTION)
    final_count = int(info.points_count or 0)

    if final_count != 3:
        raise SystemExit(
            f"Pilot verification failed: expected 3 points, found {final_count}."
        )

    point_ids = [point.id for point in points]
    stored = client.retrieve(
        collection_name=COLLECTION,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,
    )

    if len(stored) != 3:
        raise SystemExit(
            f"Pilot verification failed: retrieve returned {len(stored)} points."
        )

    print("\nStored pilot points:")
    for point in stored:
        payload = point.payload or {}
        req = payload.get("requirement", {})
        print(
            f"  {req.get('requirement_id')} | "
            f"{req.get('clause')} | "
            f"{req.get('title')}"
        )

    print("\nPILOT INGEST RESULT: PASS")
    print(f"Collection: {COLLECTION}")
    print(f"points_count: {final_count}")
    print("No other JSONL records were ingested.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

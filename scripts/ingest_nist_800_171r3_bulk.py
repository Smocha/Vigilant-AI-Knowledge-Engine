#!/usr/bin/env python3
"""
Bulk ingest the audited NIST SP 800-171r3 structured JSONL into
Qdrant collection `vigilant_compliance_v1`.

Safety behavior:
- expects the audited dataset to contain exactly:
    255 total records
    222 active requirements
     33 withdrawn records
- expects the Qdrant collection to contain either:
    3 pilot points, or
    255 already-completed points
- if 3 pilot points exist, verifies they are the expected pilot IDs
- uses deterministic UUID5 point IDs, so pilot points are replaced cleanly
  by their final production payloads
- batches embeddings and upserts to reduce memory pressure
- no LLM generation is used

Dense vector: Ollama embeddinggemma:latest
Sparse vector: FastEmbed Qdrant/bm25
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
from collections import Counter
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

EXPECTED_TOTAL = 255
EXPECTED_ACTIVE = 222
EXPECTED_WITHDRAWN = 33
BATCH_SIZE = 16

PILOT_RECORD_IDS = [
    "nist_sp_800_171_r3|03.05.04|base",
    "nist_sp_800_171_r3|03.07.05|b",
    "nist_sp_800_171_r3|03.08.05|a",
]


def point_id_for(record_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"vigilant-compliance:{record_id}",
        )
    )


def load_records() -> list[dict]:
    if not JSONL_PATH.exists():
        raise SystemExit(f"Missing JSONL: {JSONL_PATH}")

    records: list[dict] = []

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

            records.append(record)

    record_ids = [record.get("record_id") for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise SystemExit("Dataset safety check failed: duplicate record_id values.")

    type_counts = Counter(record.get("record_type") for record in records)

    if len(records) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Dataset safety check failed: expected {EXPECTED_TOTAL} records, "
            f"found {len(records)}."
        )

    if type_counts["requirement"] != EXPECTED_ACTIVE:
        raise SystemExit(
            f"Dataset safety check failed: expected {EXPECTED_ACTIVE} active "
            f"requirements, found {type_counts['requirement']}."
        )

    if type_counts["withdrawn"] != EXPECTED_WITHDRAWN:
        raise SystemExit(
            f"Dataset safety check failed: expected {EXPECTED_WITHDRAWN} withdrawn "
            f"records, found {type_counts['withdrawn']}."
        )

    for record in records:
        record_type = record.get("record_type")
        normative = record.get("normative")

        if record_type == "requirement" and normative is not True:
            raise SystemExit(
                f"Active requirement is not normative: {record.get('record_id')}"
            )

        if record_type == "withdrawn" and normative is not False:
            raise SystemExit(
                f"Withdrawn record is marked normative: {record.get('record_id')}"
            )

    return records


def build_search_text(record: dict) -> str:
    document = record["document"]
    family = record["family"]
    requirement = record["requirement"]
    references = record.get("references", {})

    status = (
        "Active normative security requirement"
        if record["record_type"] == "requirement"
        else "Withdrawn security requirement"
    )

    lines = [
        f"Document: NIST SP {document['document_number']} {document['revision']}",
        f"Authority: {document['authority']}",
        f"Status: {status}",
        f"Family: {family['family_id']} {family['family_name']}",
        f"Requirement: {requirement['requirement_id']} {requirement['title']}",
        f"Clause: {requirement['clause']}",
        f"Text: {requirement['text']}",
    ]

    source_controls = references.get("source_controls") or []
    if source_controls:
        lines.append("Source controls: " + ", ".join(source_controls))

    supporting = references.get("supporting_publications") or []
    if supporting:
        lines.append("Supporting publications: " + ", ".join(supporting))

    return "\n".join(lines)


def embed_dense(
    texts: list[str],
    *,
    unload_after: bool,
) -> list[list[float]]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": DENSE_MODEL,
            "input": texts,
            "keep_alive": "0" if unload_after else "10m",
        },
        timeout=600,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama response did not contain an embeddings list.")

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts."
        )

    vectors: list[list[float]] = []

    for index, embedding in enumerate(embeddings, 1):
        if len(embedding) != EXPECTED_DENSE_DIM:
            raise RuntimeError(
                f"Dense vector {index} has {len(embedding)} dimensions; "
                f"expected {EXPECTED_DENSE_DIM}."
            )

        vectors.append([float(value) for value in embedding])

    return vectors


def sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(value) for value in embedding.indices.tolist()],
        values=[float(value) for value in embedding.values.tolist()],
    )


def verify_collection_state(client: QdrantClient) -> int:
    info = client.get_collection(COLLECTION)
    count = int(info.points_count or 0)

    if count == EXPECTED_TOTAL:
        print(
            f"{COLLECTION} already contains {EXPECTED_TOTAL} points. "
            "Bulk ingest appears complete."
        )
        return count

    if count != len(PILOT_RECORD_IDS):
        raise SystemExit(
            f"SAFETY STOP: expected either {len(PILOT_RECORD_IDS)} pilot points "
            f"or {EXPECTED_TOTAL} completed points, found {count}."
        )

    expected_pilot_ids = [point_id_for(record_id) for record_id in PILOT_RECORD_IDS]

    stored = client.retrieve(
        collection_name=COLLECTION,
        ids=expected_pilot_ids,
        with_payload=True,
        with_vectors=False,
    )

    if len(stored) != len(PILOT_RECORD_IDS):
        raise SystemExit(
            "SAFETY STOP: collection has 3 points, but they are not the "
            "expected pilot points."
        )

    stored_record_ids = {
        (point.payload or {}).get("record_id")
        for point in stored
    }

    if stored_record_ids != set(PILOT_RECORD_IDS):
        raise SystemExit(
            "SAFETY STOP: pilot payload record IDs do not match expectations."
        )

    print("Existing 3-point pilot state: VERIFIED")
    return count


def scroll_all_payloads(client: QdrantClient) -> list[dict]:
    payloads: list[dict] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        payloads.extend((point.payload or {}) for point in points)

        if next_offset is None:
            break

        offset = next_offset

    return payloads


def main() -> None:
    print("=" * 88)
    print("VIGILANT COMPLIANCE NIST SP 800-171r3 BULK INGEST")
    print("=" * 88)

    records = load_records()

    print(f"Dataset: {JSONL_PATH}")
    print(f"Total records: {len(records)}")
    print(f"Active requirements: {EXPECTED_ACTIVE}")
    print(f"Withdrawn records: {EXPECTED_WITHDRAWN}")
    print(f"Batch size: {BATCH_SIZE}")

    client = QdrantClient(url=QDRANT_URL)

    current_count = verify_collection_state(client)
    if current_count == EXPECTED_TOTAL:
        print("BULK INGEST RESULT: ALREADY COMPLETE")
        return

    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"\nBeginning bulk upsert of {EXPECTED_TOTAL} deterministic point IDs "
        f"in {total_batches} batches."
    )
    print(
        "The 3 pilot points will be overwritten with identical record IDs "
        "and final non-pilot payload metadata."
    )

    for batch_number, start in enumerate(
        range(0, len(records), BATCH_SIZE),
        1,
    ):
        batch = records[start:start + BATCH_SIZE]
        search_texts = [build_search_text(record) for record in batch]

        unload_after = batch_number == total_batches

        print(
            f"\nBatch {batch_number}/{total_batches}: "
            f"records {start + 1}-{start + len(batch)}"
        )

        dense_vectors = embed_dense(
            search_texts,
            unload_after=unload_after,
        )

        sparse_embeddings = list(sparse_model.embed(search_texts))

        if len(sparse_embeddings) != len(batch):
            raise RuntimeError(
                f"FastEmbed returned {len(sparse_embeddings)} sparse vectors "
                f"for a batch of {len(batch)} records."
            )

        points: list[models.PointStruct] = []

        for record, text, dense, sparse_embedding in zip(
            batch,
            search_texts,
            dense_vectors,
            sparse_embeddings,
        ):
            payload = copy.deepcopy(record)

            payload["search"] = {
                "text": text,
                "dense_model": DENSE_MODEL,
                "sparse_model": SPARSE_MODEL,
            }

            payload["ingestion"] = {
                "pipeline": "vigilant_compliance_v1",
                "pilot": False,
                "dataset": "nist_sp_800_171_r3",
            }

            points.append(
                models.PointStruct(
                    id=point_id_for(record["record_id"]),
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: sparse_vector(sparse_embedding),
                    },
                    payload=payload,
                )
            )

        result = client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

        if str(result.status).lower() not in {"completed", "acknowledged"}:
            raise RuntimeError(
                f"Unexpected Qdrant upsert status in batch {batch_number}: "
                f"{result.status}"
            )

        info = client.get_collection(COLLECTION)
        current = int(info.points_count or 0)
        print(f"  Qdrant points_count: {current}")

    info = client.get_collection(COLLECTION)
    final_count = int(info.points_count or 0)

    if final_count != EXPECTED_TOTAL:
        raise SystemExit(
            f"Final verification failed: expected {EXPECTED_TOTAL} points, "
            f"found {final_count}."
        )

    payloads = scroll_all_payloads(client)

    if len(payloads) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Scroll verification failed: expected {EXPECTED_TOTAL} payloads, "
            f"found {len(payloads)}."
        )

    type_counts = Counter(payload.get("record_type") for payload in payloads)
    normative_counts = Counter(payload.get("normative") for payload in payloads)
    document_ids = Counter(
        (payload.get("document") or {}).get("document_id")
        for payload in payloads
    )

    print("\n" + "=" * 88)
    print("FINAL VERIFICATION")
    print("=" * 88)
    print(f"points_count: {final_count}")
    print(f"record_type=requirement: {type_counts['requirement']}")
    print(f"record_type=withdrawn: {type_counts['withdrawn']}")
    print(f"normative=true: {normative_counts[True]}")
    print(f"normative=false: {normative_counts[False]}")
    print(
        "document_id=nist_sp_800_171_r3: "
        f"{document_ids['nist_sp_800_171_r3']}"
    )

    if type_counts["requirement"] != EXPECTED_ACTIVE:
        raise SystemExit("Final verification failed for active requirement count.")

    if type_counts["withdrawn"] != EXPECTED_WITHDRAWN:
        raise SystemExit("Final verification failed for withdrawn record count.")

    if normative_counts[True] != EXPECTED_ACTIVE:
        raise SystemExit("Final verification failed for normative=true count.")

    if normative_counts[False] != EXPECTED_WITHDRAWN:
        raise SystemExit("Final verification failed for normative=false count.")

    if document_ids["nist_sp_800_171_r3"] != EXPECTED_TOTAL:
        raise SystemExit("Final verification failed for document_id count.")

    print("\nBULK INGEST RESULT: PASS")
    print(
        "All 255 audited NIST SP 800-171r3 records are now stored in "
        f"{COLLECTION}."
    )
    print(
        "Normal compliance search should filter to record_type=requirement "
        "and normative=true, which excludes the 33 withdrawn records."
    )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

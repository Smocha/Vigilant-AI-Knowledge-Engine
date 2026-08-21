#!/usr/bin/env python3
"""
Resume-safe bulk ingest for the audited NIST SP 800-171r3 dataset.

Collection: vigilant_compliance_v1
Dataset expectations:
  255 total records
  222 active requirements
   33 withdrawn records

Resume behavior:
- validates every existing Qdrant point against the audited JSONL
- rejects unknown/mismatched points
- skips already-finalized production points
- reprocesses pilot=True points so they receive final production payload metadata
- ingests only records that are missing or still marked as pilot
- safe to rerun after interruption

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

CANONICAL_PAYLOAD_KEYS = (
    "schema_version",
    "record_id",
    "document",
    "record_type",
    "normative",
    "family",
    "requirement",
    "references",
    "source",
)


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


def canonical_payload(record_or_payload: dict) -> dict:
    return {
        key: copy.deepcopy(record_or_payload.get(key))
        for key in CANONICAL_PAYLOAD_KEYS
    }


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


def scroll_all_points(client: QdrantClient):
    points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)

        if next_offset is None:
            break

        offset = next_offset

    return points


def validate_existing_points(
    client: QdrantClient,
    records_by_id: dict[str, dict],
) -> tuple[set[str], set[str]]:
    """
    Return:
      finalized_ids: existing production points that can be skipped
      pilot_ids: existing pilot points that must be reprocessed
    """
    points = scroll_all_points(client)

    if len(points) > EXPECTED_TOTAL:
        raise SystemExit(
            f"SAFETY STOP: collection contains {len(points)} points; "
            f"dataset only contains {EXPECTED_TOTAL}."
        )

    finalized_ids: set[str] = set()
    pilot_ids: set[str] = set()
    seen_record_ids: set[str] = set()

    for point in points:
        payload = point.payload or {}
        record_id = payload.get("record_id")

        if not record_id:
            raise SystemExit(
                f"SAFETY STOP: Qdrant point {point.id} has no payload record_id."
            )

        if record_id not in records_by_id:
            raise SystemExit(
                f"SAFETY STOP: unknown record_id exists in collection: {record_id}"
            )

        if record_id in seen_record_ids:
            raise SystemExit(
                f"SAFETY STOP: duplicate logical record_id in collection: {record_id}"
            )
        seen_record_ids.add(record_id)

        expected_point_id = point_id_for(record_id)
        if str(point.id) != expected_point_id:
            raise SystemExit(
                "SAFETY STOP: deterministic point ID mismatch:\n"
                f"  record_id: {record_id}\n"
                f"  stored point id: {point.id}\n"
                f"  expected point id: {expected_point_id}"
            )

        expected_record = records_by_id[record_id]
        if canonical_payload(payload) != canonical_payload(expected_record):
            raise SystemExit(
                "SAFETY STOP: existing Qdrant payload does not match audited JSONL:\n"
                f"  {record_id}"
            )

        ingestion = payload.get("ingestion") or {}
        pilot = ingestion.get("pilot")

        if pilot is True:
            pilot_ids.add(record_id)
        elif pilot is False:
            finalized_ids.add(record_id)
        else:
            raise SystemExit(
                "SAFETY STOP: existing point has unexpected ingestion.pilot value:\n"
                f"  {record_id}: {pilot!r}"
            )

    return finalized_ids, pilot_ids


def final_verify(client: QdrantClient) -> None:
    points = scroll_all_points(client)

    if len(points) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Final verification failed: expected {EXPECTED_TOTAL} points, "
            f"found {len(points)}."
        )

    payloads = [point.payload or {} for point in points]

    type_counts = Counter(payload.get("record_type") for payload in payloads)
    normative_counts = Counter(payload.get("normative") for payload in payloads)
    pilot_counts = Counter(
        (payload.get("ingestion") or {}).get("pilot")
        for payload in payloads
    )
    document_ids = Counter(
        (payload.get("document") or {}).get("document_id")
        for payload in payloads
    )

    print("\n" + "=" * 88)
    print("FINAL VERIFICATION")
    print("=" * 88)
    print(f"points_count: {len(points)}")
    print(f"record_type=requirement: {type_counts['requirement']}")
    print(f"record_type=withdrawn: {type_counts['withdrawn']}")
    print(f"normative=true: {normative_counts[True]}")
    print(f"normative=false: {normative_counts[False]}")
    print(f"ingestion.pilot=false: {pilot_counts[False]}")
    print(f"ingestion.pilot=true: {pilot_counts[True]}")
    print(
        "document_id=nist_sp_800_171_r3: "
        f"{document_ids['nist_sp_800_171_r3']}"
    )

    failures = []

    if type_counts["requirement"] != EXPECTED_ACTIVE:
        failures.append("active requirement count")

    if type_counts["withdrawn"] != EXPECTED_WITHDRAWN:
        failures.append("withdrawn record count")

    if normative_counts[True] != EXPECTED_ACTIVE:
        failures.append("normative=true count")

    if normative_counts[False] != EXPECTED_WITHDRAWN:
        failures.append("normative=false count")

    if pilot_counts[False] != EXPECTED_TOTAL or pilot_counts[True] != 0:
        failures.append("pilot finalization")

    if document_ids["nist_sp_800_171_r3"] != EXPECTED_TOTAL:
        failures.append("document_id count")

    if failures:
        raise SystemExit(
            "Final verification failed: " + ", ".join(failures)
        )

    print("\nRESUME-SAFE BULK INGEST RESULT: PASS")
    print(
        f"All {EXPECTED_TOTAL} audited NIST SP 800-171r3 records are finalized "
        f"in {COLLECTION}."
    )


def main() -> None:
    print("=" * 88)
    print("VIGILANT COMPLIANCE RESUME-SAFE NIST SP 800-171r3 BULK INGEST")
    print("=" * 88)

    records = load_records()
    records_by_id = {record["record_id"]: record for record in records}

    print(f"Dataset: {JSONL_PATH}")
    print(f"Total records: {len(records)}")
    print(f"Active requirements: {EXPECTED_ACTIVE}")
    print(f"Withdrawn records: {EXPECTED_WITHDRAWN}")
    print(f"Batch size: {BATCH_SIZE}")

    client = QdrantClient(url=QDRANT_URL)

    finalized_ids, pilot_ids = validate_existing_points(
        client,
        records_by_id,
    )

    missing_ids = set(records_by_id) - finalized_ids - pilot_ids

    print("\nExisting collection validation: PASS")
    print(f"Already finalized production points: {len(finalized_ids)}")
    print(f"Pilot points to finalize: {len(pilot_ids)}")
    print(f"Missing points to ingest: {len(missing_ids)}")

    pending_records = [
        record
        for record in records
        if record["record_id"] in pilot_ids
        or record["record_id"] in missing_ids
    ]

    print(f"Total records to process this run: {len(pending_records)}")

    if not pending_records:
        final_verify(client)
        return

    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    total_batches = (
        len(pending_records) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(pending_records), BATCH_SIZE),
        1,
    ):
        batch = pending_records[start:start + BATCH_SIZE]
        search_texts = [build_search_text(record) for record in batch]

        unload_after = batch_number == total_batches

        print(
            f"\nBatch {batch_number}/{total_batches}: "
            f"{len(batch)} records"
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

    final_verify(client)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

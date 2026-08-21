from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "DOD INSTRUCTION 8510.01"

# Start in dry-run mode so nothing is modified yet.
DRY_RUN = False

VIGILANT_METADATA = {
    "document_id": "dodi_8510_01",
    "document": "DoDI 8510.01",
    "document_type": "DoDI",
    "title": "Risk Management Framework for DoD Systems",
    "authority": "DoD CIO",
    "source_version_date": "2022-07-19",
    "domains": [
        "rmf",
        "cybersecurity"
    ],
    "automation_tags": [
        "rmf",
        "authorization",
        "continuous_monitoring"
    ]
}

client = QdrantClient(url=QDRANT_URL)

offset = None
total_found = 0
already_enriched = 0
updated = 0
would_update = 0
failures = 0

print(f"\nCollection: {COLLECTION}")
print(f"Dry run: {DRY_RUN}\n")

while True:
    points, next_offset = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        break

    for point in points:
        total_found += 1

        payload = point.payload or {}
        vigilant = payload.get("vigilant", {})

        if vigilant.get("document_id") == "dodi_8510_01":
            already_enriched += 1
            continue

        if DRY_RUN:
            would_update += 1
            continue

        try:
            client.set_payload(
                collection_name=COLLECTION,
                payload={
                    "vigilant": VIGILANT_METADATA
                },
                points=[point.id],
                wait=True
            )
            updated += 1

        except Exception as exc:
            failures += 1
            print(f"FAILED: {point.id}")
            print(exc)

    if next_offset is None:
        break

    offset = next_offset

print("\n------------------------------")
print("Enrichment summary")
print("------------------------------")
print(f"Points found:       {total_found}")
print(f"Already enriched:   {already_enriched}")
print(f"Would update:       {would_update}")
print(f"Points updated:     {updated}")
print(f"Failures:           {failures}")

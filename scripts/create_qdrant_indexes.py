from qdrant_client import QdrantClient, models

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "DOD INSTRUCTION 8510.01"

client = QdrantClient(url=QDRANT_URL)

indexes = [
    "vigilant.document_id",
    "vigilant.domains",
    "vigilant.automation_tags",
]

print(f"\nCreating payload indexes for: {COLLECTION}\n")

for field_name in indexes:
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True
        )

        print(f"[OK] {field_name}")

    except Exception as exc:
        print(f"[ERROR] {field_name}")
        print(exc)

print("\nIndex creation complete.")

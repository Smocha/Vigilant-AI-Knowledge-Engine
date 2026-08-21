import json
import urllib.request

from qdrant_client import QdrantClient, models


OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
QDRANT_URL = "http://127.0.0.1:6333"

MODEL = "embeddinggemma:latest"
COLLECTION = "DOD INSTRUCTION 8510.01"

QUERY = "What are the DoD requirements for RMF system authorization and continuous monitoring?"


def create_embedding(text):
    payload = {
        "model": MODEL,
        "input": text
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"][0]


print(f"\nEmbedding model: {MODEL}")
print(f"Query: {QUERY}\n")

query_vector = create_embedding(QUERY)

print(f"Embedding dimensions: {len(query_vector)}")

if len(query_vector) != 768:
    raise RuntimeError(
        f"Expected 768 dimensions, received {len(query_vector)}"
    )

client = QdrantClient(url=QDRANT_URL)

query_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="vigilant.document_id",
            match=models.MatchValue(
                value="dodi_8510_01"
            )
        )
    ]
)

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    query_filter=query_filter,
    limit=5,
    with_payload=True,
    with_vectors=False
).points

print("\nTop results:\n")

for number, point in enumerate(results, start=1):

    payload = point.payload or {}

    content = payload.get(
        "content",
        "NO CONTENT FOUND"
    )

    metadata = payload.get(
        "metadata",
        {}
    )

    line_info = (
        metadata
        .get("loc", {})
        .get("lines", {})
    )

    print("=" * 70)

    print(f"RESULT {number}")
    print(f"Score: {point.score:.4f}")
    print(f"Point ID: {point.id}")

    print(
        "Lines:",
        line_info.get("from", "?"),
        "-",
        line_info.get("to", "?")
    )

    print()

    print(content[:1000])

    print()

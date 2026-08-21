from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"

client = QdrantClient(url=QDRANT_URL)

collections = client.get_collections()

print("\nConnected to Qdrant successfully.\n")
print("Available collections:")

for collection in collections.collections:
    print(f"  -{collection.name}")

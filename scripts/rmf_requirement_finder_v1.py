import json
import socket
import sys
import time
import urllib.error
import urllib.request

from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_URL = "http://127.0.0.1:11434"

COLLECTION = "DOD INSTRUCTION 8510.01"
DOCUMENT_ID = "dodi_8510_01"

EMBED_MODEL = "embeddinggemma:latest"
CHAT_MODEL = "llama3.2:3b"

EXPECTED_EMBEDDING_DIMENSIONS = 768

# Reduced from 5 to 3 to keep the LLM context smaller.
TOP_K = 3

# Give Ollama up to 10 minutes on this host.
OLLAMA_TIMEOUT_SECONDS = 600

# Keep the model resident after requests.
KEEP_ALIVE = "30m"

# Prevent excessively long answers.
NUM_PREDICT = 700

# Prevent a single retrieved chunk from making the prompt huge.
MAX_CHARS_PER_SOURCE = 3000


# ============================================================
# HTTP / OLLAMA HELPERS
# ============================================================

def ollama_post(endpoint, payload):
    """
    Send a JSON request to the local Ollama API.
    """

    url = f"{OLLAMA_URL}{endpoint}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_TIMEOUT_SECONDS
        ) as response:

            raw = response.read().decode("utf-8")
            return json.loads(raw)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        raise RuntimeError(
            f"Ollama HTTP error {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}: {exc}"
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama timed out after "
            f"{OLLAMA_TIMEOUT_SECONDS} seconds."
        ) from exc


def warm_chat_model():
    """
    Load the chat model before performing the full RAG request.
    """

    print(
        f"Loading {CHAT_MODEL} into Ollama..."
    )

    started = time.time()

    result = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Reply with the single word READY."
                    )
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 5
            }
        }
    )

    elapsed = time.time() - started

    content = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    print(
        f"Chat model loaded in {elapsed:.1f} seconds."
    )

    if content:
        print(
            f"Warm-up response: {content}"
        )


def create_embedding(text):
    """
    Generate a 768-dimensional embedding using EmbeddingGemma.
    """

    result = ollama_post(
        "/api/embed",
        {
            "model": EMBED_MODEL,
            "input": text,
            "keep_alive": KEEP_ALIVE
        }
    )

    embeddings = result.get(
        "embeddings",
        []
    )

    if not embeddings:
        raise RuntimeError(
            "Ollama did not return an embedding."
        )

    vector = embeddings[0]

    dimensions = len(vector)

    if dimensions != EXPECTED_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Unexpected embedding dimensions. "
            f"Expected "
            f"{EXPECTED_EMBEDDING_DIMENSIONS}, "
            f"received {dimensions}."
        )

    return vector


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def search_qdrant(question):
    """
    Search only the enriched DoDI 8510.01 points.
    """

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=30
    )

    query_vector = create_embedding(
        question
    )

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="vigilant.document_id",
                match=models.MatchValue(
                    value=DOCUMENT_ID
                )
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False
    )

    return response.points


# ============================================================
# SOURCE PROCESSING
# ============================================================

def build_context(points):
    """
    Convert Qdrant results into grounded context for the LLM.
    """

    sections = []

    for number, point in enumerate(
        points,
        start=1
    ):
        payload = point.payload or {}

        content = payload.get(
            "content",
            ""
        )

        # Limit context size per source.
        content = content[
            :MAX_CHARS_PER_SOURCE
        ]

        metadata = payload.get(
            "metadata",
            {}
        )

        line_info = (
            metadata
            .get("loc", {})
            .get("lines", {})
        )

        line_start = line_info.get(
            "from",
            "unknown"
        )

        line_end = line_info.get(
            "to",
            "unknown"
        )

        vigilant = payload.get(
            "vigilant",
            {}
        )

        document = vigilant.get(
            "document",
            "DoDI 8510.01"
        )

        section = f"""
[SOURCE {number}]
Document: {document}
Point ID: {point.id}
Similarity Score: {point.score:.4f}
Lines: {line_start}-{line_end}

{content}
"""

        sections.append(section)

    return "\n".join(sections)


# ============================================================
# LLM ANALYSIS
# ============================================================

def generate_answer(question, context):
    """
    Generate a grounded RMF response using only retrieved evidence.
    """

    system_prompt = """
You are the Vigilant DoD RMF Requirement Finder.

You analyze authoritative passages retrieved from
DoDI 8510.01.

STRICT RULES:

1. Use only the supplied source passages as evidence.

2. Do not invent DoD requirements, responsibilities,
   controls, citations, page numbers, or policy language.

3. Clearly distinguish:
   - requirements stated in the source
   - interpretation of those requirements
   - recommended company actions

4. Cite factual statements using:
   [SOURCE 1]
   [SOURCE 2]
   [SOURCE 3]

5. If the retrieved evidence does not adequately answer
   the question, explicitly state:

   "The retrieved evidence is insufficient to establish
   this requirement."

6. Preserve official DoD terminology such as:
   RMF
   AO
   authorization
   authorization boundary
   reciprocity
   cybersecurity risk
   continuous monitoring

7. Do not claim that a recommendation is a formal DoD
   requirement unless the retrieved text establishes it.

8. Keep the answer concise and operationally useful.

Return the response using exactly these headings:

DIRECT ANSWER

DOD REQUIREMENTS

ROLES / RESPONSIBILITIES

RECOMMENDED COMPANY ACTIONS

SOURCE TRACEABILITY
"""

    user_prompt = f"""
QUESTION

{question}


RETRIEVED DOD SOURCE MATERIAL

{context}
"""

    started = time.time()

    result = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": NUM_PREDICT
            }
        }
    )

    elapsed = time.time() - started

    answer = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    return answer, elapsed


# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_retrieval_results(points):
    """
    Show which Qdrant points were selected.
    """

    print(
        f"\nRetrieved {len(points)} "
        f"passages from Qdrant:\n"
    )

    for number, point in enumerate(
        points,
        start=1
    ):
        payload = point.payload or {}

        metadata = payload.get(
            "metadata",
            {}
        )

        lines = (
            metadata
            .get("loc", {})
            .get("lines", {})
        )

        line_start = lines.get(
            "from",
            "?"
        )

        line_end = lines.get(
            "to",
            "?"
        )

        print(
            f"Source {number}"
        )

        print(
            f"  Score:    {point.score:.4f}"
        )

        print(
            f"  Point ID: {point.id}"
        )

        print(
            f"  Lines:    "
            f"{line_start}-{line_end}"
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print()
        print(
            "Usage:"
        )

        print(
            'python '
            'scripts/rmf_requirement_finder.py '
            '"your RMF question"'
        )

        print()

        sys.exit(1)

    question = " ".join(
        sys.argv[1:]
    )

    print()
    print(
        "=" * 70
    )

    print(
        "VIGILANT RMF REQUIREMENT FINDER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestion:\n{question}\n"
    )

    try:
        # ----------------------------------------------------
        # 1. Warm up Llama
        # ----------------------------------------------------

        print(
            "STEP 1: Preparing local LLM"
        )

        warm_chat_model()

        # ----------------------------------------------------
        # 2. Embed and retrieve
        # ----------------------------------------------------

        print()
        print(
            "STEP 2: Searching DoDI 8510.01"
        )

        print(
            f"Embedding model: {EMBED_MODEL}"
        )

        print(
            f"Qdrant collection: {COLLECTION}"
        )

        print(
            f"Metadata filter: "
            f"vigilant.document_id="
            f"{DOCUMENT_ID}"
        )

        points = search_qdrant(
            question
        )

        if not points:
            print(
                "\nNo matching Qdrant "
                "points were returned."
            )

            sys.exit(1)

        display_retrieval_results(
            points
        )

        # ----------------------------------------------------
        # 3. Build grounded evidence
        # ----------------------------------------------------

        context = build_context(
            points
        )

        # ----------------------------------------------------
        # 4. Ask Llama to analyze evidence
        # ----------------------------------------------------

        print(
            "STEP 3: Generating grounded RMF analysis"
        )

        print(
            f"Chat model: {CHAT_MODEL}"
        )

        print(
            f"Timeout: "
            f"{OLLAMA_TIMEOUT_SECONDS} seconds"
        )

        print(
            f"Maximum generated tokens: "
            f"{NUM_PREDICT}"
        )

        print()

        answer, elapsed = generate_answer(
            question,
            context
        )

        # ----------------------------------------------------
        # 5. Output
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "RMF ANALYSIS"
        )

        print(
            "=" * 70
        )

        print()

        print(
            answer
        )

        print()

        print(
            "=" * 70
        )

        print(
            f"LLM generation time: "
            f"{elapsed:.1f} seconds"
        )

        print(
            f"Sources used: "
            f"{len(points)}"
        )

        print(
            "=" * 70
        )

        print()

    except KeyboardInterrupt:

        print(
            "\nOperation cancelled."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(exc)
        )

        print()

        print(
            "Diagnostic commands:"
        )

        print(
            "  docker exec ollama ollama ps"
        )

        print(
            "  docker exec ollama ollama list"
        )

        print(
            "  curl http://127.0.0.1:11434/api/tags"
        )

        print(
            "  curl http://127.0.0.1:6333/collections"
        )

        print()

        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
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
DOCUMENT_NAME = "DoDI 8510.01"

EMBED_MODEL = "embeddinggemma:latest"
CHAT_MODEL = "llama3.2:3b"

EXPECTED_EMBEDDING_DIMENSIONS = 768

# Retrieve more candidates than we ultimately use.
CANDIDATE_K = 8

# Maximum number of passages sent to the LLM.
MAX_SOURCES = 3

# Heuristic similarity threshold.
# Tune this later using evaluation data.
MIN_SCORE = 0.55

OLLAMA_TIMEOUT_SECONDS = 600
KEEP_ALIVE = "30m"
NUM_PREDICT = 650

# Prevent giant chunks from overwhelming the 3B model.
MAX_CHARS_PER_SOURCE = 3000


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Vigilant RMF Requirement Finder - "
            "grounded DoDI 8510.01 analysis"
        )
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="RMF question to analyze"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Return machine-readable JSON only"
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the Ollama chat-model warm-up"
    )

    return parser.parse_args()


# ============================================================
# OLLAMA HTTP
# ============================================================

def ollama_post(endpoint, payload):
    url = f"{OLLAMA_URL}{endpoint}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
            f"Unable to connect to Ollama at {url}: {exc}"
        ) from exc

    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama timed out after "
            f"{OLLAMA_TIMEOUT_SECONDS} seconds."
        ) from exc


# ============================================================
# MODEL PREPARATION
# ============================================================

def warm_chat_model(quiet=False):
    if not quiet:
        print(f"Loading {CHAT_MODEL}...")

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
                    "content": "Reply with exactly READY."
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 5
            }
        }
    )

    elapsed = time.time() - started

    response_text = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not quiet:
        print(
            f"Model warm-up completed "
            f"in {elapsed:.1f} seconds."
        )

        if response_text:
            print(f"Warm-up response: {response_text}")


def create_embedding(text):
    result = ollama_post(
        "/api/embed",
        {
            "model": EMBED_MODEL,
            "input": text,
            "keep_alive": KEEP_ALIVE
        }
    )

    embeddings = result.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no embedding."
        )

    vector = embeddings[0]

    dimensions = len(vector)

    if dimensions != EXPECTED_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Embedding dimension mismatch. "
            f"Expected {EXPECTED_EMBEDDING_DIMENSIONS}, "
            f"received {dimensions}."
        )

    return vector


# ============================================================
# QDRANT SEARCH
# ============================================================

def search_qdrant(question):
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=30
    )

    query_vector = create_embedding(question)

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
        limit=CANDIDATE_K,
        with_payload=True,
        with_vectors=False
    )

    qualified = []

    for point in response.points:
        if point.score is None:
            continue

        if point.score < MIN_SCORE:
            continue

        qualified.append(point)

    return qualified[:MAX_SOURCES]


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_source(point, source_number):
    payload = point.payload or {}

    content = payload.get("content", "")

    metadata = payload.get(
        "metadata",
        {}
    )

    lines = (
        metadata
        .get("loc", {})
        .get("lines", {})
    )

    vigilant = payload.get(
        "vigilant",
        {}
    )

    return {
        "source_number": source_number,
        "document": vigilant.get(
            "document",
            DOCUMENT_NAME
        ),
        "document_id": vigilant.get(
            "document_id",
            DOCUMENT_ID
        ),
        "point_id": str(point.id),
        "score": round(float(point.score), 4),
        "line_start": lines.get("from"),
        "line_end": lines.get("to"),
        "content": content
    }


def build_sources(points):
    sources = []

    for number, point in enumerate(
        points,
        start=1
    ):
        sources.append(
            extract_source(
                point,
                number
            )
        )

    return sources


def build_llm_context(sources):
    context_parts = []

    for source in sources:
        text = (
            source.get("content", "")
            [:MAX_CHARS_PER_SOURCE]
        )

        context_parts.append(
            f"""
[SOURCE {source['source_number']}]

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# LLM OUTPUT PARSING
# ============================================================

def strip_json_fence(text):
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def parse_llm_json(text):
    cleaned = strip_json_fence(text)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The LLM did not return valid JSON.\n\n"
            f"Raw response:\n{text}"
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError(
            "The LLM response was not a JSON object."
        )

    return result


# ============================================================
# GROUNDED ANALYSIS
# ============================================================

def generate_analysis(question, sources):
    context = build_llm_context(sources)

    allowed_source_labels = [
        f"SOURCE {source['source_number']}"
        for source in sources
    ]

    system_prompt = f"""
You are the Vigilant DoD RMF Requirement Finder.

You analyze ONLY the retrieved passages supplied to you
from {DOCUMENT_NAME}.

You are not permitted to use outside knowledge as evidence.

STRICT REQUIREMENTS:

1. Use only the supplied passages.

2. Do not invent:
   - DoD requirements
   - page numbers
   - section numbers
   - paragraph numbers
   - document titles
   - policy citations
   - control identifiers
   - responsibilities

3. Do not generate source traceability metadata.
   The application will generate traceability separately.

4. You may cite evidence only using these labels:

   {", ".join(allowed_source_labels)}

5. When a requirement is not established by the retrieved
   evidence, say so explicitly.

6. Never convert a recommended company action into a formal
   DoD requirement.

7. Preserve official terms found in the passages.

8. Keep conclusions concise.

Return ONLY valid JSON.

Do not use Markdown fences.

Use exactly this schema:

{{
  "direct_answer": "string",
  "evidence_status": "supported | partial | insufficient",
  "dod_requirements": [
    {{
      "requirement": "string",
      "sources": ["SOURCE 1"]
    }}
  ],
  "roles_responsibilities": [
    {{
      "role": "string",
      "responsibility": "string",
      "sources": ["SOURCE 1"]
    }}
  ],
  "recommended_company_actions": [
    "string"
  ],
  "limitations": [
    "string"
  ]
}}
"""

    user_prompt = f"""
QUESTION:

{question}


RETRIEVED EVIDENCE:

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

    raw_answer = (
        result
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not raw_answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    parsed = parse_llm_json(
        raw_answer
    )

    return parsed, elapsed


# ============================================================
# INSUFFICIENT-EVIDENCE RESULT
# ============================================================

def insufficient_result(question):
    return {
        "question": question,
        "document": DOCUMENT_NAME,
        "document_id": DOCUMENT_ID,
        "status": "insufficient_evidence",
        "minimum_score": MIN_SCORE,
        "analysis": {
            "direct_answer": (
                "The retrieved evidence is insufficient "
                "to establish this requirement."
            ),
            "evidence_status": "insufficient",
            "dod_requirements": [],
            "roles_responsibilities": [],
            "recommended_company_actions": [],
            "limitations": [
                (
                    "No retrieved passage met the configured "
                    f"similarity threshold of {MIN_SCORE}."
                )
            ]
        },
        "sources": [],
        "generation_seconds": 0
    }


# ============================================================
# FINAL REPORT
# ============================================================

def build_report(
    question,
    analysis,
    sources,
    generation_seconds
):
    traceability = []

    for source in sources:
        traceability.append(
            {
                "source": (
                    f"SOURCE "
                    f"{source['source_number']}"
                ),
                "document": source["document"],
                "document_id": source["document_id"],
                "point_id": source["point_id"],
                "line_start": source["line_start"],
                "line_end": source["line_end"],
                "similarity_score": source["score"]
            }
        )

    return {
        "question": question,
        "document": DOCUMENT_NAME,
        "document_id": DOCUMENT_ID,
        "status": analysis.get(
            "evidence_status",
            "unknown"
        ),
        "retrieval": {
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "minimum_score": MIN_SCORE,
            "candidate_count": CANDIDATE_K,
            "sources_used": len(sources)
        },
        "analysis": analysis,
        "sources": traceability,
        "generation_seconds": round(
            generation_seconds,
            1
        )
    }


# ============================================================
# HUMAN-READABLE OUTPUT
# ============================================================

def print_human_report(report):
    analysis = report["analysis"]

    print()
    print("=" * 70)
    print("VIGILANT RMF REQUIREMENT FINDER")
    print("=" * 70)

    print()
    print("QUESTION")
    print(report["question"])

    print()
    print("DIRECT ANSWER")
    print(
        analysis.get(
            "direct_answer",
            "No answer returned."
        )
    )

    print()
    print("EVIDENCE STATUS")
    print(
        analysis.get(
            "evidence_status",
            "unknown"
        ).upper()
    )

    print()
    print("DOD REQUIREMENTS")

    requirements = analysis.get(
        "dod_requirements",
        []
    )

    if requirements:
        for item in requirements:
            print(
                f"- {item.get('requirement', '')}"
            )

            citations = item.get(
                "sources",
                []
            )

            if citations:
                print(
                    f"  Evidence: "
                    f"{', '.join(citations)}"
                )
    else:
        print(
            "- No supported requirement extracted."
        )

    print()
    print("ROLES / RESPONSIBILITIES")

    roles = analysis.get(
        "roles_responsibilities",
        []
    )

    if roles:
        for item in roles:
            role = item.get(
                "role",
                "Unspecified role"
            )

            responsibility = item.get(
                "responsibility",
                ""
            )

            print(
                f"- {role}: {responsibility}"
            )

            citations = item.get(
                "sources",
                []
            )

            if citations:
                print(
                    f"  Evidence: "
                    f"{', '.join(citations)}"
                )
    else:
        print(
            "- No supported role information extracted."
        )

    print()
    print("RECOMMENDED COMPANY ACTIONS")

    actions = analysis.get(
        "recommended_company_actions",
        []
    )

    if actions:
        for action in actions:
            print(f"- {action}")
    else:
        print(
            "- No recommendation generated."
        )

    limitations = analysis.get(
        "limitations",
        []
    )

    if limitations:
        print()
        print("LIMITATIONS")

        for limitation in limitations:
            print(f"- {limitation}")

    print()
    print("SOURCE TRACEABILITY")

    sources = report.get(
        "sources",
        []
    )

    if not sources:
        print(
            "No qualifying source passages."
        )

    for source in sources:
        print()
        print(source["source"])
        print(
            f"  Document: {source['document']}"
        )
        print(
            f"  Point ID: {source['point_id']}"
        )

        line_start = source.get(
            "line_start"
        )

        line_end = source.get(
            "line_end"
        )

        if (
            line_start is not None
            and line_end is not None
        ):
            print(
                f"  Lines: {line_start}-{line_end}"
            )
        else:
            print(
                "  Lines: unavailable"
            )

        print(
            "  Similarity: "
            f"{source['similarity_score']:.4f}"
        )

    print()
    print("=" * 70)
    print(
        f"Sources used: "
        f"{report['retrieval']['sources_used']}"
    )
    print(
        f"Minimum similarity: "
        f"{report['retrieval']['minimum_score']}"
    )
    print(
        f"LLM generation time: "
        f"{report['generation_seconds']} seconds"
    )
    print("=" * 70)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    question = " ".join(
        args.question
    ).strip()

    quiet = args.json_output

    try:
        if not args.no_warmup:
            warm_chat_model(
                quiet=quiet
            )

        if not quiet:
            print()
            print(
                f"Searching {DOCUMENT_NAME}..."
            )
            print(
                f"Similarity threshold: {MIN_SCORE}"
            )

        points = search_qdrant(
            question
        )

        if not points:
            report = insufficient_result(
                question
            )

            if args.json_output:
                print(
                    json.dumps(
                        report,
                        indent=2
                    )
                )
            else:
                print_human_report(
                    report
                )

            return

        sources = build_sources(
            points
        )

        if not quiet:
            print(
                f"Qualified sources: "
                f"{len(sources)}"
            )

            for source in sources:
                print(
                    f"  SOURCE "
                    f"{source['source_number']}: "
                    f"{source['score']:.4f}"
                )

            print()
            print(
                f"Generating analysis with "
                f"{CHAT_MODEL}..."
            )

        analysis, elapsed = generate_analysis(
            question,
            sources
        )

        report = build_report(
            question,
            analysis,
            sources,
            elapsed
        )

        if args.json_output:
            print(
                json.dumps(
                    report,
                    indent=2
                )
            )
        else:
            print_human_report(
                report
            )

    except KeyboardInterrupt:
        if not args.json_output:
            print(
                "\nOperation cancelled."
            )

        sys.exit(130)

    except Exception as exc:
        error_result = {
            "status": "error",
            "error": str(exc)
        }

        if args.json_output:
            print(
                json.dumps(
                    error_result,
                    indent=2
                )
            )
        else:
            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)
            print(str(exc))
            print()
            print("Useful diagnostics:")
            print(
                "  docker exec ollama ollama ps"
            )
            print(
                "  docker exec ollama ollama list"
            )
            print(
                "  curl "
                "http://127.0.0.1:11434/api/tags"
            )
            print(
                "  curl "
                "http://127.0.0.1:6333/collections"
            )
            print()

        sys.exit(1)


if __name__ == "__main__":
    main()

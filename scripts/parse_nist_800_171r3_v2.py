#!/usr/bin/env python3
"""
Parse NIST SP 800-171 Rev. 3 text produced by `pdftotext -layout`
into structured JSONL compliance requirement records.

This parser is deterministic:
- no LLM
- no Qdrant writes
- one JSONL record per lettered requirement clause
- preserves requirement ID, title, clause, source controls, and source line numbers
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SOURCE_PDF = Path("/home/vigilantceo/documents/NIST.SP.800-171r3.pdf")
SOURCE_TEXT = Path(
    "/home/vigilantceo/vigilant-automation/data/source_text/NIST.SP.800-171r3.txt"
)
OUTPUT_PATH = Path(
    "/home/vigilantceo/vigilant-automation/data/structured/"
    "nist_sp_800_171_r3_requirements.jsonl"
)

SOURCE_SHA256 = (
    "3e4631df8b5d61f40a6e542b52779ef30ddbbfff31e09214fa94ad6e6f5e6d08"
)

DOCUMENT = {
    "document_id": "nist_sp_800_171_r3",
    "title": (
        "Protecting Controlled Unclassified Information "
        "in Nonfederal Systems and Organizations"
    ),
    "document_type": "NIST SP",
    "document_number": "800-171",
    "revision": "r3",
    "authority": "NIST",
    "publication_date": "2024-05",
    "source_sha256": SOURCE_SHA256,
}

FAMILY_NAMES = {
    "03.01": "Access Control",
    "03.02": "Awareness and Training",
    "03.03": "Audit and Accountability",
    "03.04": "Configuration Management",
    "03.05": "Identification and Authentication",
    "03.06": "Incident Response",
    "03.07": "Maintenance",
    "03.08": "Media Protection",
    "03.09": "Personnel Security",
    "03.10": "Physical Protection",
    "03.11": "Risk Assessment",
    "03.12": "Security Assessment and Monitoring",
    "03.13": "System and Communications Protection",
    "03.14": "System and Information Integrity",
    "03.15": "Planning",
    "03.16": "System and Services Acquisition",
    "03.17": "Supply Chain Risk Management",
}

HEADER_RE = re.compile(r"^\s*(03\.\d{2}\.\d{2})\s+(.+?)\s*$")
SECTION3_START_RE = re.compile(r"^\s*3\.1\.\s+Access Control\s*$", re.IGNORECASE)
APPENDIX_A_RE = re.compile(r"^\s*Appendix A\.\s+Acronyms\s*$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^\s*([a-z])\.\s+(.*\S)?\s*$", re.IGNORECASE)
SOURCE_CONTROLS_RE = re.compile(
    r"^\s*Source Controls?:\s*(.*?)\s*$", re.IGNORECASE
)
SUPPORTING_PUBS_RE = re.compile(
    r"^\s*Supporting Publications?:\s*(.*?)\s*$", re.IGNORECASE
)

BLOCK_SECTION_MARKERS = {
    "DISCUSSION",
    "REFERENCES",
}

NOISE_PATTERNS = (
    re.compile(
        r"^\s*NIST SP 800-171r3 Protecting Controlled Unclassified Information\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*May 2024\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(pattern.match(line) for pattern in NOISE_PATTERNS)


def join_wrapped(parts: list[str]) -> str:
    """Join PDF-wrapped lines without damaging legitimate hyphenated words."""
    result = ""
    for raw in parts:
        piece = re.sub(r"\s+", " ", raw.strip())
        if not piece:
            continue
        if not result:
            result = piece
        elif result.endswith("-"):
            result += piece
        else:
            result += " " + piece
    return re.sub(r"\s+", " ", result).strip()


def parse_csvish(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_references(block_lines: list[str]) -> tuple[list[str], list[str]]:
    source_controls: list[str] = []
    supporting_publications: list[str] = []

    for line in block_lines:
        match = SOURCE_CONTROLS_RE.match(line)
        if match:
            source_controls.extend(parse_csvish(match.group(1)))
            continue

        match = SUPPORTING_PUBS_RE.match(line)
        if match:
            supporting_publications.extend(parse_csvish(match.group(1)))

    # Stable de-duplication.
    source_controls = list(dict.fromkeys(source_controls))
    supporting_publications = list(dict.fromkeys(supporting_publications))
    return source_controls, supporting_publications


def parse_requirement_clauses(
    lines: list[str],
    block_start: int,
    block_end: int,
) -> list[dict]:
    """
    Parse requirement body from immediately after header until DISCUSSION/REFERENCES.

    Returns:
      [{"clause": "a", "text": "...", "line_start": 123, "line_end": 125}, ...]
    """
    clauses: list[dict] = []
    current_clause: str | None = None
    current_parts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    base_parts: list[str] = []
    base_start: int | None = None
    base_end: int | None = None

    def flush_clause() -> None:
        nonlocal current_clause, current_parts, current_start, current_end
        if current_clause is None:
            return
        text = join_wrapped(current_parts)
        if text:
            clauses.append(
                {
                    "clause": current_clause.lower(),
                    "text": text,
                    "line_start": current_start,
                    "line_end": current_end,
                }
            )
        current_clause = None
        current_parts = []
        current_start = None
        current_end = None

    # +1 skips the requirement header itself.
    for idx in range(block_start + 1, block_end):
        raw = lines[idx]
        stripped = raw.strip()
        line_no = idx + 1

        if stripped.upper() in BLOCK_SECTION_MARKERS:
            flush_clause()
            break

        # Defensive stop if another header somehow appears.
        if HEADER_RE.match(raw):
            flush_clause()
            break

        clause_match = CLAUSE_RE.match(raw)
        if clause_match:
            flush_clause()
            current_clause = clause_match.group(1).lower()
            first_text = (clause_match.group(2) or "").strip()
            current_parts = [first_text] if first_text else []
            current_start = line_no
            current_end = line_no
            continue

        if is_noise_line(raw):
            continue

        if current_clause is not None:
            current_parts.append(stripped)
            current_end = line_no
        else:
            # Some standards records may contain an unlettered normative body.
            # Preserve it as a "base" clause only if no lettered clauses are found.
            if base_start is None:
                base_start = line_no
            base_parts.append(stripped)
            base_end = line_no

    flush_clause()

    if not clauses:
        base_text = join_wrapped(base_parts)
        if base_text:
            clauses.append(
                {
                    "clause": "base",
                    "text": base_text,
                    "line_start": base_start,
                    "line_end": base_end,
                }
            )

    return clauses


def find_authoritative_section3_bounds(lines: list[str]) -> tuple[int, int]:
    """
    Locate the authoritative NIST SP 800-171r3 Section 3 requirement body.

    This intentionally excludes:
    - the illustrative example before Section 3
    - Appendix A and all later appendices/tables
    - NIST SP 800-53 mapping tables
    - ODP/reference tables
    """
    start_idx: int | None = None
    end_idx: int | None = None

    for idx, line in enumerate(lines):
        if start_idx is None and SECTION3_START_RE.match(line):
            start_idx = idx
            continue

        if start_idx is not None and APPENDIX_A_RE.match(line):
            end_idx = idx
            break

    if start_idx is None:
        raise SystemExit(
            "Could not find authoritative Section 3 start: '3.1. Access Control'."
        )

    if end_idx is None:
        raise SystemExit(
            "Could not find Section 3 end boundary: 'Appendix A. Acronyms'."
        )

    if end_idx <= start_idx:
        raise SystemExit("Invalid Section 3 boundaries detected.")

    return start_idx, end_idx


def parse() -> list[dict]:
    if not SOURCE_TEXT.exists():
        raise SystemExit(f"Source text does not exist: {SOURCE_TEXT}")

    lines = SOURCE_TEXT.read_text(encoding="utf-8", errors="replace").splitlines()

    section_start, section_end = find_authoritative_section3_bounds(lines)

    headers: list[tuple[int, str, str]] = []
    for idx in range(section_start, section_end):
        match = HEADER_RE.match(lines[idx])
        if match:
            headers.append((idx, match.group(1), match.group(2).strip()))

    if not headers:
        raise SystemExit(
            "No NIST 03.xx.xx requirement headers were found inside authoritative Section 3."
        )

    records: list[dict] = []

    for header_index, (start_idx, requirement_id, title) in enumerate(headers):
        end_idx = (
            headers[header_index + 1][0]
            if header_index + 1 < len(headers)
            else section_end
        )

        block_lines = lines[start_idx:end_idx]
        source_controls, supporting_publications = extract_references(block_lines)

        clauses = parse_requirement_clauses(lines, start_idx, end_idx)
        family_id = ".".join(requirement_id.split(".")[:2])
        family_name = FAMILY_NAMES.get(family_id, "Unknown")

        for clause in clauses:
            record = {
                "schema_version": "1.0",
                "record_id": (
                    f"{DOCUMENT['document_id']}|{requirement_id}|{clause['clause']}"
                ),
                "document": DOCUMENT.copy(),
                "record_type": "requirement",
                "normative": True,
                "family": {
                    "family_id": family_id,
                    "family_name": family_name,
                },
                "requirement": {
                    "requirement_id": requirement_id,
                    "title": title,
                    "clause": clause["clause"],
                    "text": clause["text"],
                },
                "references": {
                    "source_controls": source_controls,
                    "supporting_publications": supporting_publications,
                },
                "source": {
                    "source_file": str(SOURCE_PDF),
                    "text_file": str(SOURCE_TEXT),
                    "line_start": clause["line_start"],
                    "line_end": clause["line_end"],
                    "pdf_page": None,
                },
            }
            records.append(record)

    # Expose the detected window to main() without changing every function signature.
    parse.section_start_line = section_start + 1
    parse.section_end_line = section_end
    return records


def validate(records: list[dict]) -> None:
    ids = [record["record_id"] for record in records]
    duplicates = [rid for rid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(
            "Duplicate record_id values found:\n" + "\n".join(duplicates[:20])
        )

    required_examples = {"03.01.08", "03.07.05", "03.08.05"}
    parsed_ids = {
        record["requirement"]["requirement_id"] for record in records
    }
    missing = sorted(required_examples - parsed_ids)
    if missing:
        raise SystemExit(
            "Parser validation failed. Missing known requirement IDs: "
            + ", ".join(missing)
        )

    empty = [
        record["record_id"]
        for record in records
        if not record["requirement"]["text"].strip()
    ]
    if empty:
        raise SystemExit(
            "Parser validation failed. Empty requirement text found for: "
            + ", ".join(empty[:20])
        )


def main() -> None:
    records = parse()
    validate(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    requirement_ids = {
        record["requirement"]["requirement_id"] for record in records
    }
    family_counts = Counter(
        record["family"]["family_id"] for record in records
    )

    print(f"Output: {OUTPUT_PATH}")
    print(
        "Authoritative Section 3 window: "
        f"lines {getattr(parse, 'section_start_line', '?')} "
        f"through {getattr(parse, 'section_end_line', '?')} "
        "(stops before Appendix A)"
    )
    print(f"Requirement IDs parsed: {len(requirement_ids)}")
    print(f"Clause records written: {len(records)}")
    print(f"Unique record IDs: {len({r['record_id'] for r in records})}")
    print("Family record counts:")
    for family_id in sorted(family_counts):
        print(
            f"  {family_id} "
            f"{FAMILY_NAMES.get(family_id, 'Unknown')}: "
            f"{family_counts[family_id]}"
        )

    print("\nKnown-control validation: PASS")
    for wanted in ("03.01.08", "03.07.05", "03.08.05"):
        matches = [
            r for r in records
            if r["requirement"]["requirement_id"] == wanted
        ]
        print(f"\n{wanted} {matches[0]['requirement']['title']}")
        for record in matches:
            clause = record["requirement"]["clause"]
            text = record["requirement"]["text"]
            start = record["source"]["line_start"]
            end = record["source"]["line_end"]
            print(f"  {clause}. [{start}-{end}] {text}")


if __name__ == "__main__":
    main()

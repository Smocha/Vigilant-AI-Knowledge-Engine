#!/usr/bin/env python3
"""
Integrity audit for structured NIST SP 800-171r3 requirement JSONL.

Read-only:
- does not modify JSONL
- does not write to Qdrant
- verifies IDs, clauses, provenance, source-text alignment, and common leakage
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

JSONL_PATH = Path(
    "/home/vigilantceo/vigilant-automation/data/structured/"
    "nist_sp_800_171_r3_requirements.jsonl"
)
SOURCE_TEXT_PATH = Path(
    "/home/vigilantceo/vigilant-automation/data/source_text/NIST.SP.800-171r3.txt"
)

REQ_ID_RE = re.compile(r"^03\.\d{2}\.\d{2}$")
VALID_CLAUSE_RE = re.compile(r"^(?:[a-z]|base)$")

EXPECTED_FAMILIES = {
    "03.01", "03.02", "03.03", "03.04", "03.05", "03.06",
    "03.07", "03.08", "03.09", "03.10", "03.11", "03.12",
    "03.13", "03.14", "03.15", "03.16", "03.17",
}

LEAKAGE_PATTERNS = [
    re.compile(r"\bDISCUSSION\b", re.IGNORECASE),
    re.compile(r"\bREFERENCES\b", re.IGNORECASE),
    re.compile(r"\bSupporting Publications?:\b", re.IGNORECASE),
    re.compile(r"\bSource Controls?:\b", re.IGNORECASE),
    re.compile(r"NIST SP 800-171r3 Protecting Controlled Unclassified Information", re.IGNORECASE),
    re.compile(r"\bMay 2024\b", re.IGNORECASE),
]


def normalize(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_jsonl() -> list[dict]:
    if not JSONL_PATH.exists():
        raise SystemExit(f"Missing JSONL: {JSONL_PATH}")

    records = []
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on JSONL line {line_no}: {exc}") from exc
    return records


def main() -> None:
    records = load_jsonl()

    if not SOURCE_TEXT_PATH.exists():
        raise SystemExit(f"Missing source text: {SOURCE_TEXT_PATH}")

    # Literal newline split keeps line numbers aligned with grep -n.
    source_lines = SOURCE_TEXT_PATH.read_text(
        encoding="utf-8", errors="replace"
    ).split("\n")

    errors: list[str] = []
    warnings: list[str] = []

    record_ids = [r.get("record_id") for r in records]
    duplicate_ids = [rid for rid, n in Counter(record_ids).items() if n > 1]
    if duplicate_ids:
        errors.append(f"Duplicate record_id values: {duplicate_ids[:20]}")

    family_counts = Counter()
    requirement_counts = Counter()
    clause_counts = Counter()
    base_records = []
    leakage_hits = []
    provenance_mismatches = []
    malformed_ids = []
    malformed_clauses = []
    empty_text = []
    bad_line_ranges = []

    by_requirement = defaultdict(list)

    for record in records:
        record_id = record.get("record_id", "")
        family = record.get("family", {})
        requirement = record.get("requirement", {})
        source = record.get("source", {})

        family_id = family.get("family_id", "")
        requirement_id = requirement.get("requirement_id", "")
        clause = requirement.get("clause", "")
        text = requirement.get("text", "")
        line_start = source.get("line_start")
        line_end = source.get("line_end")

        family_counts[family_id] += 1
        requirement_counts[requirement_id] += 1
        clause_counts[clause] += 1
        by_requirement[requirement_id].append(record)

        if not REQ_ID_RE.match(requirement_id):
            malformed_ids.append((record_id, requirement_id))

        if not VALID_CLAUSE_RE.match(str(clause)):
            malformed_clauses.append((record_id, clause))

        if clause == "base":
            base_records.append(record)

        if not str(text).strip():
            empty_text.append(record_id)

        for pattern in LEAKAGE_PATTERNS:
            if pattern.search(str(text)):
                leakage_hits.append((record_id, pattern.pattern, text[:180]))
                break

        if (
            not isinstance(line_start, int)
            or not isinstance(line_end, int)
            or line_start < 1
            or line_end < line_start
            or line_end > len(source_lines)
        ):
            bad_line_ranges.append((record_id, line_start, line_end))
            continue

        source_slice = " ".join(
            s.strip() for s in source_lines[line_start - 1: line_end]
        )
        if normalize(text) not in normalize(source_slice):
            provenance_mismatches.append(
                {
                    "record_id": record_id,
                    "line_start": line_start,
                    "line_end": line_end,
                    "record_text": text,
                    "source_slice": source_slice,
                }
            )

    seen_families = set(family_counts)
    missing_families = sorted(EXPECTED_FAMILIES - seen_families)
    extra_families = sorted(seen_families - EXPECTED_FAMILIES)

    if missing_families:
        errors.append(f"Missing expected families: {missing_families}")
    if extra_families:
        warnings.append(f"Unexpected families: {extra_families}")
    if malformed_ids:
        errors.append(f"Malformed requirement IDs: {malformed_ids[:20]}")
    if malformed_clauses:
        errors.append(f"Malformed clauses: {malformed_clauses[:20]}")
    if empty_text:
        errors.append(f"Empty requirement text: {empty_text[:20]}")
    if bad_line_ranges:
        errors.append(f"Bad provenance line ranges: {bad_line_ranges[:20]}")
    if provenance_mismatches:
        errors.append(
            f"Requirement/source provenance mismatches: {len(provenance_mismatches)}"
        )
    if leakage_hits:
        errors.append(f"Section/header leakage found: {len(leakage_hits)}")

    # Known exact expectations from manual inspection.
    expected_examples = {
        ("03.01.08", "a"): "Enforce a limit of [Assignment: organization-defined number] consecutive invalid logon attempts by a user during a [Assignment: organization-defined time period].",
        ("03.07.05", "b"): "Implement multi-factor authentication and replay resistance in the establishment of nonlocal maintenance and diagnostic sessions.",
        ("03.08.05", "c"): "Document activities associated with the transport of system media that contain CUI.",
    }

    exact_example_failures = []
    for (req_id, clause), expected_text in expected_examples.items():
        matches = [
            r for r in records
            if r["requirement"]["requirement_id"] == req_id
            and r["requirement"]["clause"] == clause
        ]
        if len(matches) != 1:
            exact_example_failures.append(
                f"{req_id}|{clause}: expected 1 record, found {len(matches)}"
            )
            continue
        actual = matches[0]["requirement"]["text"]
        if normalize(actual) != normalize(expected_text):
            exact_example_failures.append(
                f"{req_id}|{clause}: text mismatch\n  expected={expected_text}\n  actual={actual}"
            )

    if exact_example_failures:
        errors.append(
            "Known-control exact checks failed:\n" + "\n".join(exact_example_failures)
        )

    print("=" * 80)
    print("NIST SP 800-171r3 STRUCTURED DATA AUDIT")
    print("=" * 80)
    print(f"JSONL: {JSONL_PATH}")
    print(f"Source text: {SOURCE_TEXT_PATH}")
    print(f"Records: {len(records)}")
    print(f"Unique record IDs: {len(set(record_ids))}")
    print(f"Unique requirement IDs: {len(requirement_counts)}")
    print(f"Families represented: {len(family_counts)}")
    print(f"Base/unlettered records: {len(base_records)}")
    print(f"Provenance mismatches: {len(provenance_mismatches)}")
    print(f"Leakage hits: {len(leakage_hits)}")

    print("\nFamily counts:")
    for family_id in sorted(family_counts):
        print(f"  {family_id}: {family_counts[family_id]}")

    print("\nClause counts:")
    for clause, count in sorted(clause_counts.items()):
        print(f"  {clause}: {count}")

    if base_records:
        print("\nBase/unlettered requirement records:")
        for r in base_records[:30]:
            req = r["requirement"]
            print(
                f"  {req['requirement_id']} {req['title']}: "
                f"{req['text'][:180]}"
            )

    if provenance_mismatches:
        print("\nFirst provenance mismatches:")
        for item in provenance_mismatches[:10]:
            print(f"  {item['record_id']} [{item['line_start']}-{item['line_end']}]")
            print(f"    RECORD: {item['record_text'][:220]}")
            print(f"    SOURCE: {item['source_slice'][:220]}")

    if leakage_hits:
        print("\nFirst leakage hits:")
        for record_id, pattern, sample in leakage_hits[:10]:
            print(f"  {record_id}: {sample}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nKnown-control exact checks:")
    if exact_example_failures:
        print("  FAIL")
    else:
        print("  PASS")

    print("\nAUDIT RESULT:")
    if errors:
        print("  FAIL")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  PASS")
    print("  Structured JSONL is internally consistent and provenance-aligned.")


if __name__ == "__main__":
    main()

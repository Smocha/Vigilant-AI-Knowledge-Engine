import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RMF_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "rmf_requirement_finder.py"
)

COMPLIANCE_RETRIEVER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "compliance_retriever_v1.py"
)

RMF_TIMEOUT_SECONDS = 720
COMPLIANCE_TIMEOUT_SECONDS = 300


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Vigilant Cyber Compliance API",
    description=(
        "API interface for the Vigilant DoD RMF "
        "Requirement Finder and structured compliance retrieval."
    ),
    version="1.1.0",
)


# ============================================================
# REQUEST MODELS
# ============================================================

class RMFRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="RMF or DoDI 8510.01 question to analyze.",
    )


class ComplianceSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Compliance requirement search query.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of structured requirements to return.",
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "vigilant-rmf-api",
        "rmf_script_exists": RMF_SCRIPT.exists(),
        "compliance_retriever_script_exists": COMPLIANCE_RETRIEVER_SCRIPT.exists(),
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Vigilant Cyber Compliance API",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "rmf_analyze": "/api/v1/rmf/analyze",
            "compliance_search": "/api/v1/compliance/search",
            "docs": "/docs",
        },
    }


# ============================================================
# COMPLIANCE SEARCH
# ============================================================

@app.post("/api/v1/compliance/search")
async def search_compliance(request: ComplianceSearchRequest):

    if not COMPLIANCE_RETRIEVER_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Compliance retriever not found at "
                f"{COMPLIANCE_RETRIEVER_SCRIPT}"
            ),
        )

    command = [
        sys.executable,
        str(COMPLIANCE_RETRIEVER_SCRIPT),
        request.query,
        "--limit",
        str(request.limit),
        "--json",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=COMPLIANCE_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()

            raise HTTPException(
                status_code=504,
                detail=(
                    "Compliance retrieval exceeded "
                    f"{COMPLIANCE_TIMEOUT_SECONDS} seconds."
                ),
            )

        stdout_text = stdout.decode(
            "utf-8",
            errors="replace",
        ).strip()

        stderr_text = stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Compliance retriever failed.",
                    "return_code": process.returncode,
                    "stderr": stderr_text,
                    "stdout": stdout_text,
                },
            )

        if not stdout_text:
            raise HTTPException(
                status_code=500,
                detail="Compliance retriever returned no output.",
            )

        try:
            result = json.loads(stdout_text)

        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "Compliance retriever did not return "
                        "valid JSON."
                    ),
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                },
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected compliance API error: {exc}",
        ) from exc


# ============================================================
# RMF ANALYSIS
# ============================================================

@app.post("/api/v1/rmf/analyze")
async def analyze_rmf(request: RMFRequest):

    if not RMF_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"RMF engine not found at "
                f"{RMF_SCRIPT}"
            ),
        )

    command = [
        sys.executable,
        str(RMF_SCRIPT),
        request.question,
        "--json",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=RMF_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError:
            process.kill()

            await process.communicate()

            raise HTTPException(
                status_code=504,
                detail=(
                    "RMF analysis exceeded "
                    f"{RMF_TIMEOUT_SECONDS} seconds."
                ),
            )

        stdout_text = stdout.decode(
            "utf-8",
            errors="replace",
        ).strip()

        stderr_text = stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "RMF engine failed.",
                    "return_code": process.returncode,
                    "stderr": stderr_text,
                    "stdout": stdout_text,
                },
            )

        if not stdout_text:
            raise HTTPException(
                status_code=500,
                detail="RMF engine returned no output.",
            )

        try:
            result = json.loads(
                stdout_text
            )

        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "RMF engine did not return "
                        "valid JSON."
                    ),
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                },
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected API error: {exc}",
        ) from exc

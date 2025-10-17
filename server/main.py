#!/usr/bin/env python3

import os
from typing import List, Optional
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env")
except ImportError:
    print("Using system environment only")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from cerebras_client import CerebrasAnalyzer
from bug_db import BugDatabase


class CodeAnalysisRequest(BaseModel):
    code: str
    language: str
    filename: str
    snarkLevel: str = "medium"


class BugIssue(BaseModel):
    type: str
    severity: str
    line: int
    column: Optional[int] = None
    message: str
    suggestion: str
    snarkLevel: int


class CodeAnalysisResponse(BaseModel):
    issues: List[BugIssue]
    overallScore: int
    snarkComment: str
    suggestions: List[str]


class SimilarBugsRequest(BaseModel):
    code: str
    language: str


class BugSubmissionRequest(BaseModel):
    code: str
    language: str
    bugType: str
    fix: str
    description: str


cerebras_analyzer: Optional[CerebrasAnalyzer] = None
bug_database: Optional[BugDatabase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cerebras_analyzer, bug_database

    print("Starting Grepal server...")

    try:
        cerebras_analyzer = CerebrasAnalyzer()
        print(f"Cerebras connected: {cerebras_analyzer.is_connected()}")

        bug_database = BugDatabase()
        await bug_database.load_database()
        print(f"Bug DB: {len(bug_database.bugs)} bugs, embeddings: {bug_database.is_ready()}")

        print("Grepal ready!")
    except Exception as e:
        print(f"ERROR: Init failed: {e}")
        raise

    yield

    print("Shutting down")


app = FastAPI(
    title="Grepal API",
    description="Snarky debug buddy",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "cerebras_connected": cerebras_analyzer.is_connected() if cerebras_analyzer else False,
        "embeddings_loaded": bug_database.is_ready() if bug_database else False,
        "bug_db_size": len(bug_database.bugs) if bug_database else 0
    }


@app.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    if not cerebras_analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    print(f"Analyzing {request.language} ({len(request.code)} chars)")

    # Find similar bugs
    similar_bugs = []
    if bug_database:
        similar_bugs = await bug_database.find_similar_bugs(
            request.code,
            request.language
        )
        print(f"Found {len(similar_bugs)} similar bugs")

    # Analyze with Cerebras
    analysis = await cerebras_analyzer.analyze_code(
        code=request.code,
        language=request.language,
        filename=request.filename,
        snark_level=request.snarkLevel,
        similar_bugs=similar_bugs
    )

    # Format response
    snark_map = {"mild": 1, "medium": 2, "savage": 3}
    snark_num = snark_map.get(request.snarkLevel, 2)

    formatted_issues = [
        BugIssue(
            type=issue.get('type', 'unknown'),
            severity=issue.get('severity', 'medium'),
            line=max(1, issue.get('line', 1) or 1),
            column=issue.get('column'),
            message=issue.get('message', 'Issue detected'),
            suggestion=issue.get('suggestion', 'Review this code'),
            snarkLevel=snark_num
        )
        for issue in analysis.issues
    ]

    return CodeAnalysisResponse(
        issues=formatted_issues,
        overallScore=analysis.overall_score,
        snarkComment=analysis.snark_comment,
        suggestions=analysis.suggestions
    )


@app.post("/similar-bugs")
async def find_similar_bugs(request: SimilarBugsRequest):
    if not bug_database:
        return {"similar_bugs": []}

    try:
        similar = await bug_database.find_similar_bugs(
            request.code,
            request.language,
            limit=5
        )
        return {"similar_bugs": similar}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"similar_bugs": []}


@app.post("/submit-bug")
async def submit_bug_fix(request: BugSubmissionRequest):
    if not bug_database:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        bug_id = await bug_database.add_bug(
            code=request.code,
            language=request.language,
            bug_type=request.bugType,
            fix=request.fix,
            description=request.description
        )

        print(f"Added bug: {bug_id}")
        return {"success": True, "bug_id": bug_id}
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    if not bug_database:
        return {"bugs_stored": 0}

    return {
        "bugs_stored": len(bug_database.bugs),
        "languages": bug_database.get_supported_languages()
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting server at http://{host}:{port}")
    print("Set CEREBRAS_API_KEY in .env!")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

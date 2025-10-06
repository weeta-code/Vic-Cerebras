#!/usr/bin/env python3

import os
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import time
import hashlib
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Using system environment variables only")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from cerebras_client import CerebrasAnalyzer
from embeddings import EmbeddingsManager
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
embeddings_manager: Optional[EmbeddingsManager] = None
bug_database: Optional[BugDatabase] = None

request_cache = {}
rate_limiter = defaultdict(list)
pending_requests = {}
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 30
CACHE_DURATION = 5


def get_request_hash(code: str, language: str) -> str:
    return hashlib.md5(f"{code}:{language}".encode()).hexdigest()


def is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    rate_limiter[client_ip] = [
        req_time for req_time in rate_limiter[client_ip] 
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    return len(rate_limiter[client_ip]) >= MAX_REQUESTS_PER_WINDOW


def record_request(client_ip: str):
    rate_limiter[client_ip].append(time.time())


def extract_bug_snippet(code: str, line_number: int, context_lines: int = 3) -> str:
    """
    Extract a code snippet around the bug line for more focused embedding
    
    Args:
        code: Full code content
        line_number: Line where bug was detected (1-indexed)
        context_lines: Number of lines before/after to include
    
    Returns:
        Code snippet focused on the bug area
    """
    lines = code.split('\n')
    total_lines = len(lines)
    
    # Convert to 0-indexed
    bug_line_idx = max(0, line_number - 1)
    
    # Calculate snippet bounds
    start_idx = max(0, bug_line_idx - context_lines)
    end_idx = min(total_lines, bug_line_idx + context_lines + 1)
    
    # Extract snippet
    snippet_lines = lines[start_idx:end_idx]
    snippet = '\n'.join(snippet_lines)
    
    print(f"SNIPPET EXTRACTION:")
    print(f"   Bug line: {line_number} (in {total_lines} total lines)")
    print(f"   Extracted lines {start_idx + 1}-{end_idx} ({len(snippet_lines)} lines)")
    print(f"   Snippet: {repr(snippet[:100])}...")
    
    return snippet


async def _auto_learn_from_issues(
    issues: List[Dict[str, Any]], 
    full_code: str, 
    language: str,
    bug_database,
    embeddings_manager
) -> None:
    """
    Automatically learn from discovered bugs by extracting snippets and adding to database
    """
    try:
        for issue in issues:
            bug_line = issue.get('line', 1)
            bug_type = issue.get('type', 'unknown')
            bug_message = issue.get('message', 'Code issue detected')
            bug_fix = issue.get('suggestion', 'Consider reviewing this code')
            
            # Extract focused snippet around the bug
            bug_snippet = extract_bug_snippet(full_code, bug_line, context_lines=2)
            
            # Skip if snippet is too small (likely not meaningful)
            if len(bug_snippet.strip()) < 10:
                print(f"AUTO-LEARNING: Skipping tiny snippet for {bug_type}")
                continue
            
            # Create a more specific bug type based on content
            enhanced_bug_type = f"{bug_type}_{language}"
            
            # Check for potential duplicates in recent bugs (basic deduplication)
            is_duplicate = await _check_for_duplicate_bug(
                bug_snippet, language, bug_type, bug_database
            )
            
            if is_duplicate:
                print(f"AUTO-LEARNING: Skipping duplicate bug: {bug_type}")
                continue
            
            print(f"AUTO-LEARNING: Adding {enhanced_bug_type} to database...")
            
            # Add to bug database (this triggers embedding generation)
            try:
                bug_id = await bug_database.add_bug(
                    code=bug_snippet,  # ← SNIPPET, not full code
                    language=language,
                    bug_type=enhanced_bug_type,
                    fix=bug_fix,
                    description=f"Auto-learned: {bug_message}",
                    severity=issue.get('severity', 'medium')
                )
                
                # Add embedding to vector cache (mark as snippet for better processing)
                await embeddings_manager.add_bug_embedding(
                    bug_id, bug_snippet, language, is_snippet=True
                )
                
                print(f"AUTO-LEARNING: Successfully added bug {bug_id} to database")
                
            except Exception as e:
                print(f"AUTO-LEARNING: Failed to add bug {enhanced_bug_type}: {e}")
                continue
                
    except Exception as e:
        print(f"AUTO-LEARNING: Failed to process issues: {e}")


async def _check_for_duplicate_bug(
    snippet: str, 
    language: str, 
    bug_type: str, 
    bug_database,
    similarity_threshold: float = 0.85
) -> bool:
    """
    Check if this bug snippet is too similar to existing bugs (basic duplicate detection)
    
    Returns True if this appears to be a duplicate
    """
    try:
        # Get recent bugs of the same type and language
        recent_bugs = [
            bug for bug in bug_database.bugs[-20:]  # Check last 20 bugs only
            if (bug.get('language') == language and 
                bug.get('bug_type', '').startswith(bug_type))
        ]
        
        if not recent_bugs:
            return False
        
        # Simple text-based similarity check (not using embeddings to avoid recursion)
        snippet_words = set(snippet.lower().split())
        
        for existing_bug in recent_bugs:
            existing_code = existing_bug.get('code', '')
            existing_words = set(existing_code.lower().split())
            
            # Calculate Jaccard similarity
            if len(snippet_words) == 0 or len(existing_words) == 0:
                continue
                
            intersection = len(snippet_words & existing_words)
            union = len(snippet_words | existing_words)
            jaccard_similarity = intersection / union if union > 0 else 0
            
            if jaccard_similarity > similarity_threshold:
                print(f"AUTO-LEARNING: Found duplicate (Jaccard: {jaccard_similarity:.3f})")
                return True
        
        return False
        
    except Exception as e:
        print(f"AUTO-LEARNING: Duplicate check failed: {e}")
        return False  # If check fails, allow the bug to be added


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global cerebras_analyzer, embeddings_manager, bug_database
    
    print("Starting Grepal server...")
    
    # Initialize components
    try:
        print("Initializing Cerebras analyzer...")
        cerebras_analyzer = CerebrasAnalyzer()
        print(f"   Cerebras connected: {cerebras_analyzer.is_connected()}")
        
        print("Initializing embeddings manager...")
        embeddings_manager = EmbeddingsManager()
        print(f"   Embeddings ready: {embeddings_manager.is_ready()}")
        if embeddings_manager.is_ready():
            cache_stats = embeddings_manager.get_cache_stats()
            print(f"   Cache stats: {cache_stats}")
        
        print("Initializing bug database...")
        bug_database = BugDatabase()
        
        print("Loading bug database...")
        await bug_database.load_database()
        print(f"   Loaded {len(bug_database.bugs)} bugs from database")
        
        if embeddings_manager.is_ready() and bug_database.bugs:
            print("Preloading embeddings for existing bugs...")
            for i, bug in enumerate(bug_database.bugs[:5]):
                bug_id = bug.get('id', f'bug_{i}')
                await embeddings_manager.add_bug_embedding(
                    bug_id, bug.get('code', ''), bug.get('language', 'unknown')
                )
            print(f"Preloaded embeddings for {min(5, len(bug_database.bugs))} bugs")
        
        print("Grepal server ready!")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize Grepal: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Cleanup
    print("Shutting down Grepal server")


# Create FastAPI app
app = FastAPI(
    title="Grepal API",
    description="The Snarky Debug Buddy - Real-time code analysis with attitude",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Grepal is ready to roast!"}


@app.get("/status")
async def get_status():
    """Get detailed server status"""
    return {
        "status": "running",
        "cerebras_connected": cerebras_analyzer.is_connected() if cerebras_analyzer else False,
        "embeddings_loaded": embeddings_manager.is_ready() if embeddings_manager else False,
        "bug_db_size": len(bug_database.bugs) if bug_database else 0,
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    client_ip = "127.0.0.1"
    """
    Analyze code and return snarky feedback with fixes
    """
    if not cerebras_analyzer:
        raise HTTPException(status_code=503, detail="Cerebras analyzer not initialized")
    
    # Generate request hash for deduplication
    request_hash = get_request_hash(request.code, request.language)
    now = time.time()
    
    # Check cache first
    if request_hash in request_cache:
        cached_result, cache_time = request_cache[request_hash]
        if now - cache_time < CACHE_DURATION:
            print(f"Returning cached result for {request.language} code")
            return cached_result
    
    # Check if request is already pending
    if request_hash in pending_requests:
        print(f"Request already pending for {request.language} code")
        # Wait for existing request to complete
        try:
            return await pending_requests[request_hash]
        except Exception:
            # If pending request failed, continue with new request
            pass
    
    # Rate limiting
    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Please wait before analyzing more code."
        )
    
    # Record this request
    record_request(client_ip)
    
    # Create async task for this request
    async def process_request():
        try:
            print(f"Analyzing {request.language} code ({len(request.code)} chars)")
            
            # Get similar bugs from our database
            similar_bugs = []
            if embeddings_manager and bug_database:
                print(f"DEBUGGING: Starting similarity search for {request.language} code...")
                print(f"DEBUGGING: Embeddings ready: {embeddings_manager.is_ready()}")
                print(f"DEBUGGING: Bug database has {len(bug_database.bugs)} total bugs")
                
                # Count bugs by language
                language_counts = {}
                for bug in bug_database.bugs:
                    lang = bug.get('language', 'unknown')
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                print(f"DEBUGGING: Language distribution: {language_counts}")
                
                similar_bugs = await embeddings_manager.find_similar_bugs(
                    request.code, 
                    request.language,
                    bug_database.bugs
                )
                
                print(f"DEBUGGING: Similarity search returned {len(similar_bugs)} bugs")
                if similar_bugs:
                    print(f"DEBUGGING: Found similar bugs:")
                    for i, bug in enumerate(similar_bugs):
                        print(f"   Bug {i+1}: {bug.get('bug_type', 'unknown')} ({bug.get('language', 'unknown')})")
                        print(f"          Code preview: {bug.get('code', '')[:50]}...")
                else:
                    print(f"DEBUGGING: No similar bugs found - will rely on LLM analysis only")
            else:
                print("DEBUGGING: No embeddings manager or bug database available")
                if not embeddings_manager:
                    print("   Embeddings manager is None")
                elif not embeddings_manager.is_ready():
                    print("   Embeddings manager not ready")
                if not bug_database:
                    print("   Bug database is None")
                elif not bug_database.bugs:
                    print(f"   Bug database is empty")
            
            # Analyze with Cerebras
            analysis_result = await cerebras_analyzer.analyze_code(
                code=request.code,
                language=request.language,
                filename=request.filename,
                snark_level=request.snarkLevel,
                similar_bugs=similar_bugs
            )
            
            # Convert AnalysisResult to CodeAnalysisResponse format
            snark_level_map = {"mild": 1, "medium": 2, "savage": 3}
            snark_level_num = snark_level_map.get(request.snarkLevel, 2)
            
            formatted_issues = []
            for issue in analysis_result.issues:
                formatted_issue = BugIssue(
                    type=issue.get('type', 'unknown'),
                    severity=issue.get('severity', 'medium'),
                    line=max(1, issue.get('line', 1) or 1),
                    column=issue.get('column'),
                    message=issue.get('message', 'Code issue detected'),
                    suggestion=issue.get('suggestion', 'Consider reviewing this code'),
                    snarkLevel=snark_level_num
                )
                formatted_issues.append(formatted_issue)
            
            # AUTOMATIC LEARNING: Add discovered bugs to database
            if analysis_result.issues and bug_database and embeddings_manager:
                print(f"AUTO-LEARNING: Processing {len(analysis_result.issues)} bugs for automatic learning...")
                await _auto_learn_from_issues(
                    analysis_result.issues,
                    request.code,
                    request.language,
                    bug_database,
                    embeddings_manager
                )
            
            response = CodeAnalysisResponse(
                issues=formatted_issues,
                overallScore=analysis_result.overall_score,
                snarkComment=analysis_result.snark_comment,
                suggestions=analysis_result.suggestions
            )
            
            # Cache the result
            request_cache[request_hash] = (response, time.time())
            
            return response
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Remove from pending requests
            if request_hash in pending_requests:
                del pending_requests[request_hash]
    
    # Store the task in pending requests to prevent duplicates
    task = asyncio.create_task(process_request())
    pending_requests[request_hash] = task
    
    try:
        return await task
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar-bugs")
async def find_similar_bugs(request: SimilarBugsRequest):
    """Find similar bugs in our database"""
    if not embeddings_manager or not bug_database:
        return {"similar_bugs": []}
    
    try:
        similar_bugs = await embeddings_manager.find_similar_bugs(
            request.code,
            request.language, 
            bug_database.bugs,
            limit=5
        )
        
        return {"similar_bugs": similar_bugs}
        
    except Exception as e:
        print(f"ERROR: Similar bugs search failed: {e}")
        return {"similar_bugs": []}


@app.post("/submit-bug")
async def submit_bug_fix(request: BugSubmissionRequest):
    """Submit a new bug and fix to the database"""
    if not bug_database or not embeddings_manager:
        raise HTTPException(status_code=503, detail="Bug database not available")
    
    try:
        # Add to bug database
        bug_id = await bug_database.add_bug(
            code=request.code,
            language=request.language,
            bug_type=request.bugType,
            fix=request.fix,
            description=request.description
        )
        
        # Update embeddings
        print(f"Adding embedding for new bug {bug_id}...")
        await embeddings_manager.add_bug_embedding(bug_id, request.code, request.language)
        
        print(f"Added new bug to database: {bug_id}")
        return {"success": True, "bug_id": bug_id}
        
    except Exception as e:
        print(f"ERROR: Bug submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get Grepal usage statistics"""
    if not bug_database:
        return {"bugs_analyzed": 0, "bugs_stored": 0}
    
    return {
        "bugs_analyzed": bug_database.get_analysis_count(),
        "bugs_stored": len(bug_database.bugs),
        "languages_supported": bug_database.get_supported_languages(),
        "snark_level_distribution": bug_database.get_snark_stats()
    }


if __name__ == "__main__":
    # Load environment variables
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("Starting Grepal server...")
    print(f"Server will run at http://{host}:{port}")
    print("Make sure to set your CEREBRAS_API_KEY environment variable!")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
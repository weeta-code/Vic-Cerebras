# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Grepal** is a real-time debugging companion VSCode extension that combines Cerebras LLMs with snarky personality to provide immediate code analysis feedback. It's architected as a two-part system: a TypeScript VSCode extension frontend and a Python FastAPI backend server.

### Key Architecture Components

**Extension Layer (TypeScript):**
- `src/extension.ts` - Main extension entry point, manages VSCode integration, debounced text change detection (1.5s), popup queue system, analysis history tracking, and analyzed lines buffer to prevent redundant analysis
- `src/grepalClient.ts` - HTTP client for server communication with health checks and error handling

**Backend Layer (Python):**
- `server/main.py` - FastAPI server with CORS, rate limiting (30 req/60s), request deduplication, response caching (5s), and automatic bug learning from discovered issues
- `server/cerebras_client.py` - Cerebras LLM integration with exponential backoff retry logic for rate limiting, JSON response parsing/cleaning, and fallback analysis for offline mode
- `server/embeddings.py` - HuggingFace sentence-transformers for semantic bug similarity search
- `server/bug_db.py` - Bug database management with JSON persistence and automatic embedding generation

### Critical Design Patterns

**Analyzed Lines Buffer System:**
The extension maintains a buffer (`analyzedLinesBuffer` in extension.ts:14) to prevent re-analyzing code sections. When issues are found, it marks regions around bug lines (±10 lines) as analyzed. This buffer expires after 1 hour and can be manually cleared via the webview UI.

**Automatic Bug Learning:**
When Cerebras analyzes code and finds issues, `_auto_learn_from_issues()` in main.py:130 automatically extracts focused code snippets (bug line ±2 context lines) and adds them to the bug database with embeddings. It includes Jaccard similarity-based deduplication (0.85 threshold) to avoid storing duplicate bugs.

**Request Deduplication:**
The server uses MD5 hashing of `code:language` to deduplicate concurrent requests. Pending requests wait for the first analysis to complete rather than re-processing (main.py:351-358).

**Popup Queue Management:**
Issues are displayed via a comprehensive summary popup first, then optionally as individual sequential popups. Only one popup is shown at a time to prevent UI spam (extension.ts:133-176).

## Development Commands

### Initial Setup

```bash
# Install Node.js dependencies
npm install

# Set up Python virtual environment and dependencies
cd server && ./setup.sh
```

### Configuration

Create `server/.env` file with:
```
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### Build and Run

```bash
# Compile TypeScript (one-time)
npm run compile

# Watch mode for TypeScript development
npm run watch

# Start Python server (in separate terminal)
cd server && source grepal_env/bin/activate && python main.py

# Test extension in VSCode
# Press F5 to launch Extension Development Host
```

### Development Workflow

**Typical development session requires two terminals:**

Terminal 1:
```bash
cd server
source grepal_env/bin/activate
python main.py
```

Terminal 2:
```bash
npm run watch
```

Then press F5 in VSCode to launch the Extension Development Host.

### Testing and Quality

```bash
# Lint TypeScript code
npm run lint

# Run Python tests
cd server && pytest

# Check Python code quality
cd server && black . && flake8
```

## API Endpoints

- `GET /health` - Health check, returns 200 if server is ready
- `GET /status` - Detailed server status including Cerebras connection, embeddings status, and bug database size
- `POST /analyze` - Main analysis endpoint, expects `{code, language, filename, snarkLevel}`
- `POST /similar-bugs` - Find similar bugs via semantic search
- `POST /submit-bug` - Manually submit bug/fix to database
- `GET /stats` - Usage statistics and bug database metrics

## Language Support

Actively supported: JavaScript, TypeScript, Python, Java, C++, C, Go, Rust

The extension activates automatically when opening files in these languages (see package.json:22-30).

## Configuration

VSCode settings (adjustable via Settings UI or settings.json):
- `grepal.enabled` (boolean, default: true) - Enable/disable real-time analysis
- `grepal.snarkLevel` (enum: "mild", "medium", "savage", default: "medium") - Snark intensity
- `grepal.serverUrl` (string, default: "http://localhost:8000") - Backend server URL

## Important Implementation Details

**Debounce Timing:** Code analysis triggers 1500ms after last keystroke (extension.ts:112). This balances responsiveness with API rate limits.

**Line Number Handling:** Line numbers from VSCode are 1-indexed. When extracting snippets or marking analyzed regions, ensure proper conversion (bug_db.py, main.py:96-127).

**JSON Parsing Robustness:** Cerebras responses may contain malformed JSON with escaped quotes or trailing commas. The parser in cerebras_client.py:189-285 includes multiple cleanup strategies before parsing.

**Rate Limit Handling:** Cerebras API has rate limits. The client retries with exponential backoff (2s, 4s, 8s) for 429 responses (cerebras_client.py:154-187).

**Fallback Analysis:** When Cerebras is unavailable or rate-limited, the system uses regex-based static analysis to detect common bugs (cerebras_client.py:287-439). This ensures the extension remains functional offline.

## Bug Database Schema

Bugs stored in `server/bugs.json` with structure:
```json
{
  "id": "unique_bug_id",
  "code": "code_snippet",
  "language": "python",
  "bug_type": "logic_python",
  "fix": "how_to_fix",
  "description": "bug_description",
  "severity": "high|medium|low|critical",
  "timestamp": "ISO8601"
}
```

Embeddings cached separately in `server/bug_embeddings.pkl` using sentence-transformers model.

## Troubleshooting

**"Server not responding" errors:**
- Verify Python server is running: `curl http://localhost:8000/health`
- Check `server/.env` contains valid CEREBRAS_API_KEY
- Review server logs for initialization errors

**Extension not activating:**
- Ensure file language ID matches supported languages
- Check Output panel → Grepal for activation logs

**Analysis not triggering:**
- Verify `grepal.enabled` setting is true
- Code must change significantly (different hash) to trigger re-analysis
- Check analyzed lines buffer isn't preventing re-analysis (clear via webview UI)

**Rate limit errors:**
- Cerebras free tier has rate limits; client retries automatically
- Consider increasing debounce delay in extension.ts:112
- Server-side rate limiter: 30 requests per 60 seconds per client

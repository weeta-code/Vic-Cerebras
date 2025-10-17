"""
Cerebras LLM integration for code analysis
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    print("WARNING: Cerebras SDK not installed")
    Cerebras = None


@dataclass
class AnalysisResult:
    issues: List[Dict[str, Any]]
    overall_score: int
    snark_comment: str
    suggestions: List[str]


class CerebrasAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        self.client = None
        self.model = "llama3.1-8b"

        if not self.api_key:
            print("WARNING: CEREBRAS_API_KEY not found")
            return

        if Cerebras:
            try:
                self.client = Cerebras(api_key=self.api_key)
                print("Cerebras client initialized")
            except Exception as e:
                print(f"ERROR: Failed to initialize Cerebras: {e}")

    def is_connected(self) -> bool:
        return self.client is not None

    async def analyze_code(
        self,
        code: str,
        language: str,
        filename: str = "unknown",
        snark_level: str = "medium",
        similar_bugs: List[Dict] = None
    ) -> AnalysisResult:
        if not self.is_connected():
            return AnalysisResult(
                issues=[],
                overall_score=50,
                snark_comment="Can't analyze without Cerebras API key.",
                suggestions=["Set CEREBRAS_API_KEY in server/.env"]
            )

        similar_context = ""
        if similar_bugs:
            similar_context = "\n\nSimilar bugs found:\n" + "\n".join([
                f"- {bug.get('description', 'Unknown')} (Fix: {bug.get('fix', 'N/A')})"
                for bug in similar_bugs[:3]
            ])

        snark_prompts = {
            "mild": "You're a code reviewer who points out actual bugs with light sarcasm.",
            "medium": "You're a sarcastic code reviewer who roasts actual bugs - syntax errors, logic errors, runtime errors.",
            "savage": "You're a merciless code reviewer who destroys code with actual bugs. Be brutal about genuine mistakes."
        }

        prompt = f"""{snark_prompts.get(snark_level, snark_prompts['medium'])}

Analyze this {language} code and find ONLY genuine bugs:
- Syntax errors
- Logic errors
- Runtime errors (null pointer, out of bounds, etc)
- Security vulnerabilities

```{language}
{code}
```
{similar_context}

Respond in JSON:
{{
    "issues": [
        {{
            "type": "syntax|logic|runtime|security",
            "severity": "low|medium|high|critical",
            "line": line_number,
            "message": "snarky bug description",
            "suggestion": "how to fix"
        }}
    ],
    "overall_score": 0-100,
    "snark_comment": "overall snarky comment",
    "suggestions": ["fix suggestions"]
}}

Be {snark_level} about real bugs only.
"""

        try:
            response = await asyncio.to_thread(self._get_completion, prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
            return AnalysisResult(
                issues=[],
                overall_score=50,
                snark_comment="Analysis failed. Server issues.",
                suggestions=[]
            )

    def _get_completion(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.7
                )
                return completion.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        print(f"Rate limited, waiting {delay}s")
                        time.sleep(delay)
                        continue
                raise Exception(f"Cerebras API failed: {e}")
        raise Exception("Cerebras API failed after retries")

    def _parse_response(self, response: str) -> AnalysisResult:
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                return AnalysisResult(
                    issues=data.get('issues', []),
                    overall_score=int(data.get('overall_score', 50)),
                    snark_comment=data.get('snark_comment', "Your code exists."),
                    suggestions=data.get('suggestions', [])
                )
        except Exception as e:
            print(f"Parse error: {e}")

        return AnalysisResult(
            issues=[],
            overall_score=50,
            snark_comment="Failed to parse analysis results.",
            suggestions=[]
        )

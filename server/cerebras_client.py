"""
Cerebras LLM integration for real-time code analysis
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    print("WARNING: Cerebras SDK not installed. Install with: pip install cerebras-cloud-sdk")
    Cerebras = None


@dataclass
class AnalysisResult:
    issues: List[Dict[str, Any]]
    overall_score: int
    snark_comment: str
    suggestions: List[str]


class CerebrasAnalyzer:
    """Handles code analysis using Cerebras LLMs"""
    
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        self.client = None
        self.model = "llama3.1-8b"  # Fast model for real-time analysis
        
        if not self.api_key:
            print("WARNING: CEREBRAS_API_KEY not found in environment variables")
            return
            
        if Cerebras:
            try:
                self.client = Cerebras(api_key=self.api_key)
                print("Cerebras client initialized")
            except Exception as e:
                print(f"ERROR: Failed to initialize Cerebras client: {e}")
        else:
            print("ERROR: Cerebras SDK not available")
    
    def is_connected(self) -> bool:
        """Check if Cerebras client is ready"""
        return self.client is not None and self.api_key is not None
    
    async def analyze_code(
        self, 
        code: str, 
        language: str, 
        filename: str = "unknown",
        snark_level: str = "medium",
        similar_bugs: List[Dict] = None
    ) -> AnalysisResult:
        """
        Analyze code using Cerebras LLM with snarky personality
        """
        if not self.is_connected():
            # Fallback analysis for demo purposes
            return self._fallback_analysis(code, language, snark_level)
        
        # Build context with similar bugs
        similar_bugs_context = ""
        if similar_bugs:
            similar_bugs_context = "\\n\\nSimilar bugs found in database:\\n" + "\\n".join([
                f"- {bug.get('description', 'Unknown bug')} (Fix: {bug.get('fix', 'No fix available')})"
                for bug in similar_bugs[:3]
            ])
        
        # Craft the prompt based on snark level
        snark_prompts = {
            "mild": "You are a code reviewer who only points out actual bugs that will cause the code to break or behave incorrectly. Be mildly sarcastic about genuine errors.",
            "medium": "You are a sarcastic code reviewer who ONLY roasts actual bugs - syntax errors, logic errors, or runtime errors that will cause crashes or incorrect behavior. Ignore style preferences. Be brutally sarcastic about genuine mistakes.",
            "savage": "You are a merciless code reviewer who ONLY destroys code with actual bugs - syntax errors, logic errors, runtime errors, or security vulnerabilities. Do NOT comment on style choices, naming conventions, or framework preferences. Only roast code that is genuinely broken or will cause problems."
        }
        
        personality = snark_prompts.get(snark_level, snark_prompts["medium"])
        
        prompt = f"""
{personality}

Analyze this {language} code and identify ONLY genuine bugs that will cause the code to malfunction, crash, or behave incorrectly. DO NOT comment on:
- Style preferences (var vs let, naming conventions)
- Code organization or structure
- Performance optimizations that don't cause actual errors
- Framework or library choices

```{language}
{code}
```

{similar_bugs_context}

ONLY report these types of actual bugs:
- Syntax errors that prevent compilation/execution
- Logic errors that cause incorrect behavior or crashes
- Runtime errors (null pointer, index out of bounds, division by zero)
- Security vulnerabilities that expose actual risks

DO NOT report style issues, naming conventions, or developer preferences.

IMPORTANT OUTPUT REQUIREMENTS:
- There should be no emojis or em-dashes in your output
- Use plain text only
- Keep responses text-based without special characters

Please respond in this JSON format:
{{
    "issues": [
        {{
            "type": "syntax|logic|runtime|security",
            "severity": "low|medium|high|critical", 
            "line": line_number,
            "column": column_number,
            "message": "snarky description of the ACTUAL BUG",
            "suggestion": "how to fix the actual bug"
        }}
    ],
    "overall_score": score_0_to_100,
    "snark_comment": "snarky comment about actual bugs found (or praise if no real bugs)",
    "suggestions": ["suggestions to fix actual bugs only"]
}}

Be {snark_level} about genuine bugs only. If there are no actual bugs, give credit where it's due.
"""
        
        try:
            # Use async completion
            response = await asyncio.to_thread(
                self._get_completion, prompt
            )
            
            # Parse response and create result
            try:
                return self._parse_response(response, snark_level)
            except Exception as parse_error:
                print(f"ERROR: Response parsing failed completely: {parse_error}")
                print("Using fallback analysis with original code...")
                return self._fallback_analysis(code, language, snark_level)
            
        except Exception as e:
            print(f"ERROR: Cerebras analysis failed: {e}")
            return self._fallback_analysis(code, language, snark_level)
    
    def _get_completion(self, prompt: str) -> str:
        """Get completion from Cerebras (blocking call) with retry logic"""
        import time
        
        max_retries = 3
        base_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.7
                )
                return completion.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle rate limiting (429) with exponential backoff
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limited, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                        continue
                
                # For other errors, don't retry
                raise Exception(f"Cerebras API call failed after {attempt + 1} attempts: {e}")
        
        raise Exception(f"Cerebras API failed after {max_retries} attempts")
    
    def _parse_response(self, response: str, snark_level: str) -> AnalysisResult:
        """Parse Cerebras response into AnalysisResult"""
        try:
            import json
            import re
            
            print(f"Raw Cerebras response: {repr(response[:200])}...")
            
            # Clean up the response first
            cleaned_response = response.strip()
            
            # Try to extract JSON from response with multiple strategies
            json_str = None
            
            # Strategy 1: Find complete JSON block
            start = cleaned_response.find('{')
            end = cleaned_response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = cleaned_response[start:end]
            
            # Strategy 2: Try to fix common JSON issues
            if json_str:
                print(f"Original JSON: {repr(json_str[:300])}...")
                
                # Remove any trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                
                # The main issue: fix incorrectly escaped quotes in the response
                # Replace \" with " but be careful about field names vs string content
                
                # First, let's try a more aggressive approach to fix all quote issues
                lines = json_str.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Fix field names that have escaped quotes
                    line = re.sub(r'"(\w+)\\":', r'"\1":', line)  # Fix field names like "issues\":
                    
                    # Fix string values with escaped apostrophes - convert \' to '
                    line = line.replace("\\\'s ", "'s ").replace("\\\'re ", "'re ").replace("\\\'ll ", "'ll ")
                    line = line.replace("\\\'ve ", "'ve ").replace("\\\'t ", "'t ").replace("\\\'d ", "'d ")
                    
                    # Fix other common quote issues in strings
                    # For message and suggestion fields, fix escaped quotes within strings
                    if ('"message"' in line or '"suggestion"' in line or '"snark_comment"' in line) and '\\"' in line:
                        # Replace \" with ' inside string values to avoid JSON parsing issues
                        line = re.sub(r'("(?:message|suggestion|snark_comment)":\s*"[^"]*?)\\\"([^"]*?")', r"\1'\2", line)
                    
                    fixed_lines.append(line)
                
                json_str = '\n'.join(fixed_lines)
                
                print(f"Cleaned JSON: {repr(json_str[:200])}...")
                
                try:
                    data = json.loads(json_str)
                    
                    # Validate required fields
                    issues = data.get('issues', [])
                    if not isinstance(issues, list):
                        issues = []
                    
                    # Clean up any remaining emojis or special characters in the response
                    snark_comment = str(data.get('snark_comment', "Your code exists."))
                    snark_comment = re.sub(r'[^\w\s.,!?;:()-]', '', snark_comment)
                    
                    suggestions = data.get('suggestions', [])
                    if not isinstance(suggestions, list):
                        suggestions = []
                    
                    # Clean suggestions too
                    clean_suggestions = []
                    for suggestion in suggestions:
                        clean_suggestion = re.sub(r'[^\w\s.,!?;:()-]', '', str(suggestion))
                        clean_suggestions.append(clean_suggestion)
                    
                    print(f"Successfully parsed {len(issues)} issues from Cerebras response")
                    
                    return AnalysisResult(
                        issues=issues,
                        overall_score=int(data.get('overall_score', 50)),
                        snark_comment=snark_comment,
                        suggestions=clean_suggestions
                    )
                    
                except json.JSONDecodeError as je:
                    print(f"JSON decode error: {je}")
                    print(f"Problematic JSON: {repr(json_str)}")
                    
        except Exception as e:
            print(f"Failed to parse Cerebras response: {e}")
        
        print("Using fallback analysis due to parsing failure")
        # Fallback if parsing fails - but we don't have access to original code/language here
        # This is a limitation we need to fix by restructuring
        return self._fallback_analysis("", "unknown", snark_level)
    
    def _fallback_analysis(self, code: str, language: str, snark_level: str) -> AnalysisResult:
        """Enhanced fallback analysis when Cerebras is not available"""
        import re
        
        print(f"FALLBACK ANALYSIS: Analyzing {len(code)} chars of {language} code")
        
        snark_comments = {
            "mild": "Your code exists, which is... something, I guess.",
            "medium": "This code makes me question everything I know about programming. Did you even try?",
            "savage": "I've seen garbage disposals produce better code than this. My CPU is literally crying right now."
        }
        
        issues = []
        lines = code.split('\n')
        
        print(f"FALLBACK: Processing {len(lines)} lines of code")
        
        # Multi-language bug detection
        if language in ["javascript", "typescript", "js"]:
            print(f"FALLBACK: Checking JavaScript patterns...")
            
            for line_num, line in enumerate(lines, 1):
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('//') or line_clean.startswith('/*'):
                    continue
                
                print(f"   Line {line_num}: {repr(line_clean[:50])}...")
                
                # Off-by-one error in for loops
                if re.search(r'for\s*\([^;]*;[^;]*<=.*\.length', line_clean):
                    issues.append({
                        "type": "logic",
                        "severity": "high",
                        "line": line_num,
                        "column": 0,
                        "message": "Off-by-one error! Using <= with array.length will cause index out of bounds.",
                        "suggestion": "Change <= to < when iterating through arrays"
                    })
                    print(f"     Found off-by-one error!")
                
                # Array modification during iteration
                if ".push(" in line_clean or ".unshift(" in line_clean or ".splice(" in line_clean:
                    # Check if we're inside a for loop by looking at surrounding context
                    context_start = max(0, line_num - 5)
                    context_end = min(len(lines), line_num + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    if re.search(r'for\s*\([^)]*\)', context):
                        issues.append({
                            "type": "runtime",
                            "severity": "critical",
                            "line": line_num,
                            "column": 0,
                            "message": "Modifying array during iteration will cause infinite loop or skip elements!",
                            "suggestion": "Create a copy of the array before iterating or use a different approach"
                        })
                        print(f"     Found array modification during iteration!")
                
                # Null/undefined access
                if re.search(r'\w+\[\w+\]\.\w+', line_clean) and "?" not in line_clean:
                    issues.append({
                        "type": "runtime",
                        "severity": "high",
                        "line": line_num,
                        "column": 0,
                        "message": "Potential null reference error - accessing property without null check!",
                        "suggestion": "Use optional chaining (?.) or add null checks"
                    })
                    print(f"     Found potential null reference!")
        
        elif language in ["python", "py"]:
            print(f"FALLBACK: Checking Python patterns...")
            
            for line_num, line in enumerate(lines, 1):
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('#'):
                    continue
                
                print(f"   Line {line_num}: {repr(line_clean[:50])}...")
                
                # Mutable default arguments
                if re.search(r'def\s+\w+\([^)]*\[\]', line_clean):
                    issues.append({
                        "type": "logic",
                        "severity": "high",
                        "line": line_num,
                        "column": 0,
                        "message": "Mutable default argument! This is a classic Python gotcha that will cause bugs.",
                        "suggestion": "Use None as default and check inside the function"
                    })
                    print(f"     Found mutable default argument!")
                
                # Division by zero - more sophisticated detection
                if "/" in line_clean and not "//" in line_clean:  # Avoid floor division
                    # Look for patterns like x/0, x / 0, etc.
                    if re.search(r'/\s*0(?![\d\.])', line_clean):
                        issues.append({
                            "type": "runtime",
                            "severity": "critical",
                            "line": line_num,
                            "column": 0,
                            "message": "Division by zero will crash your program!",
                            "suggestion": "Check divisor is not zero before division"
                        })
                        print(f"     Found division by zero!")
                
                # List modification during iteration
                if ((".append(" in line_clean or ".remove(" in line_clean or ".pop(" in line_clean) and 
                    "for " in code and "in " in code):
                    # Check context for iteration
                    context_start = max(0, line_num - 5)
                    context_end = min(len(lines), line_num + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    if re.search(r'for\s+\w+\s+in\s+\w+', context):
                        issues.append({
                            "type": "runtime",
                            "severity": "high",
                            "line": line_num,
                            "column": 0,
                            "message": "Modifying list during iteration causes unexpected behavior!",
                            "suggestion": "Create a copy of the list before iterating: for item in list.copy()"
                        })
                        print(f"     Found list modification during iteration!")
                
                # Unsafe list/dict access
                if re.search(r'\w+\[\d+\]', line_clean) and "len(" not in line_clean:
                    issues.append({
                        "type": "runtime",
                        "severity": "high",
                        "line": line_num,
                        "column": 0,
                        "message": "Potential index out of bounds error - accessing list without bounds check!",
                        "suggestion": "Check index is within list bounds before accessing"
                    })
                    print(f"     Found unsafe list access!")
        
        print(f"FALLBACK ANALYSIS COMPLETE: Found {len(issues)} issues")
        for i, issue in enumerate(issues):
            print(f"   Issue {i+1}: {issue['type']} - {issue['message'][:60]}...")
        
        score = max(10, 100 - len(issues) * 20)  # More aggressive scoring
        
        return AnalysisResult(
            issues=issues,
            overall_score=score,
            snark_comment=snark_comments.get(snark_level, snark_comments["medium"]),
            suggestions=[
                "Consider using a linter",
                "Maybe read the docs?", 
                "Test your code before committing"
            ]
        )

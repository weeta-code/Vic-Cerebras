"""
Bug database management for storing and learning from bug patterns
"""

import os
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter


class BugDatabase:
    """Manages the bug database with learning capabilities"""
    
    def __init__(self, db_file: str = "bugs.json"):
        self.db_file = db_file
        self.bugs: List[Dict[str, Any]] = []
        self.analysis_count = 0
        self.bug_patterns = defaultdict(list)
        self.language_stats = Counter()
        
    async def load_database(self):
        """Load existing bug database from disk"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.bugs = data.get('bugs', [])
                self.analysis_count = data.get('analysis_count', 0)
                
                # Rebuild stats
                self._rebuild_stats()
                
                print(f"Loaded {len(self.bugs)} bugs from database")
                
            except Exception as e:
                print(f"WARNING: Failed to load bug database: {e}")
                await self._initialize_with_samples()
        else:
            print("Creating new bug database")
            await self._initialize_with_samples()
    
    async def save_database(self):
        """Save bug database to disk"""
        try:
            data = {
                'bugs': self.bugs,
                'analysis_count': self.analysis_count,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"ERROR: Failed to save bug database: {e}")
    
    def _rebuild_stats(self):
        """Rebuild statistics from loaded bugs"""
        self.bug_patterns.clear()
        self.language_stats.clear()
        
        for bug in self.bugs:
            bug_type = bug.get('bug_type', 'unknown')
            language = bug.get('language', 'unknown')
            
            self.bug_patterns[bug_type].append(bug)
            self.language_stats[language] += 1
    
    async def _initialize_with_samples(self):
        """Initialize database with common bug patterns"""
        sample_bugs = [
            {
                'id': str(uuid.uuid4()),
                'code': 'for (let i = 0; i <= array.length; i++) { console.log(array[i]); }',
                'language': 'javascript',
                'bug_type': 'off_by_one',
                'fix': 'Change <= to < in the loop condition',
                'description': 'Off-by-one error causing array index out of bounds',
                'severity': 'high',
                'created_at': datetime.now().isoformat(),
                'fix_count': 0
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'def divide(a, b): return a / b',
                'language': 'python',
                'bug_type': 'division_by_zero',
                'fix': 'Add check: if b == 0: raise ValueError("Division by zero")',
                'description': 'Missing division by zero check',
                'severity': 'critical',
                'created_at': datetime.now().isoformat(),
                'fix_count': 0
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'list = [1,2,3]\nfor item in list:\n    list.append(item * 2)',
                'language': 'python',
                'bug_type': 'modifying_during_iteration',
                'fix': 'Create a copy: for item in list.copy(): list.append(item * 2)',
                'description': 'Modifying list during iteration causes unexpected behavior',
                'severity': 'high',
                'created_at': datetime.now().isoformat(),
                'fix_count': 0
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'function getUser(id) {\n    return users[id].name;\n}',
                'language': 'javascript',
                'bug_type': 'null_reference',
                'fix': 'Check if users[id] exists: return users[id] ? users[id].name : null;',
                'description': 'Null reference error when accessing undefined object property',
                'severity': 'critical',
                'created_at': datetime.now().isoformat(),
                'fix_count': 0
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'def process_data(data=[]):\n    data.append("processed")\n    return data',
                'language': 'python',
                'bug_type': 'mutable_default_argument',
                'fix': 'Use None as default: def process_data(data=None): if data is None: data = []',
                'description': 'Mutable default argument will retain state across function calls',
                'severity': 'high',
                'created_at': datetime.now().isoformat(),
                'fix_count': 0
            }
        ]
        
        self.bugs = sample_bugs
        self._rebuild_stats()
        await self.save_database()
        
        print(f"Initialized bug database with {len(sample_bugs)} sample bugs")
    
    async def add_bug(
        self, 
        code: str, 
        language: str, 
        bug_type: str, 
        fix: str,
        description: str,
        severity: str = "medium"
    ) -> str:
        """Add a new bug to the database"""
        
        bug_id = str(uuid.uuid4())
        
        bug_data = {
            'id': bug_id,
            'code': code,
            'language': language.lower(),
            'bug_type': bug_type.lower(),
            'fix': fix,
            'description': description,
            'severity': severity,
            'created_at': datetime.now().isoformat(),
            'fix_count': 0
        }
        
        self.bugs.append(bug_data)
        self.bug_patterns[bug_type.lower()].append(bug_data)
        self.language_stats[language.lower()] += 1
        
        # Save database periodically
        if len(self.bugs) % 5 == 0:
            await self.save_database()
        
        return bug_id
    
    def get_bugs_by_type(self, bug_type: str) -> List[Dict[str, Any]]:
        """Get all bugs of a specific type"""
        return self.bug_patterns.get(bug_type.lower(), [])
    
    def get_bugs_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get all bugs for a specific language"""
        return [bug for bug in self.bugs if bug.get('language') == language.lower()]
    
    def increment_analysis_count(self):
        """Increment the analysis counter"""
        self.analysis_count += 1
    
    def increment_fix_count(self, bug_id: str):
        """Increment fix count for a specific bug"""
        for bug in self.bugs:
            if bug.get('id') == bug_id:
                bug['fix_count'] = bug.get('fix_count', 0) + 1
                break
    
    def get_analysis_count(self) -> int:
        """Get total analysis count"""
        return self.analysis_count
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.language_stats.keys())
    
    def get_bug_types(self) -> List[str]:
        """Get list of bug types in database"""
        return list(self.bug_patterns.keys())
    
    def get_most_common_bugs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most frequently encountered bugs"""
        bug_type_counts = {
            bug_type: len(bugs) 
            for bug_type, bugs in self.bug_patterns.items()
        }
        
        sorted_types = sorted(
            bug_type_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = []
        for bug_type, count in sorted_types[:limit]:
            # Get the most recently added bug of this type
            bugs_of_type = self.bug_patterns[bug_type]
            latest_bug = max(bugs_of_type, key=lambda x: x.get('created_at', ''))
            latest_bug['occurrence_count'] = count
            result.append(latest_bug)
        
        return result
    
    def get_snark_stats(self) -> Dict[str, int]:
        """Get statistics for snark level usage (placeholder)"""
        return {
            'mild': self.analysis_count // 4,
            'medium': self.analysis_count // 2,
            'savage': self.analysis_count // 4
        }
    
    async def search_bugs(self, query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search bugs by description or code content"""
        results = []
        query_lower = query.lower()
        
        for bug in self.bugs:
            # Skip if language filter doesn't match
            if language and bug.get('language') != language.lower():
                continue
            
            # Search in description, bug_type, and fix
            searchable_text = ' '.join([
                bug.get('description', ''),
                bug.get('bug_type', ''),
                bug.get('fix', ''),
                bug.get('code', '')
            ]).lower()
            
            if query_lower in searchable_text:
                results.append(bug)
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        severity_counts = Counter(bug.get('severity', 'unknown') for bug in self.bugs)
        
        return {
            'total_bugs': len(self.bugs),
            'total_analyses': self.analysis_count,
            'languages': dict(self.language_stats),
            'bug_types': {k: len(v) for k, v in self.bug_patterns.items()},
            'severity_distribution': dict(severity_counts),
            'database_size_mb': os.path.getsize(self.db_file) / (1024*1024) if os.path.exists(self.db_file) else 0
        }
    
    async def cleanup_old_bugs(self, days_old: int = 30):
        """Clean up very old, unused bugs to keep database manageable"""
        if len(self.bugs) < 1000:  # Don't cleanup if database is small
            return
            
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        bugs_to_keep = []
        for bug in self.bugs:
            try:
                bug_date = datetime.fromisoformat(bug.get('created_at', '')).timestamp()
                fix_count = bug.get('fix_count', 0)
                
                # Keep bug if it's recent OR has been useful (fixed multiple times)
                if bug_date > cutoff_date or fix_count > 2:
                    bugs_to_keep.append(bug)
                    
            except (ValueError, TypeError):
                # Keep bug if date parsing fails
                bugs_to_keep.append(bug)
        
        removed_count = len(self.bugs) - len(bugs_to_keep)
        if removed_count > 0:
            self.bugs = bugs_to_keep
            self._rebuild_stats()
            await self.save_database()
            print(f"Cleaned up {removed_count} old bugs from database")

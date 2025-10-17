"""
Bug database with embeddings
"""

import os
import json
import pickle
import uuid
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("WARNING: sentence-transformers not installed")
    SentenceTransformer = None
    cosine_similarity = None


class BugDatabase:
    def __init__(self, db_file: str = "bugs.json"):
        self.db_file = db_file
        self.bugs: List[Dict[str, Any]] = []
        self.model = None
        self.embeddings_cache = {}
        self.cache_file = "bug_embeddings.pkl"

        # Load embedding model
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Embeddings model loaded")
                self._load_cache()
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"WARNING: Cache load failed: {e}")

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"WARNING: Cache save failed: {e}")

    async def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                self.bugs = data.get('bugs', [])
                print(f"Loaded {len(self.bugs)} bugs")
            except Exception as e:
                print(f"WARNING: DB load failed: {e}")
                self._initialize_samples()
        else:
            self._initialize_samples()

    async def save_database(self):
        try:
            with open(self.db_file, 'w') as f:
                json.dump({
                    'bugs': self.bugs,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"ERROR: DB save failed: {e}")

    def _initialize_samples(self):
        self.bugs = [
            {
                'id': str(uuid.uuid4()),
                'code': 'for (let i = 0; i <= array.length; i++) {}',
                'language': 'javascript',
                'bug_type': 'off_by_one',
                'fix': 'Change <= to <',
                'description': 'Off-by-one array access',
                'severity': 'high'
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'def divide(a, b): return a / b',
                'language': 'python',
                'bug_type': 'division_by_zero',
                'fix': 'Check if b == 0',
                'description': 'Missing zero check',
                'severity': 'critical'
            },
            {
                'id': str(uuid.uuid4()),
                'code': 'function getUser(id) { return users[id].name; }',
                'language': 'javascript',
                'bug_type': 'null_reference',
                'fix': 'Check users[id] exists first',
                'description': 'Null reference error',
                'severity': 'critical'
            }
        ]
        print(f"Initialized with {len(self.bugs)} sample bugs")

    async def add_bug(
        self,
        code: str,
        language: str,
        bug_type: str,
        fix: str,
        description: str,
        severity: str = "medium"
    ) -> str:
        bug_id = str(uuid.uuid4())
        self.bugs.append({
            'id': bug_id,
            'code': code,
            'language': language.lower(),
            'bug_type': bug_type.lower(),
            'fix': fix,
            'description': description,
            'severity': severity,
            'created_at': datetime.now().isoformat()
        })

        if len(self.bugs) % 10 == 0:
            await self.save_database()

        return bug_id

    def get_supported_languages(self) -> List[str]:
        return list(set(bug.get('language') for bug in self.bugs))

    async def _get_embedding(self, code: str, language: str) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None

        cache_key = f"{language}:{hash(code)}"
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]

        try:
            # Simple preprocessing
            processed = f"[{language}] {code.strip()}"[:500]
            embedding = await asyncio.to_thread(self.model.encode, processed)
            self.embeddings_cache[cache_key] = embedding

            if len(self.embeddings_cache) % 20 == 0:
                self._save_cache()

            return embedding
        except Exception as e:
            print(f"ERROR: Embedding failed: {e}")
            return None

    async def find_similar_bugs(
        self,
        code: str,
        language: str,
        limit: int = 5,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []

        query_embedding = await self._get_embedding(code, language)
        if query_embedding is None:
            return []

        similarities = []
        for bug in self.bugs:
            if bug.get('language') != language:
                continue

            bug_embedding = await self._get_embedding(bug['code'], language)
            if bug_embedding is None:
                continue

            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                bug_embedding.reshape(1, -1)
            )[0, 0]

            if threshold < similarity < 0.99:
                similarities.append({'bug': bug, 'similarity': float(similarity)})

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['bug'] for item in similarities[:limit]]

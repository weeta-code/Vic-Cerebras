"""
HuggingFace embeddings for smart bug lookup and similarity matching
"""

import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import threading

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("WARNING: sentence-transformers or scikit-learn not installed")
    print("Install with: pip install sentence-transformers scikit-learn")
    SentenceTransformer = None
    cosine_similarity = None


class EmbeddingsManager:
    """Manages code embeddings for similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embeddings manager with a lightweight model
        all-MiniLM-L6-v2 is fast and good for code similarity
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.cache_file = "bug_embeddings.pkl"
        self._model_lock = threading.Lock()
        
        if SentenceTransformer:
            try:
                print(f"Loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                print("Embeddings model loaded")
                
                # Load cached embeddings
                self._load_embeddings_cache()
                
            except Exception as e:
                print(f"ERROR: Failed to load embeddings model: {e}")
        else:
            print("ERROR: SentenceTransformers not available")
    
    def is_ready(self) -> bool:
        """Check if embeddings model is loaded"""
        return self.model is not None
    
    def _load_embeddings_cache(self):
        """Load cached embeddings from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"WARNING: Failed to load embeddings cache: {e}")
                self.embeddings_cache = {}
    
    def _save_embeddings_cache(self):
        """Save embeddings cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"WARNING: Failed to save embeddings cache: {e}")
    
    async def get_code_embedding(self, code: str, language: str, is_snippet: bool = False) -> Optional[np.ndarray]:
        """
        Get embedding for a piece of code
        
        Args:
            code: Code to embed
            language: Programming language
            is_snippet: If True, this is a focused bug snippet (affects caching strategy)
        """
        if not self.is_ready():
            print(f"Embeddings model not ready - cannot generate embedding")
            return None
        
        # Create a more specific cache key using processed code hash
        processed_preview = self._preprocess_code(code, language, is_snippet)
        
        # Enhanced cache key with snippet indicator
        cache_key = f"{language}:{hash(processed_preview)}:{len(code)}:{'snippet' if is_snippet else 'full'}"
        
        if cache_key in self.embeddings_cache:
            print(f"Using cached embedding for {language} {'snippet' if is_snippet else 'code'}")
            return self.embeddings_cache[cache_key]
        
        try:
            # Preprocess code for better embeddings
            processed_code = self._preprocess_code(code, language, is_snippet)
            print(f"Processing {language} {'snippet' if is_snippet else 'code'} for embedding:")
            print(f"   Original length: {len(code)} chars")
            print(f"   Processed length: {len(processed_code)} chars")
            print(f"   Processed code preview: {processed_code[:200]}...")
            
            try:
                def encode_with_lock():
                    with self._model_lock:
                        return self.model.encode(processed_code)
                
                print(f"Generating embedding with model {self.model_name}...")
                embedding = await asyncio.to_thread(encode_with_lock)
                print(f"Generated embedding shape: {embedding.shape}")
                
            except Exception as e:
                print(f"Failed to generate embedding: {e}")
                return None
            
            self.embeddings_cache[cache_key] = embedding
            print(f"Cached embedding for {language} code (cache size: {len(self.embeddings_cache)})")
            
            return embedding
            
        except Exception as e:
            print(f"ERROR: Failed to generate embedding: {e}")
            return None
    
    def _preprocess_code(self, code: str, language: str, is_snippet: bool = False) -> str:
        """
        Preprocess code to improve embedding quality
        Focus on structure and logic, less on syntax
        
        Args:
            code: Code to preprocess
            language: Programming language
            is_snippet: If True, preserve more structure for focused analysis
        """
        # Remove comments and clean up
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # For snippets, be less aggressive about comment removal
            if not is_snippet:
                # Skip comments (basic detection)
                if language in ['python'] and line.startswith('#'):
                    continue
                if language in ['javascript', 'typescript', 'java', 'cpp'] and line.startswith('//'):
                    continue
            
            # Add language context
            cleaned_lines.append(line)
        
        # For snippets, preserve line structure; for full code, flatten
        if is_snippet:
            # Preserve line breaks for better bug context
            processed_code = "\n".join(cleaned_lines)
            processed = f"[{language}_bug] {processed_code}"
        else:
            # Flatten for general similarity
            processed = f"[{language}] " + " ".join(cleaned_lines)
        
        # Different limits for snippets vs full code
        max_length = 500 if is_snippet else 1000
        return processed[:max_length]
    
    async def find_similar_bugs(
        self, 
        code: str, 
        language: str, 
        bug_database: List[Dict[str, Any]],
        limit: int = 5,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find similar bugs in the database using embeddings
        """
        print(f"Searching for similar {language} bugs (threshold: {threshold})...")
        
        if not self.is_ready():
            print(f"Cannot search - embeddings model not ready")
            return []
            
        if not bug_database:
            print(f"Cannot search - bug database is empty")
            return []
        
        try:
            print(f"Getting embedding for query code...")
            query_embedding = await self.get_code_embedding(code, language)
            if query_embedding is None:
                print(f"Failed to get query embedding")
                return []
            
            similarities = []
            language_matches = 0
            
            for i, bug in enumerate(bug_database):
                # Skip bugs from different languages (unless we want cross-language similarity)
                if bug.get('language') != language:
                    continue
                
                language_matches += 1
                bug_code = bug.get('code', '')
                bug_type = bug.get('bug_type', 'unknown')
                
                if not bug_code:
                    continue
                
                # Get bug embedding
                bug_embedding = await self.get_code_embedding(bug_code, language)
                if bug_embedding is None:
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    bug_embedding.reshape(1, -1)
                )[0, 0]
                
                print(f"   Bug {i} ({bug_type}): similarity = {similarity:.3f} (threshold: {threshold})")
                
                if similarity > 0.99:
                    print(f"     Skipping near-perfect match (likely same/similar code)")
                    continue
                
                if similarity > threshold:
                    similarities.append({
                        'bug': bug,
                        'similarity': float(similarity)
                    })
                    print(f"     Above threshold - added to results")
                else:
                    print(f"     Below threshold - skipped")
            
            print(f"SIMILARITY SEARCH RESULTS:")
            print(f"   Language matches: {language_matches}/{len(bug_database)}")
            print(f"   Similar bugs found: {len(similarities)}")
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            results = [item['bug'] for item in similarities[:limit]]
            
            for i, result in enumerate(results):
                print(f"   Result {i+1}: {result.get('bug_type', 'unknown')} (similarity: {similarities[i]['similarity']:.3f})")
            
            return results
            
        except Exception as e:
            print(f"ERROR: Similar bugs search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def add_bug_embedding(self, bug_id: str, code: str, language: str, is_snippet: bool = True):
        """
        Add a new bug embedding to the cache
        
        Args:
            bug_id: Unique identifier for the bug
            code: Bug code (usually a snippet)
            language: Programming language
            is_snippet: Whether this is a focused bug snippet (default True)
        """
        print(f"ADDING BUG EMBEDDING:")
        print(f"   Bug ID: {bug_id}")
        print(f"   Language: {language}")
        print(f"   Code length: {len(code)} chars")
        print(f"   Code type: {'snippet' if is_snippet else 'full code'}")
        print(f"   Code preview: {repr(code[:100])}...")
        
        if not self.is_ready():
            print(f"Cannot add bug embedding - model not ready")
            return
        
        try:
            # Use snippet-aware embedding generation
            embedding = await self.get_code_embedding(code, language, is_snippet)
            if embedding is not None:
                print(f"Successfully added embedding for bug {bug_id}")
                print(f"   Embedding shape: {embedding.shape}")
                print(f"   Cache size after addition: {len(self.embeddings_cache)}")
                
                # Save cache more frequently for auto-learned bugs
                if len(self.embeddings_cache) % 5 == 0:
                    self._save_embeddings_cache()
                    print(f"Saved embeddings cache to disk")
            else:
                print(f"Failed to generate embedding for bug {bug_id}")
        except Exception as e:
            print(f"ERROR: Failed to add bug embedding for {bug_id}: {e}")
    
    async def cluster_similar_bugs(self, bug_database: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Group similar bugs into clusters for analysis
        """
        if not self.is_ready() or len(bug_database) < 2:
            return {}
        
        try:
            # Get embeddings for all bugs
            embeddings = []
            valid_bugs = []
            
            for bug in bug_database:
                code = bug.get('code', '')
                language = bug.get('language', 'unknown')
                
                if code:
                    embedding = await self.get_code_embedding(code, language)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_bugs.append(bug)
            
            if len(embeddings) < 2:
                return {}
            
            # Simple clustering using similarity threshold
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            clusters = {}
            used_indices = set()
            cluster_id = 0
            
            for i, bug in enumerate(valid_bugs):
                if i in used_indices:
                    continue
                
                # Find similar bugs
                similar_indices = np.where(similarity_matrix[i] > 0.7)[0]
                cluster_bugs = [valid_bugs[j] for j in similar_indices]
                
                if len(cluster_bugs) > 1:
                    clusters[f"cluster_{cluster_id}"] = cluster_bugs
                    used_indices.update(similar_indices)
                    cluster_id += 1
            
            return clusters
            
        except Exception as e:
            print(f"ERROR: Bug clustering failed: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embeddings cache statistics"""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "model_loaded": self.is_ready(),
            "model_name": self.model_name,
            "cache_file_exists": os.path.exists(self.cache_file)
        }
    
    def cleanup(self):
        """Cleanup resources and save cache"""
        if self.embeddings_cache:
            self._save_embeddings_cache()
            print("Embeddings cache saved")

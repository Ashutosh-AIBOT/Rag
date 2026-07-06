import json
import redis
import numpy as np
from typing import Optional, Dict, Any
from app.config import settings
from app.core.logging import get_logger
from app.embeddings.sentence_transformer import load_embedding_model

logger = get_logger(__name__)


class RedisSemanticCache:
    def __init__(self):
        try:
            self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.embeddings = load_embedding_model()
            logger.info("RedisSemanticCache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RedisSemanticCache: {e}")
            self.client = None

    def get(self, query: str, threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            # 1. Fetch all keys matching `rag_cache:*`
            keys = self.client.keys("rag_cache:*")
            if not keys:
                return None

            # 2. Embed current query
            query_vector = np.array(self.embeddings.embed_query(query))
            query_norm = np.linalg.norm(query_vector)

            best_match = None
            best_score = -1.0

            # 3. Scan keys for semantic match
            for key in keys:
                try:
                    cached_data_str = self.client.get(key)
                    if not cached_data_str:
                        continue
                    cached_data = json.loads(cached_data_str)
                    
                    cached_vector = np.array(cached_data["embedding"])
                    cached_norm = np.linalg.norm(cached_vector)
                    
                    # Calculate Cosine Similarity
                    similarity = np.dot(query_vector, cached_vector) / (query_norm * cached_norm)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = cached_data
                except Exception as ex:
                    logger.warning(f"Failed to evaluate cache key {key}: {ex}")

            logger.info(f"Semantic Cache Lookup. Query: '{query}'. Best Score: {best_score:.4f}")
            if best_score >= threshold and best_match:
                logger.info(f"Semantic Cache HIT for query: '{query}'")
                return {
                    "answer": best_match["answer"],
                    "sources": best_match.get("sources", []),
                    "trace": best_match.get("trace", {})
                }
        except Exception as e:
            logger.error(f"Semantic Cache lookup failed: {e}")
        return None

    def set(self, query: str, answer: str, sources: list[str], trace: dict) -> None:
        if not self.client:
            return
        try:
            query_vector = self.embeddings.embed_query(query)
            key = f"rag_cache:{query.strip()}"
            
            cache_payload = {
                "query": query,
                "embedding": query_vector,
                "answer": answer,
                "sources": sources,
                "trace": trace
            }
            
            self.client.set(key, json.dumps(cache_payload), ex=86400)  # cache TTL: 24 hours
            logger.info(f"Semantic Cache set successfully for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic Cache store failed: {e}")


semantic_cache = RedisSemanticCache()

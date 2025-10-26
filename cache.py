"""
Caching utilities for performance optimization
"""
import json
import sqlite3
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os


class ResponseCache:
    """Simple in-memory cache for LLM responses to improve performance"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
    
    def _generate_key(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Generate cache key from parameters"""
        key_string = f"{prompt}|{temperature}|{top_p}|{max_tokens}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._generate_key(prompt, temperature, top_p, max_tokens)
        
        if key not in self.cache:
            return None
        
        cached_data = self.cache[key]
        created_at = datetime.fromisoformat(cached_data['created_at'])
        
        # Check if expired
        if datetime.now() - created_at > timedelta(minutes=self.ttl_minutes):
            del self.cache[key]
            return None
        
        return cached_data['response']
    
    def set(self, prompt: str, temperature: float, top_p: float, max_tokens: int, response: str):
        """Cache a response"""
        key = self._generate_key(prompt, temperature, top_p, max_tokens)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'created_at': datetime.now().isoformat()
        }
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


class DatabaseOptimizer:
    """Database optimization utilities"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_created_at 
                ON experiments(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name 
                ON experiments(name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_response_count 
                ON experiments(response_count)
            """)
            
            conn.commit()
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create indexes: {e}")
        finally:
            conn.close()
    
    def optimize_database(self):
        """Run database optimization commands"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Analyze tables for better query planning
            cursor.execute("ANALYZE")
            
            # Vacuum to reclaim space and optimize
            cursor.execute("VACUUM")
            
            conn.commit()
            print("✅ Database optimization completed")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not optimize database: {e}")
        finally:
            conn.close()


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
    
    def record_api_call(self, response_time: float, cache_hit: bool = False):
        """Record API call metrics"""
        self.metrics['api_calls'] += 1
        self.metrics['total_response_time'] += response_time
        self.metrics['avg_response_time'] = (
            self.metrics['total_response_time'] / self.metrics['api_calls']
        )
        
        if cache_hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_requests == 0:
            return 0.0
        return (self.metrics['cache_hits'] / total_requests) * 100
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        return {
            **self.metrics,
            'cache_hit_rate': self.get_cache_hit_rate()
        }
    
    def log_metrics(self):
        """Log current metrics"""
        metrics = self.get_metrics()
        print(f"📊 Performance Metrics:")
        print(f"   API Calls: {metrics['api_calls']}")
        print(f"   Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"   Cache Hits: {metrics['cache_hits']}")
        print(f"   Cache Misses: {metrics['cache_misses']}")


# Global instances
response_cache = ResponseCache()
db_optimizer = DatabaseOptimizer()
performance_monitor = PerformanceMonitor()

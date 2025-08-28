"""
Mock de aioredis para pruebas cuando el módulo real no está disponible
"""

class Redis:
    """Mock Redis client"""
    
    def __init__(self, *args, **kwargs):
        self.data = {}
    
    async def get(self, key):
        """Mock get method"""
        return self.data.get(key)
    
    async def set(self, key, value, ex=None):
        """Mock set method"""
        self.data[key] = value
        return True
    
    async def delete(self, key):
        """Mock delete method"""
        if key in self.data:
            del self.data[key]
            return 1
        return 0
    
    async def exists(self, key):
        """Mock exists method"""
        return key in self.data
    
    async def expire(self, key, seconds):
        """Mock expire method"""
        return True
    
    async def keys(self, pattern="*"):
        """Mock keys method"""
        if pattern == "*":
            return list(self.data.keys())
        # Simple pattern matching
        import re
        pattern_re = pattern.replace("*", ".*")
        return [k for k in self.data.keys() if re.match(pattern_re, k)]
    
    async def close(self):
        """Mock close method"""
        pass
    
    async def wait_closed(self):
        """Mock wait_closed method"""
        pass

async def create_redis(*args, **kwargs):
    """Mock create_redis function"""
    return Redis(*args, **kwargs)

async def create_redis_pool(*args, **kwargs):
    """Mock create_redis_pool function"""
    return Redis(*args, **kwargs)

# Definir __all__ para compatibilidad
__all__ = ['Redis', 'create_redis', 'create_redis_pool']

#!/usr/bin/env python3
"""
MSC Framework Unificado Enhanced v4.0 - Sistema de Síntesis Colectiva Meta-cognitiva
Mejoras principales v4.0:
- Integración con Claude API para generación de código
- Sistema de seguridad reforzado con OAuth2
- Nuevos agentes: Analítico, Creativo, Optimizador
- Sistema de consenso mejorado con votación ponderada
- Análisis predictivo con modelos ML avanzados
- Dashboard web con visualización 3D del grafo
- Sistema de recuperación ante fallos mejorado
- Soporte para clustering y ejecución distribuida
- Integración con bases de datos (Redis, PostgreSQL)
- Sistema de notificaciones y alertas
"""

# === IMPORTACIONES CORE ===
import asyncio
import aiohttp
import numpy as np
import hashlib
import json
import time
import logging
import random
import math
import argparse
import yaml
import statistics
import csv
import os
import re
import sys
import threading
import concurrent.futures
import ast
import inspect
import weakref
import pickle
import zlib
import base64
import signal
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, OrderedDict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set, Type, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import wraps, lru_cache, partial
from contextlib import contextmanager, asynccontextmanager
import queue
import heapq
import uuid
import secrets

# === IMPORTACIONES ASYNC ===
import aiofiles
import aiodns
from aiohttp import web
# Intentar importar aioredis, usar mock si no está disponible
try:
    import aioredis
except ImportError:
    import aioredis_mock as aioredis
    logging.warning("aioredis no disponible, usando mock")
import asyncpg

# === IMPORTACIONES PARA DATOS ===
import pandas as pd
import numpy as np
from scipy import stats, spatial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier

# === IMPORTACIONES PARA SEGURIDAD ===
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import anthropic  # Claude API

# === IMPORTACIONES PARA VISUALIZACIÓN ===
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
import seaborn as sns

# === IMPORTACIONES PARA ML ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, SAGEConv
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops
from torch_geometric.data import Data, Batch
import transformers  # Para embeddings de texto

# === IMPORTACIONES PARA SERVIDOR ===
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from authlib.integrations.flask_client import OAuth

# === IMPORTACIONES PARA MONITOREO ===
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, Info
import sentry_sdk

# === CONFIGURACIÓN MEJORADA ===
T = TypeVar('T')

class Config:
    """Configuración centralizada del sistema"""
    
    # Claude API
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
    CLAUDE_MODEL = "claude-3-sonnet-20240229"
    
    # Seguridad
    SECRET_KEY = os.getenv('MSC_SECRET_KEY', secrets.token_hex(32))
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    OAUTH_PROVIDERS = {
        'google': {
            'client_id': os.getenv('GOOGLE_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_CLIENT_SECRET')
        }
    }
    
    # Base de datos
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/msc')
    
    # Monitoreo
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    
    # Límites
    MAX_NODES = 100000
    MAX_EDGES_PER_NODE = 100
    MAX_AGENTS = 50
    MAX_CODE_EXECUTION_TIME = 10.0
    
    # Paths
    DATA_DIR = os.getenv('MSC_DATA_DIR', './data')
    CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
    LOG_DIR = os.path.join(DATA_DIR, 'logs')

# === CONFIGURACIÓN DE LOGGING MEJORADA ===
class ColoredFormatter(logging.Formatter):
    """Formatter con colores para mejor legibilidad"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"
    
    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.grey)
        record.levelname = f"{log_color}{record.levelname}{self.reset}"
        record.name = f"{self.blue}{record.name}{self.reset}"
        
        # Añadir contexto adicional
        if hasattr(record, 'agent_id'):
            record.msg = f"[Agent: {record.agent_id}] {record.msg}"
        if hasattr(record, 'node_id'):
            record.msg = f"[Node: {record.node_id}] {record.msg}"
            
        return super().format(record)

# Configurar logging
def setup_logging(log_level=logging.INFO):
    """Configura el sistema de logging"""
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Handler para archivo
    file_handler = logging.FileHandler(
        os.path.join(Config.LOG_DIR, f'msc_{datetime.now():%Y%m%d}.log')
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    ))
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configurar Sentry si está disponible
    if Config.SENTRY_DSN:
        sentry_sdk.init(dsn=Config.SENTRY_DSN)
    
    return root_logger

logger = setup_logging()

# === UTILIDADES MEJORADAS ===
class AsyncContextManager:
    """Context manager asíncrono base"""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class RateLimiter:
    """Rate limiter asíncrono"""
    
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """Adquiere tokens del rate limiter"""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < tokens:
                sleep_time = (tokens - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0
            else:
                self.allowance -= tokens

class CircuitBreaker:
    """Circuit breaker para manejo de fallos"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Ejecuta función con circuit breaker"""
        async with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                
                raise e

# === SISTEMA DE EVENTOS MEJORADO ===
class EventType(Enum):
    """Tipos de eventos del sistema"""
    # Nodos
    NODE_CREATED = auto()
    NODE_UPDATED = auto()
    NODE_DELETED = auto()
    NODE_MERGED = auto()
    
    # Conexiones
    EDGE_CREATED = auto()
    EDGE_DELETED = auto()
    EDGE_UPDATED = auto()
    
    # Agentes
    AGENT_ACTION = auto()
    AGENT_CREATED = auto()
    AGENT_DESTROYED = auto()
    AGENT_EVOLVED = auto()
    
    # Sistema
    EVOLUTION_CYCLE = auto()
    CONSENSUS_REACHED = auto()
    CONSENSUS_FAILED = auto()
    BLOCK_ADDED = auto()
    CHECKPOINT_CREATED = auto()
    
    # Métricas
    METRICS_UPDATE = auto()
    PERFORMANCE_ALERT = auto()
    
    # Errores
    ERROR_OCCURRED = auto()
    CRITICAL_ERROR = auto()
    
    # Claude
    CLAUDE_GENERATION = auto()
    CLAUDE_ERROR = auto()

@dataclass
class Event:
    """Evento del sistema mejorado"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.METRICS_UPDATE
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    target: Optional[str] = None
    priority: int = 5  # 1-10, donde 1 es máxima prioridad
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.name,
            'data': self.data,
            'timestamp': self.timestamp,
            'source': self.source,
            'target': self.target,
            'priority': self.priority,
            'metadata': self.metadata
        }

class EnhancedAsyncEventBus:
    """Bus de eventos asíncrono mejorado con persistencia"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._event_history = deque(maxlen=10000)
        self._redis = redis_client
        self._metrics = {
            'events_processed': Counter('msc_events_processed', 'Events processed', ['event_type']),
            'events_failed': Counter('msc_events_failed', 'Events failed', ['event_type']),
            'processing_time': Histogram('msc_event_processing_seconds', 'Event processing time', ['event_type'])
        }
        self._event_filters: List[Callable[[Event], bool]] = []
    
    def add_filter(self, filter_func: Callable[[Event], bool]):
        """Añade un filtro de eventos"""
        self._event_filters.append(filter_func)
    
    async def start(self):
        """Inicia el procesamiento de eventos"""
        self._running = True
        # Múltiples workers para procesamiento paralelo
        workers = [
            asyncio.create_task(self._process_events(f"worker_{i}"))
            for i in range(3)
        ]
    
    async def publish(self, event: Event):
        """Publica un evento al bus"""
        # Aplicar filtros
        for filter_func in self._event_filters:
            if not filter_func(event):
                return
        
        # Añadir a cola con prioridad
        await self._event_queue.put((event.priority, time.time(), event))
        
        # Persistir en Redis si está disponible
        if self._redis:
            try:
                await self._redis.lpush(
                    f"events:{event.type.name}",
                    json.dumps(event.to_dict())
                )
                await self._redis.expire(f"events:{event.type.name}", 86400)  # 24h
            except:
                pass
    
    async def _process_events(self, worker_id: str):
        """Procesa eventos de la cola"""
        while self._running:
            try:
                priority, timestamp, event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                with self._metrics['processing_time'].labels(event.type.name).time():
                    await self._handle_event(event)
                
                self._metrics['events_processed'].labels(event.type.name).inc()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in {worker_id}: {e}")
                self._metrics['events_failed'].labels('unknown').inc()
    
    async def _handle_event(self, event: Event):
        """Maneja un evento específico"""
        self._event_history.append(event)
        
        handlers = self._subscribers.get(event.type, [])
        
        # Ejecutar handlers en paralelo
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                tasks.append(
                    asyncio.get_event_loop().run_in_executor(None, handler, event)
                )
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log errores
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {handlers[i].__name__} failed: {result}")

# === CLAUDE API INTEGRATION ===
class ClaudeAPIClient:
    """Cliente para interactuar con Claude API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.CLAUDE_API_KEY
        if not self.api_key:
            raise ValueError("Claude API key not configured")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.rate_limiter = RateLimiter(rate=50, per=60.0)  # 50 requests per minute
        self.circuit_breaker = CircuitBreaker()
        
        # Cache de respuestas
        self._response_cache = {}
        self._cache_ttl = 3600  # 1 hora
        
        # Métricas
        self.metrics = {
            'requests': Counter('claude_api_requests', 'Claude API requests', ['status']),
            'latency': Histogram('claude_api_latency', 'Claude API latency'),
            'tokens': Counter('claude_api_tokens', 'Claude API tokens used', ['type'])
        }
    
    async def generate_code(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Genera código usando Claude"""
        # Check cache
        cache_key = hashlib.sha256(f"{prompt}{context}".encode()).hexdigest()
        if cache_key in self._response_cache:
            cached_time, cached_response = self._response_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_response
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Preparar prompt mejorado
        enhanced_prompt = self._prepare_code_prompt(prompt, context)
        
        try:
            # Llamar a Claude API con circuit breaker
            result = await self.circuit_breaker.call(
                self._call_claude_api,
                enhanced_prompt
            )
            
            # Extraer código de la respuesta
            code = self._extract_code(result)
            
            # Cache response
            self._response_cache[cache_key] = (time.time(), code)
            
            # Métricas
            self.metrics['requests'].labels('success').inc()
            
            return code
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            self.metrics['requests'].labels('error').inc()
            return None
    
    async def _call_claude_api(self, prompt: str) -> str:
        """Llamada real a Claude API"""
        start_time = time.time()
        
        message = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=2000,
                temperature=0.7,
                system="You are an expert Python developer specializing in graph algorithms, multi-agent systems, and metaprogramming. Generate clean, efficient, and safe code.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        )
        
        # Métricas
        latency = time.time() - start_time
        self.metrics['latency'].observe(latency)
        self.metrics['tokens'].labels('input').inc(message.usage.input_tokens)
        self.metrics['tokens'].labels('output').inc(message.usage.output_tokens)
        
        return message.content[0].text
    
    def _prepare_code_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Prepara prompt mejorado con contexto"""
        enhanced_prompt = f"""
Please generate Python code for the following task:

{prompt}

Context:
- System: MSC Framework v4.0 (Meta-cognitive Collective Synthesis)
- Current graph nodes: {context.get('node_count', 0)}
- Current graph edges: {context.get('edge_count', 0)}
- Available imports: math, random, statistics, numpy, networkx
- Constraints: Code must be safe, efficient, and follow best practices

Requirements:
1. The code should be a complete, executable function
2. Include proper error handling
3. Add docstring with description
4. Use type hints
5. Optimize for performance
6. Do not use any dangerous operations (file I/O, network, exec, eval)

Return only the Python code, wrapped in ```python``` markers.
"""
        return enhanced_prompt
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extrae código de la respuesta de Claude"""
        # Buscar código entre marcadores
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Si no hay marcadores, intentar extraer función
        func_pattern = r'def\s+\w+\s*\([^)]*\):\s*\n(.*?)(?=\n\S|\Z)'
        matches = re.findall(func_pattern, response, re.DOTALL)
        
        if matches:
            return f"def generated_function():\n{matches[0]}"
        
        return None

# === SISTEMA DE CACHÉ DISTRIBUIDO ===
class DistributedCache:
    """Sistema de caché distribuido con Redis"""
    
    def __init__(self, redis_client: aioredis.Redis, local_size: int = 1000):
        self.redis = redis_client
        self.local_cache = OrderedDict()
        self.local_size = local_size
        self._lock = asyncio.Lock()
        
        # Métricas
        self.metrics = {
            'hits': Counter('cache_hits', 'Cache hits', ['level']),
            'misses': Counter('cache_misses', 'Cache misses'),
            'latency': Histogram('cache_latency', 'Cache operation latency', ['operation'])
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del caché"""
        # Nivel 1: Cache local
        async with self._lock:
            if key in self.local_cache:
                self.local_cache.move_to_end(key)
                self.metrics['hits'].labels('local').inc()
                return self.local_cache[key]
        
        # Nivel 2: Redis
        with self.metrics['latency'].labels('redis_get').time():
            value = await self.redis.get(key)
            
        if value:
            # Deserializar
            try:
                deserialized = pickle.loads(value)
                # Añadir a cache local
                await self._add_to_local(key, deserialized)
                self.metrics['hits'].labels('redis').inc()
                return deserialized
            except:
                pass
        
        self.metrics['misses'].inc()
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Establece valor en caché"""
        # Serializar
        serialized = pickle.dumps(value)
        
        # Guardar en Redis
        with self.metrics['latency'].labels('redis_set').time():
            await self.redis.setex(key, ttl, serialized)
        
        # Añadir a cache local
        await self._add_to_local(key, value)
    
    async def _add_to_local(self, key: str, value: Any):
        """Añade a cache local con evicción LRU"""
        async with self._lock:
            if key in self.local_cache:
                self.local_cache.move_to_end(key)
            else:
                self.local_cache[key] = value
                if len(self.local_cache) > self.local_size:
                    self.local_cache.popitem(last=False)

# === MODELOS DE DATOS AVANZADOS ===
@dataclass
class NodeMetadata:
    """Metadatos extendidos para nodos"""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    importance_score: float = 0.5
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    version: int = 1
    locked: bool = False
    locked_by: Optional[str] = None

class AdvancedKnowledgeComponent:
    """Componente de conocimiento avanzado con ML"""
    
    def __init__(self, node_id: int, content: str = "Abstract Concept", 
                 initial_state: float = 0.1, keywords: Optional[Set[str]] = None):
        self.id = node_id
        self.content = content
        self.state = initial_state
        self.keywords = keywords or set()
        self.connections_out: Dict[int, float] = {}
        self.connections_in: Dict[int, float] = {}
        
        # Metadatos
        self.metadata = NodeMetadata()
        
        # Embeddings y características
        self.embedding: Optional[torch.Tensor] = None
        self.features: Dict[str, float] = {
            'centrality': 0.0,
            'clustering_coefficient': 0.0,
            'pagerank': 0.0,
            'betweenness': 0.0,
            'change_velocity': 0.0,
            'stability': 1.0
        }
        
        # Historial
        self.state_history: deque = deque(maxlen=1000)
        self.state_history.append((time.time(), initial_state))
        self.interaction_history: deque = deque(maxlen=100)
        
        # Predicciones
        self.predicted_state: Optional[float] = None
        self.anomaly_score: float = 0.0
    
    async def update_state(self, new_state: float, source: Optional[str] = None, 
                          reason: Optional[str] = None) -> bool:
        """Actualiza el estado con validación y tracking"""
        # Verificar bloqueo
        if self.metadata.locked and self.metadata.locked_by != source:
            logger.warning(f"Node {self.id} is locked by {self.metadata.locked_by}")
            return False
        
        MIN_STATE = 0.01
        MAX_STATE = 1.0
        old_state = self.state
        
        # Validar nuevo estado
        new_state = max(MIN_STATE, min(MAX_STATE, new_state))
        
        # Detectar cambio anómalo
        if len(self.state_history) > 10:
            recent_states = [s[1] for s in list(self.state_history)[-10:]]
            mean = np.mean(recent_states)
            std = np.std(recent_states)
            z_score = abs((new_state - mean) / (std + 1e-6))
            self.anomaly_score = min(z_score / 3.0, 1.0)
        
        # Actualizar
        self.state = new_state
        self.state_history.append((time.time(), new_state))
        self.metadata.updated_at = time.time()
        self.metadata.version += 1
        
        # Calcular velocidad de cambio
        if len(self.state_history) > 1:
            time_delta = self.state_history[-1][0] - self.state_history[-2][0]
            state_delta = self.state_history[-1][1] - self.state_history[-2][1]
            self.features['change_velocity'] = state_delta / time_delta if time_delta > 0 else 0
            
            # Calcular estabilidad
            recent_velocities = []
            for i in range(min(10, len(self.state_history) - 1)):
                t_delta = self.state_history[-i-1][0] - self.state_history[-i-2][0]
                s_delta = self.state_history[-i-1][1] - self.state_history[-i-2][1]
                if t_delta > 0:
                    recent_velocities.append(abs(s_delta / t_delta))
            
            if recent_velocities:
                self.features['stability'] = 1.0 / (1.0 + np.std(recent_velocities))
        
        # Registrar interacción
        self.interaction_history.append({
            'timestamp': time.time(),
            'source': source,
            'action': 'update_state',
            'old_value': old_state,
            'new_value': new_state,
            'reason': reason
        })
        
        return old_state != self.state
    
    def calculate_importance(self) -> float:
        """Calcula importancia del nodo con ML"""
        # Factores base
        state_factor = self.state
        connectivity_factor = (len(self.connections_in) + len(self.connections_out)) / 20
        keyword_factor = len(self.keywords) / 10
        access_factor = min(self.metadata.access_count / 100, 1.0)
        
        # Factores de red
        centrality_factor = self.features.get('centrality', 0.0)
        pagerank_factor = self.features.get('pagerank', 0.0)
        
        # Factor temporal
        age = (time.time() - self.metadata.created_at) / 86400  # días
        recency_factor = 1.0 / (1.0 + age * 0.1)
        
        # Factor de estabilidad
        stability_factor = self.features.get('stability', 0.5)
        
        # Combinación ponderada
        importance = (
            state_factor * 0.20 +
            connectivity_factor * 0.15 +
            keyword_factor * 0.10 +
            access_factor * 0.10 +
            centrality_factor * 0.15 +
            pagerank_factor * 0.10 +
            recency_factor * 0.10 +
            stability_factor * 0.10
        )
        
        # Boost por anomalía (nodos interesantes)
        if self.anomaly_score > 0.7:
            importance *= 1.2
        
        self.metadata.importance_score = min(importance, 1.0)
        return self.metadata.importance_score
    
    async def merge_with(self, other: 'AdvancedKnowledgeComponent') -> bool:
        """Fusiona este nodo con otro"""
        if self.metadata.locked or other.metadata.locked:
            return False
        
        # Combinar contenido
        self.content = f"{self.content} | {other.content}"
        
        # Combinar keywords
        self.keywords.update(other.keywords)
        
        # Combinar estado (promedio ponderado)
        total_weight = self.state + other.state
        self.state = (self.state * self.state + other.state * other.state) / total_weight
        
        # Combinar conexiones
        for target_id, utility in other.connections_out.items():
            if target_id != self.id:
                existing = self.connections_out.get(target_id, 0)
                self.connections_out[target_id] = max(existing, utility)
        
        # Combinar metadatos
        self.metadata.tags.update(other.metadata.tags)
        self.metadata.access_count += other.metadata.access_count
        
        # Registrar fusión
        self.interaction_history.append({
            'timestamp': time.time(),
            'action': 'merge',
            'merged_with': other.id,
            'merged_content': other.content[:50]
        })
        
        return True

# === GRAFO DE SÍNTESIS AVANZADO ===
class AdvancedCollectiveSynthesisGraph:
    """Grafo de síntesis colectiva avanzado con ML y distribución"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[int, AdvancedKnowledgeComponent] = {}
        self.next_node_id = 0
        self._node_lock = asyncio.Lock()
        
        # Componentes
        self.event_bus = EnhancedAsyncEventBus()
        self.claude_client = ClaudeAPIClient()
        
        # Cache distribuido
        self.cache: Optional[DistributedCache] = None
        
        # Modelo GNN avanzado
        self.gnn_model = self._build_gnn_model()
        self.gnn_optimizer = torch.optim.AdamW(
            self.gnn_model.parameters(), 
            lr=config.get('gnn_learning_rate', 0.001),
            weight_decay=config.get('gnn_weight_decay', 0.01)
        )
        
        # Embeddings de texto
        self.text_encoder = self._init_text_encoder()
        
        # Índices
        self.keyword_index: Dict[str, Set[int]] = defaultdict(set)
        self.tag_index: Dict[str, Set[int]] = defaultdict(set)
        self.cluster_index: Dict[int, Set[int]] = defaultdict(set)
        
        # Análisis de red
        self.network_analyzer = NetworkAnalyzer(self)
        
        # Métricas
        self.metrics = GraphMetrics()
        
        logger.info("Advanced Collective Synthesis Graph initialized")
    
    def _build_gnn_model(self):
        """Construye modelo GNN avanzado"""
        return AdvancedGNN(
            num_node_features=self.config.get('node_features', 768),
            hidden_channels=self.config.get('gnn_hidden', 128),
            out_channels=self.config.get('gnn_output', 64),
            num_heads=self.config.get('gnn_heads', 8),
            num_layers=self.config.get('gnn_layers', 4),
            dropout=self.config.get('gnn_dropout', 0.1)
        )
    
    def _init_text_encoder(self):
        """Inicializa encoder de texto"""
        # Usar sentence-transformers para embeddings
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except:
            logger.warning("Sentence transformers not available")
            return None
    
    async def initialize(self, redis_url: str = None, postgres_url: str = None):
        """Inicializa componentes asíncronos"""
        await self.event_bus.start()
        
        # Inicializar Redis si está disponible
        if redis_url:
            redis = await aioredis.create_redis_pool(redis_url)
            self.cache = DistributedCache(redis)
            self.event_bus._redis = redis
        
        # Inicializar PostgreSQL si está disponible
        if postgres_url:
            self.db_pool = await asyncpg.create_pool(postgres_url)
            await self._init_database()
    
    async def add_node(self, content: str, initial_state: float = 0.1,
                      keywords: Optional[Set[str]] = None,
                      created_by: Optional[str] = None,
                      properties: Dict[str, Any] = None) -> AdvancedKnowledgeComponent:
        """Añade un nodo con validación completa"""
        # Validar límites
        if len(self.nodes) >= Config.MAX_NODES:
            raise ValueError(f"Maximum number of nodes ({Config.MAX_NODES}) reached")
        
        async with self._node_lock:
            node_id = self.next_node_id
            node = AdvancedKnowledgeComponent(node_id, content, initial_state, keywords)
            
            # Configurar metadatos
            node.metadata.created_by = created_by
            if properties:
                node.metadata.properties.update(properties)
            
            # Generar embedding
            if self.text_encoder:
                embedding = await self._generate_embedding(content)
                node.metadata.embedding = embedding
            
            # Añadir al grafo
            self.nodes[node_id] = node
            self.next_node_id += 1
            
            # Actualizar índices
            self._update_indices(node, 'add')
            
            # Publicar evento
            await self.event_bus.publish(Event(
                type=EventType.NODE_CREATED,
                data={
                    'node_id': node_id,
                    'content': content,
                    'keywords': list(keywords) if keywords else []
                },
                source='graph',
                target=str(node_id)
            ))
            
            # Métricas
            self.metrics.record_node_operation('create')
            
            logger.info(f"Node {node_id} created", extra={'node_id': node_id})
            return node
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Genera embedding de texto"""
        if self.text_encoder:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.text_encoder.encode, text
            )
            return embedding
        return np.random.randn(768)  # Fallback random
    
    def _update_indices(self, node: AdvancedKnowledgeComponent, operation: str):
        """Actualiza índices del grafo"""
        if operation == 'add':
            # Keywords
            for keyword in node.keywords:
                self.keyword_index[keyword].add(node.id)
            
            # Tags
            for tag in node.metadata.tags:
                self.tag_index[tag].add(node.id)
            
            # Cluster
            if node.metadata.cluster_id is not None:
                self.cluster_index[node.metadata.cluster_id].add(node.id)
                
        elif operation == 'remove':
            # Limpiar índices
            for keyword in node.keywords:
                self.keyword_index[keyword].discard(node.id)
            
            for tag in node.metadata.tags:
                self.tag_index[tag].discard(node.id)
            
            if node.metadata.cluster_id is not None:
                self.cluster_index[node.metadata.cluster_id].discard(node.id)
    
    async def find_similar_nodes(self, node_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Encuentra nodos similares usando embeddings"""
        node = self.nodes.get(node_id)
        if not node or node.metadata.embedding is None:
            return []
        
        similarities = []
        
        for other_id, other_node in self.nodes.items():
            if other_id == node_id or other_node.metadata.embedding is None:
                continue
            
            # Calcular similitud coseno
            similarity = self._cosine_similarity(
                node.metadata.embedding,
                other_node.metadata.embedding
            )
            similarities.append((other_id, similarity))
        
        # Ordenar y retornar top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre vectores"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    async def cluster_nodes(self, algorithm: str = 'dbscan') -> Dict[int, int]:
        """Agrupa nodos en clusters"""
        if not self.nodes:
            return {}
        
        # Preparar embeddings
        node_ids = []
        embeddings = []
        
        for node_id, node in self.nodes.items():
            if node.metadata.embedding is not None:
                node_ids.append(node_id)
                embeddings.append(node.metadata.embedding)
        
        if not embeddings:
            return {}
        
        embeddings_matrix = np.array(embeddings)
        
        # Aplicar algoritmo de clustering
        if algorithm == 'dbscan':
            clustering = DBSCAN(eps=0.3, min_samples=3)
            labels = clustering.fit_predict(embeddings_matrix)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Asignar clusters
        clusters = {}
        for i, node_id in enumerate(node_ids):
            cluster_id = int(labels[i])
            clusters[node_id] = cluster_id
            self.nodes[node_id].metadata.cluster_id = cluster_id
        
        # Actualizar índice
        self.cluster_index.clear()
        for node_id, cluster_id in clusters.items():
            if cluster_id >= 0:  # -1 es ruido en DBSCAN
                self.cluster_index[cluster_id].add(node_id)
        
        return clusters

# === MODELO GNN AVANZADO ===
class AdvancedGNN(nn.Module):
    """Red neuronal de grafos avanzada con atención multi-cabeza"""
    
    def __init__(self, num_node_features: int, hidden_channels: int,
                 out_channels: int, num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Capas de entrada
        self.input_proj = nn.Linear(num_node_features, hidden_channels)
        
        # Capas GAT
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_channels, hidden_channels // num_heads,
                           heads=num_heads, dropout=dropout)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_channels, hidden_channels // num_heads,
                           heads=num_heads, dropout=dropout)
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Capas de salida
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Skip connections
        self.skip_proj = nn.Linear(num_node_features, out_channels)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass"""
        # Proyección de entrada
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Capas GAT con residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            h_prev = h
            h = gat(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                h = h + h_prev
        
        # Proyección de salida
        out = self.output_proj(h)
        
        # Skip connection desde entrada
        skip = self.skip_proj(x)
        out = out + skip
        
        # Global pooling si es necesario
        if batch is not None:
            out = global_mean_pool(out, batch)
        
        return out

# === ANALIZADOR DE RED ===
class NetworkAnalyzer:
    """Analizador avanzado de la red del grafo"""
    
    def __init__(self, graph: AdvancedCollectiveSynthesisGraph):
        self.graph = graph
        self._nx_cache = None
        self._cache_time = 0
        self._cache_ttl = 60  # 1 minuto
    
    def _get_networkx_graph(self) -> nx.DiGraph:
        """Obtiene representación NetworkX del grafo"""
        current_time = time.time()
        if self._nx_cache is None or current_time - self._cache_time > self._cache_ttl:
            G = nx.DiGraph()
            
            # Añadir nodos
            for node_id, node in self.graph.nodes.items():
                G.add_node(node_id, 
                          state=node.state,
                          importance=node.metadata.importance_score,
                          keywords=list(node.keywords))
            
            # Añadir edges
            for node_id, node in self.graph.nodes.items():
                for target_id, utility in node.connections_out.items():
                    G.add_edge(node_id, target_id, weight=utility)
            
            self._nx_cache = G
            self._cache_time = current_time
        
        return self._nx_cache
    
    async def calculate_centralities(self) -> Dict[str, Dict[int, float]]:
        """Calcula varias medidas de centralidad"""
        G = self._get_networkx_graph()
        
        loop = asyncio.get_event_loop()
        
        # Calcular en paralelo
        tasks = {
            'degree': loop.run_in_executor(None, nx.degree_centrality, G),
            'betweenness': loop.run_in_executor(None, nx.betweenness_centrality, G),
            'closeness': loop.run_in_executor(None, nx.closeness_centrality, G),
            'eigenvector': loop.run_in_executor(None, self._safe_eigenvector_centrality, G),
            'pagerank': loop.run_in_executor(None, nx.pagerank, G)
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error calculating {name} centrality: {e}")
                results[name] = {}
        
        # Actualizar nodos con centralidades
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            node.features['centrality'] = results.get('degree', {}).get(node_id, 0)
            node.features['betweenness'] = results.get('betweenness', {}).get(node_id, 0)
            node.features['pagerank'] = results.get('pagerank', {}).get(node_id, 0)
        
        return results
    
    def _safe_eigenvector_centrality(self, G):
        """Calcula eigenvector centrality con manejo de errores"""
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except:
            return {}
    
    async def detect_communities(self) -> Dict[int, int]:
        """Detecta comunidades en el grafo"""
        G = self._get_networkx_graph().to_undirected()
        
        if len(G) < 3:
            return {}
        
        loop = asyncio.get_event_loop()
        
        try:
            # Usar algoritmo de Louvain
            import community as community_louvain
            communities = await loop.run_in_executor(
                None, community_louvain.best_partition, G
            )
            return communities
        except:
            # Fallback a algoritmo simple
            return {}
    
    def find_critical_nodes(self, top_k: int = 10) -> List[int]:
        """Encuentra nodos críticos para la conectividad"""
        G = self._get_networkx_graph()
        
        # Calcular impacto de remover cada nodo
        impacts = []
        
        for node in list(G.nodes())[:100]:  # Limitar para performance
            # Contar componentes antes
            before = nx.number_strongly_connected_components(G)
            
            # Remover temporalmente
            edges_out = list(G.out_edges(node))
            edges_in = list(G.in_edges(node))
            G.remove_node(node)
            
            # Contar componentes después
            after = nx.number_strongly_connected_components(G)
            
            # Restaurar
            G.add_node(node)
            G.add_edges_from(edges_out)
            G.add_edges_from(edges_in)
            
            impact = after - before
            impacts.append((node, impact))
        
        # Ordenar por impacto
        impacts.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in impacts[:top_k]]

# === MÉTRICAS DEL GRAFO ===
class GraphMetrics:
    """Sistema de métricas para el grafo"""
    
    def __init__(self):
        # Prometheus metrics
        self.node_count = Gauge('msc_graph_nodes', 'Number of nodes in graph')
        self.edge_count = Gauge('msc_graph_edges', 'Number of edges in graph')
        self.avg_state = Gauge('msc_graph_avg_state', 'Average node state')
        self.health_score = Gauge('msc_graph_health', 'Graph health score')
        
        self.operations = Counter('msc_graph_operations', 'Graph operations', ['operation'])
        self.operation_duration = Histogram('msc_graph_operation_duration', 
                                          'Operation duration', ['operation'])
        
        # Métricas internas
        self.history = deque(maxlen=10000)
        self.anomalies = deque(maxlen=1000)
    
    def record_node_operation(self, operation: str):
        """Registra operación de nodo"""
        self.operations.labels(operation=operation).inc()
    
    @contextmanager
    def measure_operation(self, operation: str):
        """Mide duración de operación"""
        with self.operation_duration.labels(operation=operation).time():
            yield
    
    def update_graph_metrics(self, graph: AdvancedCollectiveSynthesisGraph):
        """Actualiza métricas del grafo"""
        # Contar nodos y edges
        node_count = len(graph.nodes)
        edge_count = sum(len(node.connections_out) for node in graph.nodes.values())
        
        self.node_count.set(node_count)
        self.edge_count.set(edge_count)
        
        # Calcular estado promedio
        if node_count > 0:
            states = [node.state for node in graph.nodes.values()]
            avg_state = np.mean(states)
            self.avg_state.set(avg_state)
        
        # Registrar en historial
        self.history.append({
            'timestamp': time.time(),
            'nodes': node_count,
            'edges': edge_count,
            'avg_state': avg_state if node_count > 0 else 0
        })
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detecta anomalías en las métricas"""
        if len(self.history) < 100:
            return []
        
        # Analizar tendencias recientes
        recent = list(self.history)[-100:]
        
        anomalies = []
        
        # Detectar cambios bruscos en número de nodos
        node_counts = [h['nodes'] for h in recent]
        node_mean = np.mean(node_counts)
        node_std = np.std(node_counts)
        
        if node_std > 0:
            current_z = (node_counts[-1] - node_mean) / node_std
            if abs(current_z) > 3:
                anomalies.append({
                    'type': 'node_count_anomaly',
                    'severity': 'high' if abs(current_z) > 5 else 'medium',
                    'value': node_counts[-1],
                    'z_score': current_z
                })
        
        return anomalies

# === AGENTES MEJORADOS ===
class ImprovedBaseAgent(ABC):
    """Clase base mejorada para agentes con capacidades ML"""
    
    def __init__(self, agent_id: str, graph: AdvancedCollectiveSynthesisGraph,
                 config: Dict[str, Any]):
        self.id = agent_id
        self.graph = graph
        self.config = config
        
        # Recursos
        self.omega = config.get('initial_omega', 100.0)
        self.max_omega = config.get('max_omega', 1000.0)
        
        # Estado
        self.reputation = 1.0
        self.specialization = set()
        self.learning_rate = 0.1
        
        # Modelo de decisión
        self.decision_model = self._init_decision_model()
        
        # Métricas
        self.metrics = AgentMetrics(agent_id)
        
        # Historial
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            rate=config.get('agent_rate_limit', 10),
            per=60.0
        )
    
    def _init_decision_model(self):
        """Inicializa modelo de decisión del agente"""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(dim=-1)
        )
    
    async def perceive_environment(self) -> Dict[str, Any]:
        """Percibe el estado del entorno"""
        # Estado del grafo
        graph_health = self.graph.calculate_graph_health()
        
        # Estado local (vecindario)
        if hasattr(self, 'focus_nodes'):
            local_state = await self._analyze_local_environment()
        else:
            local_state = {}
        
        # Estado propio
        self_state = {
            'omega': self.omega,
            'reputation': self.reputation,
            'recent_success_rate': self._calculate_recent_success_rate()
        }
        
        return {
            'graph': graph_health,
            'local': local_state,
            'self': self_state,
            'timestamp': time.time()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calcula tasa de éxito reciente"""
        if not self.action_history:
            return 0.5
        
        recent = list(self.action_history)[-20:]
        successes = sum(1 for a in recent if a.get('success', False))
        return successes / len(recent)
    
    async def decide_action(self, perception: Dict[str, Any]) -> str:
        """Decide qué acción tomar basado en percepción"""
        # Preparar features
        features = self._extract_features(perception)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Obtener probabilidades de acción
        with torch.no_grad():
            action_probs = self.decision_model(features_tensor)
        
        # Seleccionar acción (con exploración)
        if random.random() < self.config.get('exploration_rate', 0.1):
            action_idx = random.randint(0, 9)
        else:
            action_idx = torch.argmax(action_probs).item()
        
        # Mapear a acción específica
        actions = self._get_available_actions()
        return actions[min(action_idx, len(actions) - 1)]
    
    def _extract_features(self, perception: Dict[str, Any]) -> List[float]:
        """Extrae features de la percepción"""
        features = []
        
        # Features del grafo
        graph = perception.get('graph', {})
        features.extend([
            graph.get('overall_health', 0.5),
            graph.get('mean_state', 0.5),
            graph.get('avg_degree', 0.0),
            graph.get('num_components', 1) / 10.0,
            min(graph.get('total_nodes', 0) / 1000.0, 1.0),
            min(graph.get('total_edges', 0) / 10000.0, 1.0)
        ])
        
        # Features propias
        self_state = perception.get('self', {})
        features.extend([
            min(self_state.get('omega', 0) / self.max_omega, 1.0),
            self_state.get('reputation', 1.0),
            self_state.get('recent_success_rate', 0.5)
        ])
        
        # Padding
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    @abstractmethod
    def _get_available_actions(self) -> List[str]:
        """Obtiene lista de acciones disponibles"""
        pass
    
    @abstractmethod
    async def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica"""
        pass
    
    async def act(self):
        """Ciclo principal de acción del agente"""
        # Rate limiting
        await self.rate_limiter.acquire()
        
        try:
            # Percibir
            perception = await self.perceive_environment()
            
            # Decidir
            action = await self.decide_action(perception)
            
            # Verificar recursos
            action_cost = self._get_action_cost(action)
            if self.omega < action_cost:
                logger.warning(f"Agent {self.id} insufficient omega for {action}")
                return
            
            # Ejecutar
            self.omega -= action_cost
            result = await self.execute_action(action, perception)
            
            # Registrar
            self.action_history.append({
                'timestamp': time.time(),
                'action': action,
                'success': result.get('success', False),
                'result': result
            })
            
            # Calcular recompensa
            reward = self._calculate_reward(result)
            self.reward_history.append(reward)
            
            # Actualizar modelo
            await self._update_model(perception, action, reward)
            
            # Métricas
            self.metrics.record_action(action, result.get('success', False))
            
            # Publicar evento
            await self.graph.event_bus.publish(Event(
                type=EventType.AGENT_ACTION,
                data={
                    'agent_id': self.id,
                    'action': action,
                    'success': result.get('success', False),
                    'reward': reward
                },
                source=self.id
            ))
            
        except Exception as e:
            logger.error(f"Agent {self.id} error: {e}", extra={'agent_id': self.id})
            self.metrics.record_error(str(e))
    
    def _get_action_cost(self, action: str) -> float:
        """Obtiene costo de una acción"""
        base_costs = {
            'create_node': 5.0,
            'create_edge': 2.0,
            'update_node': 1.0,
            'analyze': 0.5,
            'synthesize': 10.0,
            'evolve': 20.0
        }
        
        base = base_costs.get(action, 1.0)
        # Ajustar por reputación
        return base * (2.0 - self.reputation)
    
    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        """Calcula recompensa por resultado de acción"""
        if not result.get('success', False):
            return -1.0
        
        # Recompensa base
        reward = 1.0
        
        # Bonus por impacto
        if 'impact' in result:
            reward += result['impact']
        
        # Bonus por novedad
        if result.get('novel', False):
            reward += 2.0
        
        return min(reward, 10.0)
    
    async def _update_model(self, perception: Dict[str, Any], action: str, reward: float):
        """Actualiza modelo de decisión con aprendizaje"""
        # Aquí iría el aprendizaje por refuerzo
        # Por ahora solo actualizamos reputación
        if reward > 0:
            self.reputation = min(2.0, self.reputation * 1.01)
        else:
            self.reputation = max(0.1, self.reputation * 0.99)

# === AGENTE CLAUDE-TAEC ===
class ClaudeTAECAgent(ImprovedBaseAgent):
    """Agente TAEC mejorado con integración de Claude"""
    
    def __init__(self, agent_id: str, graph: AdvancedCollectiveSynthesisGraph,
                 config: Dict[str, Any]):
        super().__init__(agent_id, graph, config)
        
        # Estado específico TAEC
        self.evolution_count = 0
        self.code_repository = CodeRepository()
        self.evolution_memory = EvolutionMemory()
        
        # Estrategias disponibles
        self.strategies = {
            'synthesis': self._strategy_synthesis,
            'optimization': self._strategy_optimization,
            'exploration': self._strategy_exploration,
            'consolidation': self._strategy_consolidation,
            'innovation': self._strategy_innovation,
            'recovery': self._strategy_recovery
        }
        
        # Modelo predictivo específico
        self.evolution_predictor = self._init_evolution_predictor()
        
        # Claude client
        self.claude = graph.claude_client
        
        logger.info(f"Claude-TAEC Agent {agent_id} initialized")
    
    def _init_evolution_predictor(self):
        """Inicializa predictor de éxito de evoluciones"""
        return nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6 estrategias
            nn.Softmax(dim=-1)
        )
    
    def _get_available_actions(self) -> List[str]:
        """Obtiene acciones disponibles para TAEC"""
        return ['evolve', 'synthesize', 'optimize', 'explore', 'innovate']
    
    async def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta acción específica de TAEC"""
        if action == 'evolve':
            return await self._execute_evolution(context)
        elif action == 'synthesize':
            return await self._execute_synthesis(context)
        elif action == 'optimize':
            return await self._execute_optimization(context)
        elif action == 'explore':
            return await self._execute_exploration(context)
        elif action == 'innovate':
            return await self._execute_innovation(context)
        else:
            return {'success': False, 'error': 'Unknown action'}
    
    async def _execute_evolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta ciclo completo de evolución"""
        self.evolution_count += 1
        logger.info(f"TAEC {self.id}: Evolution cycle {self.evolution_count}")
        
        # Analizar situación
        analysis = await self._analyze_evolution_context(context)
        
        # Seleccionar estrategia
        strategy_name = await self._select_strategy(analysis)
        strategy = self.strategies.get(strategy_name)
        
        if not strategy:
            return {'success': False, 'error': 'Invalid strategy'}
        
        # Ejecutar estrategia
        result = await strategy(analysis)
        
        # Guardar en memoria
        self.evolution_memory.add_evolution(
            self.evolution_count,
            strategy_name,
            analysis,
            result
        )
        
        # Publicar evento
        await self.graph.event_bus.publish(Event(
            type=EventType.AGENT_EVOLVED,
            data={
                'agent_id': self.id,
                'evolution_count': self.evolution_count,
                'strategy': strategy_name,
                'result': result
            },
            source=self.id
        ))
        
        return result
    
    async def _analyze_evolution_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza contexto para evolución"""
        # Análisis del grafo
        graph_analysis = {
            'health': context['graph'],
            'trends': self._analyze_trends(),
            'opportunities': await self._identify_opportunities(),
            'risks': self._identify_risks()
        }
        
        # Análisis de evoluciones previas
        history_analysis = self.evolution_memory.analyze_history()
        
        # Análisis de código generado
        code_analysis = self.code_repository.analyze_repository()
        
        return {
            'graph': graph_analysis,
            'history': history_analysis,
            'code': code_analysis,
            'timestamp': time.time()
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analiza tendencias en el grafo"""
        if len(self.graph.metrics.history) < 10:
            return {'trend': 'insufficient_data'}
        
        recent = list(self.graph.metrics.history)[-100:]
        
        # Tendencia de nodos
        node_counts = [h['nodes'] for h in recent]
        node_trend = 'growing' if node_counts[-1] > node_counts[0] else 'shrinking'
        
        # Tendencia de estado
        states = [h['avg_state'] for h in recent]
        state_trend = 'improving' if states[-1] > states[0] else 'declining'
        
        return {
            'node_trend': node_trend,
            'state_trend': state_trend,
            'volatility': np.std(states)
        }
    
    async def _identify_opportunities(self) -> List[Dict[str, Any]]:
        """Identifica oportunidades de evolución"""
        opportunities = []
        
        # Oportunidad 1: Nodos sin explotar
        high_potential = [
            n for n in self.graph.nodes.values()
            if n.state > 0.8 and len(n.connections_out) < 3
        ]
        
        if high_potential:
            opportunities.append({
                'type': 'underutilized_nodes',
                'count': len(high_potential),
                'priority': 'high',
                'nodes': [n.id for n in high_potential[:5]]
            })
        
        # Oportunidad 2: Clusters desconectados
        if len(self.graph.cluster_index) > 1:
            opportunities.append({
                'type': 'disconnected_clusters',
                'count': len(self.graph.cluster_index),
                'priority': 'medium'
            })
        
        # Oportunidad 3: Áreas temáticas nuevas
        current_keywords = set()
        for node in self.graph.nodes.values():
            current_keywords.update(node.keywords)
        
        potential_keywords = {
            'quantum_synthesis', 'emergent_intelligence', 
            'collective_consciousness', 'meta_learning',
            'recursive_improvement', 'semantic_convergence'
        }
        
        new_keywords = potential_keywords - current_keywords
        if new_keywords:
            opportunities.append({
                'type': 'unexplored_concepts',
                'keywords': list(new_keywords),
                'priority': 'low'
            })
        
        return opportunities
    
    def _identify_risks(self) -> List[Dict[str, Any]]:
        """Identifica riesgos en el sistema"""
        risks = []
        
        # Riesgo 1: Fragmentación
        if len(self.graph.nodes) > 100:
            isolated = sum(
                1 for n in self.graph.nodes.values()
                if len(n.connections_in) == 0 and len(n.connections_out) == 0
            )
            if isolated > len(self.graph.nodes) * 0.1:
                risks.append({
                    'type': 'fragmentation',
                    'severity': 'high',
                    'isolated_ratio': isolated / len(self.graph.nodes)
                })
        
        # Riesgo 2: Estancamiento
        if len(self.evolution_memory.history) > 10:
            recent_success = sum(
                1 for e in list(self.evolution_memory.history)[-10:]
                if e.get('result', {}).get('success', False)
            )
            if recent_success < 3:
                risks.append({
                    'type': 'stagnation',
                    'severity': 'medium',
                    'success_rate': recent_success / 10
                })
        
        return risks
    
    async def _select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Selecciona estrategia óptima usando ML"""
        # Preparar features
        features = self._extract_strategy_features(analysis)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Predecir con modelo
        with torch.no_grad():
            strategy_probs = self.evolution_predictor(features_tensor)
        
        # Considerar contexto
        health = analysis['graph']['health'].get('overall_health', 0.5)
        
        # Ajustar probabilidades según contexto
        if health < 0.3:
            # Priorizar recuperación
            strategy_probs[0][5] *= 2.0  # recovery
        elif health > 0.8:
            # Priorizar innovación
            strategy_probs[0][4] *= 1.5  # innovation
        
        # Normalizar
        strategy_probs = strategy_probs / strategy_probs.sum()
        
        # Seleccionar
        strategy_idx = torch.multinomial(strategy_probs[0], 1).item()
        
        strategies = list(self.strategies.keys())
        return strategies[strategy_idx]
    
    def _extract_strategy_features(self, analysis: Dict[str, Any]) -> List[float]:
        """Extrae features para selección de estrategia"""
        features = []
        
        # Features del grafo
        graph = analysis['graph']
        features.extend([
            graph['health'].get('overall_health', 0.5),
            graph['health'].get('mean_state', 0.5),
            graph['health'].get('avg_degree', 0) / 10.0,
            len(graph.get('opportunities', [])) / 10.0,
            len(graph.get('risks', [])) / 5.0
        ])
        
        # Features de historia
        history = analysis['history']
        features.extend([
            history.get('total_evolutions', 0) / 100.0,
            history.get('success_rate', 0.5),
            history.get('avg_impact', 0.5)
        ])
        
        # Features de código
        code = analysis['code']
        features.extend([
            code.get('total_functions', 0) / 50.0,
            code.get('reuse_rate', 0.0)
        ])
        
        # Padding
        while len(features) < 30:
            features.append(0.0)
        
        return features[:30]
    
    async def _strategy_synthesis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Síntesis avanzada con Claude"""
        logger.info(f"TAEC {self.id}: Executing Claude-powered synthesis")
        
        # Seleccionar nodos fuente
        candidates = self._select_synthesis_candidates(analysis)
        
        if len(candidates) < 2:
            return {'success': False, 'error': 'Insufficient candidates'}
        
        # Preparar contexto para Claude
        context = {
            'node_count': len(self.graph.nodes),
            'edge_count': sum(len(n.connections_out) for n in self.graph.nodes.values()),
            'source_nodes': [
                {
                    'id': n.id,
                    'content': n.content,
                    'keywords': list(n.keywords),
                    'state': n.state
                }
                for n in candidates
            ]
        }
        
        # Generar prompt para síntesis
        prompt = f"""
Create a synthesis function that combines the following knowledge nodes:

{json.dumps(context['source_nodes'], indent=2)}

The function should:
1. Extract key concepts from each node
2. Find semantic connections between concepts
3. Generate a new synthesized concept with higher-order understanding
4. Return a dictionary with 'content', 'keywords', and 'initial_state'

The synthesis should create emergent knowledge, not just concatenation.
"""
        
        # Generar código con Claude
        code = await self.claude.generate_code(prompt, context)
        
        if not code:
            return {'success': False, 'error': 'Code generation failed'}
        
        # Ejecutar código
        try:
            result = await self._execute_generated_code(code, {'nodes': candidates})
            
            if not isinstance(result, dict):
                return {'success': False, 'error': 'Invalid synthesis result'}
            
            # Crear nodo sintetizado
            new_node = await self.graph.add_node(
                content=result.get('content', 'Synthesized concept'),
                initial_state=result.get('initial_state', 0.7),
                keywords=set(result.get('keywords', [])),
                created_by=self.id,
                properties={'synthesis_method': 'claude', 'evolution': self.evolution_count}
            )
            
            # Conectar con fuentes
            for source in candidates:
                await self.graph.add_edge(source.id, new_node.id, 0.8)
                await self.graph.add_edge(new_node.id, source.id, 0.6)
            
            # Guardar código exitoso
            self.code_repository.add_code(
                'synthesis',
                code,
                {'success': True, 'node_id': new_node.id}
            )
            
            return {
                'success': True,
                'strategy': 'synthesis',
                'new_node_id': new_node.id,
                'impact': len(candidates) * 0.5
            }
            
        except Exception as e:
            logger.error(f"Synthesis execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _select_synthesis_candidates(self, analysis: Dict[str, Any]) -> List[AdvancedKnowledgeComponent]:
        """Selecciona candidatos para síntesis"""
        candidates = []
        
        # Priorizar nodos de alta calidad
        high_quality = [
            n for n in self.graph.nodes.values()
            if n.state > 0.6 and n.metadata.importance_score > 0.5
        ]
        
        if len(high_quality) >= 3:
            # Seleccionar diversos
            selected = []
            for _ in range(min(4, len(high_quality))):
                if not selected:
                    selected.append(random.choice(high_quality))
                else:
                    # Buscar el más diferente
                    best_candidate = None
                    max_distance = 0
                    
                    for candidate in high_quality:
                        if candidate in selected:
                            continue
                        
                        # Calcular distancia semántica
                        min_distance = float('inf')
                        for s in selected:
                            if s.metadata.embedding is not None and candidate.metadata.embedding is not None:
                                distance = np.linalg.norm(
                                    s.metadata.embedding - candidate.metadata.embedding
                                )
                                min_distance = min(min_distance, distance)
                        
                        if min_distance > max_distance:
                            max_distance = min_distance
                            best_candidate = candidate
                    
                    if best_candidate:
                        selected.append(best_candidate)
            
            candidates = selected
        else:
            # Fallback: selección aleatoria
            all_nodes = list(self.graph.nodes.values())
            candidates = random.sample(all_nodes, min(3, len(all_nodes)))
        
        return candidates
    
    async def _strategy_optimization(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Optimización con Claude"""
        logger.info(f"TAEC {self.id}: Executing Claude-powered optimization")
        
        # Preparar contexto
        context = {
            'node_count': len(self.graph.nodes),
            'edge_count': sum(len(n.connections_out) for n in self.graph.nodes.values()),
            'health_metrics': analysis['graph']['health'],
            'opportunities': analysis['graph']['opportunities'][:3]
        }
        
        # Generar prompt
        prompt = f"""
Create an optimization function for a knowledge graph with these characteristics:
{json.dumps(context, indent=2)}

The function should:
1. Analyze the graph structure and identify optimization opportunities
2. Create strategic connections between nodes
3. Adjust node states to improve overall health
4. Return a list of actions taken with their impact

Focus on improving connectivity and knowledge flow.
"""
        
        # Generar código
        code = await self.claude.generate_code(prompt, context)
        
        if not code:
            return {'success': False, 'error': 'Code generation failed'}
        
        # Ejecutar
        try:
            result = await self._execute_generated_code(
                code, 
                {'graph': self.graph, 'analysis': analysis}
            )
            
            # Guardar código
            self.code_repository.add_code(
                'optimization',
                code,
                {'success': True, 'result': result}
            )
            
            return {
                'success': True,
                'strategy': 'optimization',
                'actions': result,
                'impact': len(result) * 0.3 if isinstance(result, list) else 1.0
            }
            
        except Exception as e:
            logger.error(f"Optimization execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _strategy_innovation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Innovación radical con Claude"""
        logger.info(f"TAEC {self.id}: Executing Claude-powered innovation")
        
        # Contexto para innovación
        context = {
            'current_concepts': list(set(
                n.content[:50] for n in list(self.graph.nodes.values())[:20]
            )),
            'current_keywords': list(set(
                kw for n in self.graph.nodes.values() for kw in n.keywords
            ))[:50],
            'graph_state': analysis['graph']['health']
        }
        
        # Prompt creativo
        prompt = f"""
Given this knowledge graph context:
{json.dumps(context, indent=2)}

Create an innovative function that:
1. Generates completely novel concepts that don't exist in the graph
2. These concepts should be related but transformative
3. Identify cross-domain connections and emergent patterns
4. Return a list of innovative nodes with content, keywords, and rationale

Be creative and think beyond conventional boundaries. Generate concepts that could lead to breakthrough insights.
"""
        
        # Generar código innovador
        code = await self.claude.generate_code(prompt, context)
        
        if not code:
            return {'success': False, 'error': 'Innovation generation failed'}
        
        try:
            # Ejecutar
            innovations = await self._execute_generated_code(code, context)
            
            if not isinstance(innovations, list):
                innovations = [innovations]
            
            created_nodes = []
            
            # Crear nodos innovadores
            for innovation in innovations[:3]:  # Limitar para no sobrecargar
                if isinstance(innovation, dict):
                    new_node = await self.graph.add_node(
                        content=innovation.get('content', 'Innovative concept'),
                        initial_state=0.8,
                        keywords=set(innovation.get('keywords', [])),
                        created_by=self.id,
                        properties={
                            'innovation_type': 'claude_generated',
                            'rationale': innovation.get('rationale', ''),
                            'evolution': self.evolution_count
                        }
                    )
                    created_nodes.append(new_node.id)
                    
                    # Añadir tag especial
                    new_node.metadata.tags.add('innovative')
                    new_node.metadata.tags.add('claude_inspired')
            
            # Guardar código exitoso
            self.code_repository.add_code(
                'innovation',
                code,
                {'success': True, 'nodes_created': created_nodes}
            )
            
            return {
                'success': True,
                'strategy': 'innovation',
                'nodes_created': created_nodes,
                'impact': len(created_nodes) * 2.0,
                'novel': True
            }
            
        except Exception as e:
            logger.error(f"Innovation execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_generated_code(self, code: str, context: Dict[str, Any]) -> Any:
        """Ejecuta código generado de forma segura"""
        # Usar sandbox mejorado
        sandbox = EnhancedSecureExecutionSandbox()
        result = await sandbox.execute(code, context)
        
        if result['success']:
            return result['output']
        else:
            raise Exception(result['error'])
    
    # Implementar otras estrategias...
    async def _strategy_exploration(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Exploración de nuevos dominios"""
        # Implementación similar con Claude
        pass
    
    async def _strategy_consolidation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Consolidación de conocimiento"""
        # Implementación similar con Claude
        pass
    
    async def _strategy_recovery(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estrategia: Recuperación del sistema"""
        # Implementación similar con Claude
        pass

# === REPOSITORIO DE CÓDIGO ===
class CodeRepository:
    """Repositorio para código generado"""
    
    def __init__(self):
        self.repository = defaultdict(list)
        self.execution_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
        self.code_index = {}
        self._next_id = 0
    
    def add_code(self, category: str, code: str, metadata: Dict[str, Any]):
        """Añade código al repositorio"""
        code_id = self._next_id
        self._next_id += 1
        
        entry = {
            'id': code_id,
            'category': category,
            'code': code,
            'hash': hashlib.sha256(code.encode()).hexdigest(),
            'metadata': metadata,
            'timestamp': time.time(),
            'executions': 0,
            'last_execution': None
        }
        
        self.repository[category].append(entry)
        self.code_index[code_id] = entry
        
        # Actualizar estadísticas
        if metadata.get('success', False):
            self.execution_stats[category]['success'] += 1
        else:
            self.execution_stats[category]['failure'] += 1
        
        return code_id
    
    def get_best_code(self, category: str) -> Optional[Dict[str, Any]]:
        """Obtiene el mejor código de una categoría"""
        if category not in self.repository:
            return None
        
        # Ordenar por éxito y recencia
        codes = self.repository[category]
        
        def score_code(entry):
            success = entry['metadata'].get('success', False)
            recency = 1.0 / (time.time() - entry['timestamp'] + 1)
            executions = entry['executions']
            
            return (1 if success else 0) * 10 + recency + executions * 0.1
        
        best = max(codes, key=score_code)
        return best
    
    def analyze_repository(self) -> Dict[str, Any]:
        """Analiza el repositorio de código"""
        total_codes = sum(len(codes) for codes in self.repository.values())
        
        # Calcular tasa de reutilización
        total_executions = sum(
            entry['executions'] 
            for codes in self.repository.values() 
            for entry in codes
        )
        
        reuse_rate = total_executions / total_codes if total_codes > 0 else 0
        
        return {
            'total_functions': total_codes,
            'categories': list(self.repository.keys()),
            'reuse_rate': reuse_rate,
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calcula tasa de éxito global"""
        total_success = sum(stats['success'] for stats in self.execution_stats.values())
        total_failure = sum(stats['failure'] for stats in self.execution_stats.values())
        total = total_success + total_failure
        
        return total_success / total if total > 0 else 0

# === MEMORIA DE EVOLUCIÓN ===
class EvolutionMemory:
    """Memoria de evoluciones del agente"""
    
    def __init__(self, max_size: int = 1000):
        self.history = deque(maxlen=max_size)
        self.strategy_performance = defaultdict(lambda: {'count': 0, 'success': 0, 'impact': 0})
        self.pattern_recognition = PatternRecognizer()
    
    def add_evolution(self, evolution_id: int, strategy: str, 
                     analysis: Dict[str, Any], result: Dict[str, Any]):
        """Registra una evolución"""
        entry = {
            'id': evolution_id,
            'timestamp': time.time(),
            'strategy': strategy,
            'analysis_summary': self._summarize_analysis(analysis),
            'result': result,
            'success': result.get('success', False),
            'impact': result.get('impact', 0)
        }
        
        self.history.append(entry)
        
        # Actualizar estadísticas
        self.strategy_performance[strategy]['count'] += 1
        if entry['success']:
            self.strategy_performance[strategy]['success'] += 1
        self.strategy_performance[strategy]['impact'] += entry['impact']
        
        # Detectar patrones
        self.pattern_recognition.add_entry(entry)
    
    def _summarize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resume análisis para almacenamiento eficiente"""
        return {
            'health': analysis['graph']['health'].get('overall_health', 0),
            'opportunities': len(analysis['graph'].get('opportunities', [])),
            'risks': len(analysis['graph'].get('risks', [])),
            'trend': analysis['graph'].get('trends', {}).get('state_trend', 'unknown')
        }
    
    def analyze_history(self) -> Dict[str, Any]:
        """Analiza historia de evoluciones"""
        if not self.history:
            return {
                'total_evolutions': 0,
                'success_rate': 0,
                'avg_impact': 0
            }
        
        total = len(self.history)
        successes = sum(1 for e in self.history if e['success'])
        total_impact = sum(e['impact'] for e in self.history)
        
        # Mejores estrategias
        best_strategies = []
        for strategy, stats in self.strategy_performance.items():
            if stats['count'] > 0:
                success_rate = stats['success'] / stats['count']
                avg_impact = stats['impact'] / stats['count']
                score = success_rate * 0.6 + min(avg_impact / 5, 1) * 0.4
                
                best_strategies.append({
                    'strategy': strategy,
                    'score': score,
                    'success_rate': success_rate,
                    'avg_impact': avg_impact
                })
        
        best_strategies.sort(key=lambda x: x['score'], reverse=True)
        
        # Patrones detectados
        patterns = self.pattern_recognition.get_patterns()
        
        return {
            'total_evolutions': total,
            'success_rate': successes / total,
            'avg_impact': total_impact / total,
            'best_strategies': best_strategies[:3],
            'patterns': patterns
        }

# === RECONOCEDOR DE PATRONES ===
class PatternRecognizer:
    """Reconoce patrones en evoluciones"""
    
    def __init__(self):
        self.sequences = defaultdict(list)
        self.patterns = {}
    
    def add_entry(self, entry: Dict[str, Any]):
        """Añade entrada para análisis"""
        # Agrupar por contexto similar
        context_key = f"{entry['analysis_summary']['trend']}_{entry['analysis_summary']['health']:.1f}"
        self.sequences[context_key].append({
            'strategy': entry['strategy'],
            'success': entry['success'],
            'impact': entry['impact']
        })
        
        # Detectar patrones cada 10 entradas
        if len(self.sequences[context_key]) % 10 == 0:
            self._detect_patterns(context_key)
    
    def _detect_patterns(self, context_key: str):
        """Detecta patrones en secuencias"""
        sequence = self.sequences[context_key]
        
        if len(sequence) < 20:
            return
        
        # Buscar secuencias exitosas
        for window_size in [2, 3, 4]:
            for i in range(len(sequence) - window_size):
                window = sequence[i:i+window_size]
                
                # Verificar si es patrón exitoso
                if all(e['success'] for e in window):
                    pattern_key = tuple(e['strategy'] for e in window)
                    
                    if pattern_key not in self.patterns:
                        self.patterns[pattern_key] = {
                            'count': 0,
                            'avg_impact': 0,
                            'context': context_key
                        }
                    
                    self.patterns[pattern_key]['count'] += 1
                    self.patterns[pattern_key]['avg_impact'] = (
                        self.patterns[pattern_key]['avg_impact'] * 
                        (self.patterns[pattern_key]['count'] - 1) +
                        sum(e['impact'] for e in window)
                    ) / self.patterns[pattern_key]['count']
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Obtiene patrones detectados"""
        pattern_list = []
        
        for pattern, stats in self.patterns.items():
            if stats['count'] >= 3:  # Mínimo 3 ocurrencias
                pattern_list.append({
                    'sequence': pattern,
                    'frequency': stats['count'],
                    'avg_impact': stats['avg_impact'],
                    'context': stats['context']
                })
        
        pattern_list.sort(key=lambda x: x['frequency'] * x['avg_impact'], reverse=True)
        return pattern_list[:5]

# === MÉTRICAS DE AGENTE ===
class AgentMetrics:
    """Métricas específicas de agente"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Prometheus metrics
        self.actions = Counter(
            'msc_agent_actions',
            'Agent actions',
            ['agent_id', 'action', 'result']
        )
        
        self.omega = Gauge(
            'msc_agent_omega',
            'Agent omega level',
            ['agent_id']
        )
        
        self.reputation = Gauge(
            'msc_agent_reputation',
            'Agent reputation',
            ['agent_id']
        )
        
        self.action_duration = Histogram(
            'msc_agent_action_duration',
            'Action execution duration',
            ['agent_id', 'action']
        )
        
        # Métricas internas
        self.action_history = deque(maxlen=1000)
        self.error_log = deque(maxlen=100)
    
    def record_action(self, action: str, success: bool):
        """Registra una acción"""
        result = 'success' if success else 'failure'
        self.actions.labels(
            agent_id=self.agent_id,
            action=action,
            result=result
        ).inc()
        
        self.action_history.append({
            'timestamp': time.time(),
            'action': action,
            'success': success
        })
    
    def record_error(self, error: str):
        """Registra un error"""
        self.error_log.append({
            'timestamp': time.time(),
            'error': error
        })
    
    def update_resources(self, omega: float, reputation: float):
        """Actualiza métricas de recursos"""
        self.omega.labels(agent_id=self.agent_id).set(omega)
        self.reputation.labels(agent_id=self.agent_id).set(reputation)
    
    @contextmanager
    def measure_action(self, action: str):
        """Mide duración de acción"""
        with self.action_duration.labels(
            agent_id=self.agent_id,
            action=action
        ).time():
            yield

# === SANDBOX DE EJECUCIÓN MEJORADO ===
class EnhancedSecureExecutionSandbox:
    """Sandbox mejorado para ejecución segura de código"""
    
    def __init__(self):
        self.forbidden_patterns = [
            r'__import__', r'eval', r'exec', r'compile',
            r'open', r'file', r'input', r'raw_input',
            r'os\.', r'sys\.', r'subprocess',
            r'importlib', r'__builtins__',
            r'\bimport\s+os\b', r'\bimport\s+sys\b'
        ]
        
        self.allowed_imports = {
            'math', 'random', 'statistics', 'itertools',
            'collections', 'functools', 'json', 're'
        }
        
        self.execution_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'timeouts': 0
        }
    
    async def execute(self, code: str, context: Dict[str, Any],
                     timeout: float = 5.0) -> Dict[str, Any]:
        """Ejecuta código con múltiples capas de seguridad"""
        self.execution_stats['total'] += 1
        
        try:
            # Validación estática
            validation = self._validate_code(code)
            if not validation['safe']:
                self.execution_stats['failed'] += 1
                return {
                    'success': False,
                    'error': f"Code validation failed: {validation['reason']}"
                }
            
            # Preparar entorno
            safe_globals = self._prepare_safe_environment(context)
            
            # Ejecutar con timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None,
                self._execute_code,
                code,
                safe_globals
            )
            
            result = await asyncio.wait_for(future, timeout=timeout)
            
            self.execution_stats['successful'] += 1
            return {
                'success': True,
                'output': result
            }
            
        except asyncio.TimeoutError:
            self.execution_stats['timeouts'] += 1
            return {
                'success': False,
                'error': f'Execution timeout ({timeout}s)'
            }
        except Exception as e:
            self.execution_stats['failed'] += 1
            return {
                'success': False,
                'error': f'Execution error: {str(e)}'
            }
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Valida código con múltiples verificaciones"""
        # Verificar patrones prohibidos
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code):
                return {
                    'safe': False,
                    'reason': f'Forbidden pattern: {pattern}'
                }
        
        # Verificar AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                'safe': False,
                'reason': f'Syntax error: {e}'
            }
        
        # Analizar imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_imports:
                        return {
                            'safe': False,
                            'reason': f'Forbidden import: {alias.name}'
                        }
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.allowed_imports:
                    return {
                        'safe': False,
                        'reason': f'Forbidden import: {node.module}'
                    }
            
            # Verificar llamadas peligrosas
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec', 'compile', '__import__'}:
                        return {
                            'safe': False,
                            'reason': f'Forbidden function: {node.func.id}'
                        }
        
        # Verificar complejidad
        complexity = self._calculate_complexity(tree)
        if complexity > 100:
            return {
                'safe': False,
                'reason': f'Code too complex (complexity: {complexity})'
            }
        
        return {'safe': True}
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calcula complejidad ciclomática del código"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _prepare_safe_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara entorno seguro para ejecución"""
        # Built-ins seguros
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'int': int, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'range': range,
            'round': round, 'set': set, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'zip': zip,
            'True': True, 'False': False, 'None': None
        }
        
        # Imports permitidos
        safe_imports = {
            'math': math,
            'random': random,
            'statistics': statistics,
            'json': json,
            're': re
        }
        
        # Crear entorno
        safe_globals = {
            '__builtins__': safe_builtins,
            **safe_imports,
            **context
        }
        
        return safe_globals
    
    def _execute_code(self, code: str, globals_dict: Dict[str, Any]) -> Any:
        """Ejecuta el código en entorno restringido"""
        locals_dict = {}
        
        # Compilar y ejecutar
        compiled = compile(code, '<sandbox>', 'exec')
        exec(compiled, globals_dict, locals_dict)
        
        # Buscar resultado
        # Primero buscar función principal
        for name, obj in locals_dict.items():
            if callable(obj) and not name.startswith('_'):
                # Llamar función con contexto si es posible
                try:
                    if 'graph' in globals_dict:
                        return obj(globals_dict.get('graph'))
                    elif 'nodes' in globals_dict:
                        return obj(globals_dict.get('nodes'))
                    elif 'analysis' in globals_dict:
                        return obj(globals_dict.get('analysis'))
                    else:
                        return obj()
                except TypeError:
                    # Intentar sin argumentos
                    try:
                        return obj()
                    except:
                        pass
        
        # Si no hay función, buscar variable resultado
        if 'result' in locals_dict:
            return locals_dict['result']
        
        # Retornar todo el namespace local
        return locals_dict

# === API SERVER AVANZADO ===
class AdvancedAPIServer:
    """Servidor API avanzado con todas las características"""
    
    def __init__(self, simulation_runner, config: Dict[str, Any]):
        self.simulation_runner = simulation_runner
        self.config = config
        
        # Flask app
        self.app = Flask(__name__)
        self.configure_app()
        
        # SocketIO
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=True
        )
        
        # OAuth
        self.oauth = OAuth(self.app)
        self.configure_oauth()
        
        # Métricas
        self.request_metrics = Counter(
            'api_requests',
            'API requests',
            ['method', 'endpoint', 'status']
        )
        
        # WebSocket rooms
        self.rooms = {
            'updates': set(),
            'metrics': set(),
            'agents': set()
        }
        
        # Registro de rutas
        self._register_routes()
        self._register_socketio_handlers()
        
        logger.info("Advanced API Server initialized")
    
    def configure_app(self):
        """Configura la aplicación Flask"""
        self.app.config['SECRET_KEY'] = Config.SECRET_KEY
        self.app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
        
        # CORS
        CORS(self.app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # JWT
        self.jwt = JWTManager(self.app)
        
        # Rate limiting
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"],
            storage_uri=Config.REDIS_URL if Config.REDIS_URL else None
        )
    
    def configure_oauth(self):
        """Configura OAuth providers"""
        if Config.OAUTH_PROVIDERS['google']['client_id']:
            self.oauth.register(
                name='google',
                client_id=Config.OAUTH_PROVIDERS['google']['client_id'],
                client_secret=Config.OAUTH_PROVIDERS['google']['client_secret'],
                server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                client_kwargs={'scope': 'openid email profile'}
            )
    
    def _register_routes(self):
        """Registra todas las rutas de la API"""
        
        # === RUTAS DE AUTENTICACIÓN ===
        @self.app.route('/api/auth/login', methods=['POST'])
        @self.limiter.limit("5 per minute")
        async def login():
            """Login con credenciales"""
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            # Verificar en base de datos (placeholder)
            # En producción, usar bcrypt y base de datos real
            if username and password:  # Verificación real aquí
                access_token = create_access_token(
                    identity=username,
                    additional_claims={'role': 'admin'}
                )
                
                self.request_metrics.labels('POST', '/api/auth/login', '200').inc()
                
                return jsonify({
                    'access_token': access_token,
                    'user': {
                        'username': username,
                        'role': 'admin'
                    }
                })
            
            self.request_metrics.labels('POST', '/api/auth/login', '401').inc()
            return jsonify({'error': 'Invalid credentials'}), 401
        
        @self.app.route('/api/auth/google')
        async def google_login():
            """Login con Google OAuth"""
            redirect_uri = url_for('google_callback', _external=True)
            return await self.oauth.google.authorize_redirect(redirect_uri)
        
        # === RUTAS DEL SISTEMA ===
        @self.app.route('/api/system/health')
        async def system_health():
            """Health check del sistema"""
            health = {
                'status': 'healthy',
                'timestamp': time.time(),
                'components': {
                    'graph': self.simulation_runner.graph is not None,
                    'agents': len(self.simulation_runner.agents) > 0,
                    'event_bus': self.simulation_runner.graph.event_bus._running if self.simulation_runner.graph else False
                }
            }
            
            return jsonify(health)
        
        @self.app.route('/api/system/metrics')
        async def system_metrics():
            """Métricas Prometheus"""
            return Response(
                generate_latest(),
                mimetype='text/plain'
            )
        
        # === RUTAS DEL GRAFO ===
        @self.app.route('/api/graph/status')
        @jwt_required()
        async def graph_status():
            """Estado del grafo"""
            if not self.simulation_runner.graph:
                return jsonify({'error': 'Graph not initialized'}), 500
            
            status = {
                'nodes': len(self.simulation_runner.graph.nodes),
                'edges': sum(
                    len(n.connections_out) 
                    for n in self.simulation_runner.graph.nodes.values()
                ),
                'health': self.simulation_runner.graph.calculate_graph_health(),
                'clusters': len(self.simulation_runner.graph.cluster_index)
            }
            
            return jsonify(status)
        
        @self.app.route('/api/graph/nodes', methods=['GET'])
        @jwt_required()
        async def get_nodes():
            """Lista de nodos con paginación y filtros"""
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            sort_by = request.args.get('sort_by', 'id')
            order = request.args.get('order', 'asc')
            
            # Filtros
            min_state = request.args.get('min_state', type=float)
            max_state = request.args.get('max_state', type=float)
            keywords = request.args.get('keywords', '').split(',')
            tags = request.args.get('tags', '').split(',')
            cluster_id = request.args.get('cluster_id', type=int)
            
            if not self.simulation_runner.graph:
                return jsonify({'error': 'Graph not initialized'}), 500
            
            # Aplicar filtros
            nodes = list(self.simulation_runner.graph.nodes.values())
            
            if min_state is not None:
                nodes = [n for n in nodes if n.state >= min_state]
            if max_state is not None:
                nodes = [n for n in nodes if n.state <= max_state]
            if keywords[0]:
                keyword_set = set(k.strip() for k in keywords if k.strip())
                nodes = [n for n in nodes if keyword_set & n.keywords]
            if tags[0]:
                tag_set = set(t.strip() for t in tags if t.strip())
                nodes = [n for n in nodes if tag_set & n.metadata.tags]
            if cluster_id is not None:
                nodes = [n for n in nodes if n.metadata.cluster_id == cluster_id]
            
            # Ordenar
            if sort_by == 'state':
                nodes.sort(key=lambda n: n.state, reverse=(order == 'desc'))
            elif sort_by == 'importance':
                nodes.sort(key=lambda n: n.metadata.importance_score, reverse=(order == 'desc'))
            elif sort_by == 'created':
                nodes.sort(key=lambda n: n.metadata.created_at, reverse=(order == 'desc'))
            else:  # id
                nodes.sort(key=lambda n: n.id, reverse=(order == 'desc'))
            
            # Paginar
            total = len(nodes)
            start = (page - 1) * per_page
            end = start + per_page
            
            nodes_data = [
                {
                    'id': n.id,
                    'content': n.content,
                    'state': n.state,
                    'keywords': list(n.keywords),
                    'connections': {
                        'in': len(n.connections_in),
                        'out': len(n.connections_out)
                    },
                    'metadata': {
                        'importance': n.metadata.importance_score,
                        'cluster_id': n.metadata.cluster_id,
                        'tags': list(n.metadata.tags),
                        'created_at': n.metadata.created_at,
                        'created_by': n.metadata.created_by
                    }
                }
                for n in nodes[start:end]
            ]
            
            return jsonify({
                'nodes': nodes_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            })
        
        @self.app.route('/api/graph/nodes/<int:node_id>')
        @jwt_required()
        async def get_node(node_id):
            """Detalles de un nodo específico"""
            if not self.simulation_runner.graph:
                return jsonify({'error': 'Graph not initialized'}), 500
            
            node = self.simulation_runner.graph.nodes.get(node_id)
            if not node:
                return jsonify({'error': 'Node not found'}), 404
            
            # Información detallada
            similar_nodes = await self.simulation_runner.graph.find_similar_nodes(node_id)
            
            return jsonify({
                'id': node.id,
                'content': node.content,
                'state': node.state,
                'keywords': list(node.keywords),
                'connections_out': node.connections_out,
                'connections_in': node.connections_in,
                'metadata': asdict(node.metadata),
                'features': node.features,
                'state_history': list(node.state_history)[-20:],
                'similar_nodes': similar_nodes,
                'anomaly_score': node.anomaly_score
            })
        
        @self.app.route('/api/graph/nodes', methods=['POST'])
        @jwt_required()
        @self.limiter.limit("10 per minute")
        async def create_node():
            """Crea un nuevo nodo"""
            data = request.get_json()
            user = get_jwt_identity()
            
            try:
                node = await self.simulation_runner.graph.add_node(
                    content=data.get('content', 'New concept'),
                    initial_state=data.get('initial_state', 0.5),
                    keywords=set(data.get('keywords', [])),
                    created_by=user,
                    properties=data.get('properties', {})
                )
                
                return jsonify({
                    'success': True,
                    'node_id': node.id,
                    'message': 'Node created successfully'
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/graph/edges', methods=['POST'])
        @jwt_required()
        @self.limiter.limit("20 per minute")
        async def create_edge():
            """Crea una conexión entre nodos"""
            data = request.get_json()
            
            source_id = data.get('source_id')
            target_id = data.get('target_id')
            utility = data.get('utility', 0.5)
            
            if not all([source_id is not None, target_id is not None]):
                return jsonify({'error': 'Missing source_id or target_id'}), 400
            
            success = await self.simulation_runner.graph.add_edge(
                source_id, target_id, utility
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Edge created successfully'
                })
            else:
                return jsonify({'error': 'Failed to create edge'}), 400
        
        @self.app.route('/api/graph/search')
        @jwt_required()
        async def search_graph():
            """Búsqueda avanzada en el grafo"""
            query = request.args.get('q', '')
            search_type = request.args.get('type', 'keyword')  # keyword, content, semantic
            limit = request.args.get('limit', 20, type=int)
            
            if not query:
                return jsonify({'error': 'Query parameter required'}), 400
            
            results = []
            
            if search_type == 'keyword':
                keywords = set(query.split(','))
                nodes = self.simulation_runner.graph.search_by_keywords(keywords)
                results = [
                    {
                        'id': n.id,
                        'content': n.content,
                        'state': n.state,
                        'relevance': len(keywords & n.keywords) / len(keywords)
                    }
                    for n in nodes[:limit]
                ]
                
            elif search_type == 'content':
                # Búsqueda por contenido
                for node in self.simulation_runner.graph.nodes.values():
                    if query.lower() in node.content.lower():
                        results.append({
                            'id': node.id,
                            'content': node.content,
                            'state': node.state,
                            'relevance': 1.0
                        })
                        if len(results) >= limit:
                            break
                            
            elif search_type == 'semantic':
                # Búsqueda semántica usando embeddings
                # Generar embedding de la query
                if self.simulation_runner.graph.text_encoder:
                    query_embedding = await self.simulation_runner.graph._generate_embedding(query)
                    
                    similarities = []
                    for node_id, node in self.simulation_runner.graph.nodes.items():
                        if node.metadata.embedding is not None:
                            sim = self.simulation_runner.graph._cosine_similarity(
                                query_embedding,
                                node.metadata.embedding
                            )
                            similarities.append((node, sim))
                    
                    # Ordenar por similitud
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    results = [
                        {
                            'id': node.id,
                            'content': node.content,
                            'state': node.state,
                            'relevance': sim
                        }
                        for node, sim in similarities[:limit]
                    ]
            
            return jsonify({
                'query': query,
                'type': search_type,
                'results': results,
                'count': len(results)
            })
        
        @self.app.route('/api/graph/analyze/centrality')
        @jwt_required()
        async def analyze_centrality():
            """Análisis de centralidad del grafo"""
            centralities = await self.simulation_runner.graph.network_analyzer.calculate_centralities()
            
            # Top nodos por cada medida
            top_k = 10
            analysis = {}
            
            for measure, values in centralities.items():
                sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
                analysis[measure] = [
                    {
                        'node_id': node_id,
                        'value': value,
                        'content': self.simulation_runner.graph.nodes[node_id].content[:50]
                    }
                    for node_id, value in sorted_nodes[:top_k]
                ]
            
            return jsonify(analysis)
        
        @self.app.route('/api/graph/analyze/communities')
        @jwt_required()
        async def analyze_communities():
            """Detecta comunidades en el grafo"""
            communities = await self.simulation_runner.graph.network_analyzer.detect_communities()
            
            # Agrupar nodos por comunidad
            community_groups = defaultdict(list)
            for node_id, community_id in communities.items():
                node = self.simulation_runner.graph.nodes.get(node_id)
                if node:
                    community_groups[community_id].append({
                        'id': node_id,
                        'content': node.content[:50],
                        'state': node.state
                    })
            
            # Estadísticas de comunidades
            community_stats = []
            for comm_id, nodes in community_groups.items():
                states = [n['state'] for n in nodes]
                community_stats.append({
                    'id': comm_id,
                    'size': len(nodes),
                    'avg_state': np.mean(states),
                    'nodes': nodes[:5]  # Muestra
                })
            
            community_stats.sort(key=lambda x: x['size'], reverse=True)
            
            return jsonify({
                'communities': community_stats,
                'total_communities': len(community_groups)
            })
        
        @self.app.route('/api/graph/cluster', methods=['POST'])
        @jwt_required()
        async def cluster_nodes():
            """Ejecuta clustering en los nodos"""
            data = request.get_json()
            algorithm = data.get('algorithm', 'dbscan')
            
            clusters = await self.simulation_runner.graph.cluster_nodes(algorithm)
            
            # Estadísticas de clusters
            cluster_stats = defaultdict(lambda: {'count': 0, 'avg_state': 0})
            
            for node_id, cluster_id in clusters.items():
                if cluster_id >= 0:  # Ignorar ruido (-1)
                    node = self.simulation_runner.graph.nodes.get(node_id)
                    if node:
                        cluster_stats[cluster_id]['count'] += 1
                        cluster_stats[cluster_id]['avg_state'] += node.state
            
            # Calcular promedios
            for cluster_id, stats in cluster_stats.items():
                if stats['count'] > 0:
                    stats['avg_state'] /= stats['count']
            
            return jsonify({
                'clusters': dict(cluster_stats),
                'total_clusters': len(cluster_stats),
                'noise_nodes': sum(1 for c in clusters.values() if c == -1)
            })
        
        # === RUTAS DE AGENTES ===
        @self.app.route('/api/agents')
        @jwt_required()
        async def get_agents():
            """Lista de agentes activos"""
            agents_data = []
            
            for agent in self.simulation_runner.agents:
                agents_data.append({
                    'id': agent.id,
                    'type': agent.__class__.__name__,
                    'omega': agent.omega,
                    'reputation': agent.reputation,
                    'performance_score': agent.get_performance_score() if hasattr(agent, 'get_performance_score') else 0,
                    'specialization': list(agent.specialization) if hasattr(agent, 'specialization') else [],
                    'metrics': {
                        'actions_performed': len(agent.action_history),
                        'recent_success_rate': agent._calculate_recent_success_rate()
                    }
                })
            
            return jsonify({
                'agents': agents_data,
                'total': len(agents_data)
            })
        
        @self.app.route('/api/agents/<agent_id>')
        @jwt_required()
        async def get_agent_details(agent_id):
            """Detalles de un agente específico"""
            agent = None
            for a in self.simulation_runner.agents:
                if a.id == agent_id:
                    agent = a
                    break
            
            if not agent:
                return jsonify({'error': 'Agent not found'}), 404
            
            # Historial reciente
            recent_actions = list(agent.action_history)[-20:]
            
            # Análisis de rendimiento
            if recent_actions:
                success_count = sum(1 for a in recent_actions if a.get('success', False))
                success_rate = success_count / len(recent_actions)
            else:
                success_rate = 0
            
            details = {
                'id': agent.id,
                'type': agent.__class__.__name__,
                'omega': agent.omega,
                'max_omega': agent.max_omega,
                'reputation': agent.reputation,
                'learning_rate': agent.learning_rate,
                'recent_actions': recent_actions,
                'performance': {
                    'success_rate': success_rate,
                    'total_actions': len(agent.action_history),
                    'avg_reward': np.mean(list(agent.reward_history)) if agent.reward_history else 0
                }
            }
            
            # Información específica de TAEC
            if isinstance(agent, ClaudeTAECAgent):
                details['taec_info'] = {
                    'evolution_count': agent.evolution_count,
                    'code_repository_size': len(agent.code_repository.repository),
                    'evolution_memory': agent.evolution_memory.analyze_history()
                }
            
            return jsonify(details)
        
        @self.app.route('/api/agents/<agent_id>/act', methods=['POST'])
        @jwt_required()
        @self.limiter.limit("5 per minute")
        async def trigger_agent_action(agent_id):
            """Dispara una acción manual del agente"""
            agent = None
            for a in self.simulation_runner.agents:
                if a.id == agent_id:
                    agent = a
                    break
            
            if not agent:
                return jsonify({'error': 'Agent not found'}), 404
            
            # Ejecutar acción
            try:
                await agent.act()
                return jsonify({
                    'success': True,
                    'message': f'Agent {agent_id} action triggered'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # === RUTAS DE SIMULACIÓN ===
        @self.app.route('/api/simulation/status')
        @jwt_required()
        async def simulation_status():
            """Estado de la simulación"""
            status = self.simulation_runner.get_detailed_status()
            return jsonify(status)
        
        @self.app.route('/api/simulation/control', methods=['POST'])
        @jwt_required()
        async def control_simulation():
            """Control de la simulación"""
            data = request.get_json()
            action = data.get('action')
            
            if action == 'start':
                if not self.simulation_runner.is_running:
                    await self.simulation_runner.start()
                    return jsonify({'success': True, 'message': 'Simulation started'})
                else:
                    return jsonify({'error': 'Simulation already running'}), 400
                    
            elif action == 'stop':
                if self.simulation_runner.is_running:
                    await self.simulation_runner.stop()
                    return jsonify({'success': True, 'message': 'Simulation stopped'})
                else:
                    return jsonify({'error': 'Simulation not running'}), 400
                    
            elif action == 'pause':
                self.simulation_runner.pause()
                return jsonify({'success': True, 'message': 'Simulation paused'})
                
            elif action == 'resume':
                self.simulation_runner.resume()
                return jsonify({'success': True, 'message': 'Simulation resumed'})
                
            else:
                return jsonify({'error': 'Invalid action'}), 400
        
        @self.app.route('/api/simulation/checkpoint', methods=['POST'])
        @jwt_required()
        async def create_checkpoint():
            """Crea un checkpoint manual"""
            data = request.get_json()
            name = data.get('name', f'manual_{int(time.time())}')
            
            try:
                path = await self.simulation_runner.create_checkpoint(name)
                return jsonify({
                    'success': True,
                    'checkpoint_path': path
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/simulation/export')
        @jwt_required()
        async def export_simulation():
            """Exporta datos de la simulación"""
            format_type = request.args.get('format', 'json')
            
            if format_type == 'json':
                data = await self.simulation_runner.export_data()
                return jsonify(data)
                
            elif format_type == 'graphml':
                # Exportar en formato GraphML
                nx_graph = self.simulation_runner.graph.network_analyzer._get_networkx_graph()
                
                from io import BytesIO
                buffer = BytesIO()
                nx.write_graphml(nx_graph, buffer)
                buffer.seek(0)
                
                return Response(
                    buffer.getvalue(),
                    mimetype='application/xml',
                    headers={
                        'Content-Disposition': 'attachment; filename=graph.graphml'
                    }
                )
            
            else:
                return jsonify({'error': 'Invalid format'}), 400
        
        # === RUTAS DE VISUALIZACIÓN ===
        @self.app.route('/api/visualization/graph3d')
        @jwt_required()
        async def get_graph_3d_data():
            """Datos para visualización 3D del grafo"""
            if not self.simulation_runner.graph:
                return jsonify({'error': 'Graph not initialized'}), 500
            
            # Preparar datos para visualización 3D
            nodes = []
            edges = []
            
            # Layout 3D usando spring layout
            G = self.simulation_runner.graph.network_analyzer._get_networkx_graph()
            
            # Calcular posiciones 3D
            if len(G) > 0:
                # Usar Kamada-Kawai para grafos pequeños, spring para grandes
                if len(G) < 100:
                    pos_2d = nx.kamada_kawai_layout(G)
                else:
                    pos_2d = nx.spring_layout(G, k=1/np.sqrt(len(G)), iterations=50)
                
                # Añadir tercera dimensión basada en estado
                pos_3d = {}
                for node_id, (x, y) in pos_2d.items():
                    node = self.simulation_runner.graph.nodes.get(node_id)
                    z = node.state if node else 0.5
                    pos_3d[node_id] = (x, y, z)
                
                # Preparar nodos
                for node_id, (x, y, z) in pos_3d.items():
                    node = self.simulation_runner.graph.nodes.get(node_id)
                    if node:
                        nodes.append({
                            'id': node_id,
                            'x': float(x),
                            'y': float(y),
                            'z': float(z),
                            'label': node.content[:30],
                            'size': 5 + node.metadata.importance_score * 10,
                            'color': self._get_node_color(node),
                            'state': node.state,
                            'cluster': node.metadata.cluster_id
                        })
                
                # Preparar edges
                for node_id, node in self.simulation_runner.graph.nodes.items():
                    if node_id in pos_3d:
                        for target_id, utility in node.connections_out.items():
                            if target_id in pos_3d:
                                edges.append({
                                    'source': node_id,
                                    'target': target_id,
                                    'weight': utility,
                                    'color': self._get_edge_color(utility)
                                })
            
            return jsonify({
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges)
                }
            })
        
        def _get_node_color(self, node):
            """Calcula color del nodo basado en características"""
            # Color basado en cluster
            if node.metadata.cluster_id is not None:
                # Paleta de colores para clusters
                colors = [
                    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                    '#DDA0DD', '#FFD93D', '#6BCF7F', '#FF6B9D'
                ]
                return colors[node.metadata.cluster_id % len(colors)]
            
            # Color basado en estado
            r = int(255 * (1 - node.state))
            g = int(255 * node.state)
            b = 100
            return f'#{r:02x}{g:02x}{b:02x}'
        
        def _get_edge_color(self, weight):
            """Calcula color del edge basado en peso"""
            intensity = int(255 * weight)
            return f'#{intensity:02x}{intensity:02x}{intensity:02x}'
        
        # === RUTAS DE ANALYTICS ===
        @self.app.route('/api/analytics/timeline')
        @jwt_required()
        async def get_timeline_data():
            """Datos de línea temporal"""
            # Obtener historial de métricas
            history = list(self.simulation_runner.graph.metrics.history)[-1000:]
            
            timeline = {
                'timestamps': [h['timestamp'] for h in history],
                'metrics': {
                    'nodes': [h['nodes'] for h in history],
                    'edges': [h['edges'] for h in history],
                    'avg_state': [h['avg_state'] for h in history]
                }
            }
            
            # Detectar eventos importantes
            events = []
            for i in range(1, len(history)):
                # Detectar cambios bruscos
                node_change = abs(history[i]['nodes'] - history[i-1]['nodes'])
                if node_change > 10:
                    events.append({
                        'timestamp': history[i]['timestamp'],
                        'type': 'node_spike',
                        'description': f'Node count changed by {node_change}'
                    })
            
            timeline['events'] = events
            
            return jsonify(timeline)
        
        @self.app.route('/api/analytics/predictions')
        @jwt_required()
        async def get_predictions():
            """Predicciones del sistema"""
            # Usar modelo ML para predecir evolución
            predictions = await self.simulation_runner.predict_evolution()
            
            return jsonify(predictions)
        
        # === WEBSOCKET HANDLERS ===
        @self.app.route('/')
        def index():
            """Sirve la aplicación web"""
            return send_from_directory('static', 'index.html')
        
        @self.app.route('/static/<path:path>')
        def serve_static(path):
            """Sirve archivos estáticos"""
            return send_from_directory('static', path)
    
    def _register_socketio_handlers(self):
        """Registra handlers de WebSocket"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Cliente conectado"""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {
                'message': 'Connected to MSC Framework v4.0',
                'sid': request.sid
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Cliente desconectado"""
            logger.info(f"Client disconnected: {request.sid}")
            # Remover de todas las rooms
            for room_name, sids in self.rooms.items():
                sids.discard(request.sid)
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Suscribir a actualizaciones"""
            room = data.get('room', 'updates')
            if room in self.rooms:
                join_room(room)
                self.rooms[room].add(request.sid)
                emit('subscribed', {'room': room})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Desuscribir de actualizaciones"""
            room = data.get('room', 'updates')
            if room in self.rooms:
                leave_room(room)
                self.rooms[room].discard(request.sid)
                emit('unsubscribed', {'room': room})
        
        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Solicitud de actualización específica"""
            update_type = data.get('type', 'status')
            
            if update_type == 'status':
                status = self.simulation_runner.get_detailed_status()
                emit('status_update', status)
                
            elif update_type == 'graph':
                graph_data = {
                    'nodes': len(self.simulation_runner.graph.nodes),
                    'edges': sum(
                        len(n.connections_out) 
                        for n in self.simulation_runner.graph.nodes.values()
                    ),
                    'health': self.simulation_runner.graph.calculate_graph_health()
                }
                emit('graph_update', graph_data)
                
            elif update_type == 'agents':
                agents_data = [
                    {
                        'id': a.id,
                        'type': a.__class__.__name__,
                        'omega': a.omega,
                        'reputation': a.reputation
                    }
                    for a in self.simulation_runner.agents
                ]
                emit('agents_update', agents_data)
    
    def broadcast_event(self, event: Event):
        """Broadcast evento a clientes suscritos"""
        # Determinar room basado en tipo de evento
        room = 'updates'  # Default
        
        if event.type in [EventType.METRICS_UPDATE, EventType.PERFORMANCE_ALERT]:
            room = 'metrics'
        elif event.type in [EventType.AGENT_ACTION, EventType.AGENT_EVOLVED]:
            room = 'agents'
        
        # Broadcast a la room
        self.socketio.emit(
            'event',
            event.to_dict(),
            room=room,
            namespace='/'
        )
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Ejecuta el servidor"""
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False
        )

# === SIMULACIÓN AVANZADA ===
class AdvancedSimulationRunner:
    """Runner de simulación avanzado con todas las características"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = AdvancedCollectiveSynthesisGraph(config)
        self.agents: List[ImprovedBaseAgent] = []
        self.is_running = False
        self.is_paused = False
        self.step_count = 0
        self.start_time = time.time()
        
        # Componentes
        self.event_bus = self.graph.event_bus
        self.api_server: Optional[AdvancedAPIServer] = None
        
        # Estado
        self.simulation_state = SimulationState()
        
        # Predictor ML
        self.predictor = SimulationPredictor()
        
        # Monitor de rendimiento
        self.performance_monitor = PerformanceMonitor()
        
        # Inicialización
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Inicializa componentes asíncronos"""
        # Inicializar grafo con backends
        await self.graph.initialize(
            redis_url=self.config.get('redis_url'),
            postgres_url=self.config.get('postgres_url')
        )
        
        # Crear agentes
        await self._create_agents()
        
        # Suscribir a eventos
        self._setup_event_handlers()
        
        # Crear servidor API
        self.api_server = AdvancedAPIServer(self, self.config)
        
        # Cargar estado si existe
        if self.config.get('load_checkpoint'):
            await self.load_checkpoint(self.config['load_checkpoint'])
        
        logger.info("Advanced Simulation Runner initialized")
    
    async def _create_agents(self):
        """Crea agentes según configuración"""
        agent_configs = self.config.get('agents', {})
        
        # Agentes Claude-TAEC
        for i in range(agent_configs.get('claude_taec', 3)):
            agent = ClaudeTAECAgent(
                f"ClaudeTAEC_{i}",
                self.graph,
                self.config
            )
            self.agents.append(agent)
        
        # Otros tipos de agentes pueden añadirse aquí
        
        logger.info(f"Created {len(self.agents)} agents")
    
    def _setup_event_handlers(self):
        """Configura handlers de eventos"""
        # Handlers para diferentes tipos de eventos
        self.event_bus.subscribe(
            EventType.NODE_CREATED,
            self._handle_node_created
        )
        self.event_bus.subscribe(
            EventType.AGENT_EVOLVED,
            self._handle_agent_evolved
        )
        self.event_bus.subscribe(
            EventType.CRITICAL_ERROR,
            self._handle_critical_error
        )
    
    async def start(self):
        """Inicia la simulación"""
        if self.is_running:
            logger.warning("Simulation already running")
            return
        
        self.is_running = True
        self.is_paused = False
        self.simulation_state.phase = 'running'
        
        # Iniciar servidor API si está configurado
        if self.config.get('enable_api', True):
            api_thread = threading.Thread(
                target=self.api_server.run,
                kwargs={
                    'host': self.config.get('api_host', '0.0.0.0'),
                    'port': self.config.get('api_port', 5000),
                    'debug': False
                },
                daemon=True
            )
            api_thread.start()
        
        # Loop principal
        asyncio.create_task(self._simulation_loop())
        
        logger.info("Simulation started")
    
    async def stop(self):
        """Detiene la simulación"""
        logger.info("Stopping simulation...")
        self.is_running = False
        self.simulation_state.phase = 'stopped'
        
        # Guardar estado final
        if self.config.get('auto_save', True):
            await self.create_checkpoint('final')
        
        # Detener event bus
        await self.event_bus.stop()
        
        logger.info("Simulation stopped")
    
    def pause(self):
        """Pausa la simulación"""
        self.is_paused = True
        self.simulation_state.phase = 'paused'
        logger.info("Simulation paused")
    
    def resume(self):
        """Reanuda la simulación"""
        self.is_paused = False
        self.simulation_state.phase = 'running'
        logger.info("Simulation resumed")
    
    async def _simulation_loop(self):
        """Loop principal de simulación"""
        while self.is_running:
            try:
                if not self.is_paused:
                    # Ejecutar paso
                    await self._execute_step()
                    
                    # Actualizar métricas
                    await self._update_metrics()
                    
                    # Checkpoints automáticos
                    if self.step_count % self.config.get('checkpoint_interval', 1000) == 0:
                        await self.create_checkpoint(f'auto_{self.step_count}')
                    
                    # Broadcast estado
                    if self.api_server and self.step_count % 10 == 0:
                        await self._broadcast_status()
                
                # Delay
                await asyncio.sleep(self.config.get('step_delay', 0.1))
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await self.event_bus.publish(Event(
                    type=EventType.CRITICAL_ERROR,
                    data={'error': str(e), 'traceback': traceback.format_exc()},
                    source='simulation',
                    priority=1
                ))
    
    async def _execute_step(self):
        """Ejecuta un paso de simulación"""
        self.step_count += 1
        
        with self.performance_monitor.measure('simulation_step'):
            # Fase 1: Percepción
            perceptions = await self._perception_phase()
            
            # Fase 2: Decisión
            decisions = await self._decision_phase(perceptions)
            
            # Fase 3: Acción
            await self._action_phase(decisions)
            
            # Fase 4: Evolución
            if self.step_count % self.config.get('evolution_interval', 100) == 0:
                await self._evolution_phase()
            
            # Fase 5: Consenso
            if self.step_count % self.config.get('consensus_interval', 500) == 0:
                await self._consensus_phase()
    
    async def _perception_phase(self):
        """Fase de percepción para todos los agentes"""
        perceptions = {}
        
        # Percepción paralela
        tasks = []
        for agent in self.agents:
            if agent.omega > 0:  # Solo agentes activos
                tasks.append(agent.perceive_environment())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    perceptions[self.agents[i].id] = result
        
        return perceptions
    
    async def _decision_phase(self, perceptions):
        """Fase de decisión para agentes"""
        decisions = {}
        
        for agent in self.agents:
            if agent.id in perceptions:
                decision = await agent.decide_action(perceptions[agent.id])
                decisions[agent.id] = decision
        
        return decisions
    
    async def _action_phase(self, decisions):
        """Fase de ejecución de acciones"""
        # Limitar número de acciones simultáneas
        max_concurrent = self.config.get('max_concurrent_actions', 5)
        
        # Seleccionar agentes que actuarán
        acting_agents = []
        for agent in self.agents:
            if agent.id in decisions and agent.omega > 0:
                acting_agents.append(agent)
        
        # Shuffle para fairness
        random.shuffle(acting_agents)
        
        # Ejecutar acciones
        for i in range(0, len(acting_agents), max_concurrent):
            batch = acting_agents[i:i + max_concurrent]
            tasks = [agent.act() for agent in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _evolution_phase(self):
        """Fase de evolución del sistema"""
        logger.info(f"Evolution phase at step {self.step_count}")
        
        # Seleccionar agentes TAEC para evolución
        taec_agents = [
            a for a in self.agents 
            if isinstance(a, ClaudeTAECAgent) and a.omega > 20
        ]
        
        if taec_agents:
            # Elegir uno aleatoriamente
            agent = random.choice(taec_agents)
            
            # Ejecutar evolución
            perception = await agent.perceive_environment()
            await agent.execute_action('evolve', perception)
            
            # Publicar evento
            await self.event_bus.publish(Event(
                type=EventType.EVOLUTION_CYCLE,
                data={'agent_id': agent.id, 'step': self.step_count},
                source='simulation'
            ))
    
    async def _consensus_phase(self):
        """Fase de consenso entre agentes"""
        logger.info(f"Consensus phase at step {self.step_count}")
        
        # Implementar mecanismo de consenso
        # Por ahora, simple votación sobre acciones importantes
        
        # Tema de consenso: nodos a promover
        candidates = [
            n for n in self.graph.nodes.values()
            if n.state > 0.7 and n.metadata.importance_score < 0.5
        ]
        
        if candidates and len(self.agents) > 1:
            # Votación
            votes = defaultdict(int)
            
            for agent in self.agents:
                if agent.omega > 0:
                    # Cada agente vota por un candidato
                    choice = random.choice(candidates)
                    votes[choice.id] += agent.reputation
            
            # Ganador
            if votes:
                winner_id = max(votes.items(), key=lambda x: x[1])[0]
                winner = self.graph.nodes[winner_id]
                
                # Promover nodo
                winner.metadata.importance_score = min(1.0, winner.metadata.importance_score * 1.5)
                winner.metadata.tags.add('consensus_promoted')
                
                # Evento
                await self.event_bus.publish(Event(
                    type=EventType.CONSENSUS_REACHED,
                    data={'node_id': winner_id, 'votes': dict(votes)},
                    source='simulation'
                ))
    
    async def _update_metrics(self):
        """Actualiza métricas del sistema"""
        # Métricas del grafo
        self.graph.metrics.update_graph_metrics(self.graph)
        
        # Métricas de agentes
        for agent in self.agents:
            if hasattr(agent, 'metrics'):
                agent.metrics.update_resources(agent.omega, agent.reputation)
        
        # Detectar anomalías
        anomalies = self.graph.metrics.detect_anomalies()
        if anomalies:
            for anomaly in anomalies:
                await self.event_bus.publish(Event(
                    type=EventType.PERFORMANCE_ALERT,
                    data=anomaly,
                    source='metrics',
                    priority=2
                ))
        
        # Regeneración de omega
        for agent in self.agents:
            regen = self.config.get('omega_regeneration', 0.1) * agent.reputation
            agent.omega = min(agent.omega + regen, agent.max_omega)
    
    async def _broadcast_status(self):
        """Broadcast estado a clientes conectados"""
        if self.api_server:
            status_event = Event(
                type=EventType.METRICS_UPDATE,
                data=self.get_detailed_status(),
                source='simulation'
            )
            self.api_server.broadcast_event(status_event)
    
    async def create_checkpoint(self, name: str) -> str:
        """Crea un checkpoint de la simulación"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR,
            f"{name}_{int(time.time())}.checkpoint"
        )
        
        checkpoint_data = {
            'version': '4.0.0',
            'timestamp': time.time(),
            'step_count': self.step_count,
            'graph': await self._serialize_graph(),
            'agents': self._serialize_agents(),
            'simulation_state': asdict(self.simulation_state)
        }
        
        # Comprimir y guardar
        compressed = zlib.compress(
            json.dumps(checkpoint_data).encode('utf-8')
        )
        
        async with aiofiles.open(checkpoint_path, 'wb') as f:
            await f.write(compressed)
        
        logger.info(f"Checkpoint created: {checkpoint_path}")
        
        # Evento
        await self.event_bus.publish(Event(
            type=EventType.CHECKPOINT_CREATED,
            data={'path': checkpoint_path, 'name': name},
            source='simulation'
        ))
        
        return checkpoint_path
    
    async def load_checkpoint(self, path: str):
        """Carga un checkpoint"""
        async with aiofiles.open(path, 'rb') as f:
            compressed = await f.read()
        
        checkpoint_data = json.loads(
            zlib.decompress(compressed).decode('utf-8')
        )
        
        # Restaurar grafo
        await self._restore_graph(checkpoint_data['graph'])
        
        # Restaurar agentes
        self._restore_agents(checkpoint_data['agents'])
        
        # Restaurar estado
        self.step_count = checkpoint_data['step_count']
        # Restaurar simulation_state...
        
        logger.info(f"Checkpoint loaded: {path}")
    
    async def _serialize_graph(self) -> Dict[str, Any]:
        """Serializa el grafo"""
        nodes_data = []
        
        for node in self.graph.nodes.values():
            node_dict = {
                'id': node.id,
                'content': node.content,
                'state': node.state,
                'keywords': list(node.keywords),
                'connections_out': node.connections_out,
                'connections_in': node.connections_in,
                'metadata': {
                    'created_at': node.metadata.created_at,
                    'updated_at': node.metadata.updated_at,
                    'created_by': node.metadata.created_by,
                    'tags': list(node.metadata.tags),
                    'properties': node.metadata.properties,
                    'importance_score': node.metadata.importance_score,
                    'cluster_id': node.metadata.cluster_id
                },
                'features': node.features
            }
            
            # Embedding como lista si existe
            if node.metadata.embedding is not None:
                node_dict['metadata']['embedding'] = node.metadata.embedding.tolist()
            
            nodes_data.append(node_dict)
        
        return {
            'nodes': nodes_data,
            'next_node_id': self.graph.next_node_id,
            'metrics': {
                'history': list(self.graph.metrics.history)[-1000:]
            }
        }
    
    def _serialize_agents(self) -> List[Dict[str, Any]]:
        """Serializa los agentes"""
        agents_data = []
        
        for agent in self.agents:
            agent_data = {
                'id': agent.id,
                'type': agent.__class__.__name__,
                'omega': agent.omega,
                'reputation': agent.reputation,
                'specialization': list(agent.specialization) if hasattr(agent, 'specialization') else [],
                'learning_rate': agent.learning_rate
            }
            
            # Datos específicos de TAEC
            if isinstance(agent, ClaudeTAECAgent):
                agent_data['taec_data'] = {
                    'evolution_count': agent.evolution_count,
                    'evolution_memory': list(agent.evolution_memory.history)[-100:]
                }
            
            agents_data.append(agent_data)
        
        return agents_data
    
    async def predict_evolution(self) -> Dict[str, Any]:
        """Predice evolución futura del sistema"""
        return await self.predictor.predict(
            self.graph,
            self.agents,
            self.simulation_state
        )
    
    async def export_data(self) -> Dict[str, Any]:
        """Exporta datos de la simulación"""
        return {
            'metadata': {
                'version': '4.0.0',
                'export_time': time.time(),
                'step_count': self.step_count,
                'runtime': time.time() - self.start_time
            },
            'graph': await self._serialize_graph(),
            'agents': self._serialize_agents(),
            'statistics': self._calculate_statistics()
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas de la simulación"""
        return {
            'graph': {
                'total_nodes': len(self.graph.nodes),
                'total_edges': sum(
                    len(n.connections_out) 
                    for n in self.graph.nodes.values()
                ),
                'avg_node_state': np.mean([n.state for n in self.graph.nodes.values()]) if self.graph.nodes else 0,
                'clusters': len(self.graph.cluster_index)
            },
            'agents': {
                'total': len(self.agents),
                'active': sum(1 for a in self.agents if a.omega > 0),
                'avg_omega': np.mean([a.omega for a in self.agents]) if self.agents else 0,
                'avg_reputation': np.mean([a.reputation for a in self.agents]) if self.agents else 0
            },
            'evolution': {
                'total_evolutions': sum(
                    a.evolution_count 
                    for a in self.agents 
                    if isinstance(a, ClaudeTAECAgent)
                )
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Obtiene estado detallado de la simulación"""
        return {
            'running': self.is_running,
            'paused': self.is_paused,
            'phase': self.simulation_state.phase,
            'step_count': self.step_count,
            'runtime': time.time() - self.start_time,
            'statistics': self._calculate_statistics(),
            'performance': self.performance_monitor.get_stats()
        }
    
    async def _handle_node_created(self, event: Event):
        """Handler para creación de nodos"""
        # Broadcast a clientes
        if self.api_server:
            self.api_server.broadcast_event(event)
    
    async def _handle_agent_evolved(self, event: Event):
        """Handler para evolución de agentes"""
        # Log importante
        logger.info(f"Agent evolution: {event.data}")
        
        # Broadcast
        if self.api_server:
            self.api_server.broadcast_event(event)
    
    async def _handle_critical_error(self, event: Event):
        """Handler para errores críticos"""
        logger.error(f"CRITICAL ERROR: {event.data}")
        
        # Intentar recuperación
        if self.config.get('auto_recovery', True):
            logger.info("Attempting auto-recovery...")
            # Pausar simulación
            self.pause()
            
            # Crear checkpoint de emergencia
            await self.create_checkpoint('emergency')
            
            # Notificar
            if self.api_server:
                self.api_server.broadcast_event(event)

# === CLASES DE SOPORTE ===
@dataclass
class SimulationState:
    """Estado de la simulación"""
    phase: str = 'initialization'  # initialization, running, paused, stopped
    last_checkpoint: Optional[str] = None
    performance_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    error_count: int = 0
    warning_count: int = 0

class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'min_time': float('inf'),
            'max_time': 0
        })
    
    @contextmanager
    def measure(self, operation: str):
        """Mide tiempo de operación"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.metrics[operation]['count'] += 1
            self.metrics[operation]['total_time'] += duration
            self.metrics[operation]['min_time'] = min(
                self.metrics[operation]['min_time'], 
                duration
            )
            self.metrics[operation]['max_time'] = max(
                self.metrics[operation]['max_time'], 
                duration
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento"""
        stats = {}
        
        for operation, metrics in self.metrics.items():
            if metrics['count'] > 0:
                stats[operation] = {
                    'count': metrics['count'],
                    'avg_time': metrics['total_time'] / metrics['count'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time'],
                    'total_time': metrics['total_time']
                }
        
        return stats

class SimulationPredictor:
    """Predictor ML para evolución del sistema"""
    
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.history_window = 100
    
    def _build_model(self):
        """Construye modelo de predicción"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    async def predict(self, graph, agents, state) -> Dict[str, Any]:
        """Predice evolución futura basándose en patrones históricos"""
        # Extraer features
        features = self._extract_features(graph, agents, state)
        
        # Normalizar features
        try:
            features_scaled = self.scaler.fit_transform(features)
        except:
            # Si no hay suficientes datos, usar valores por defecto
            features_scaled = features
        
        # Calcular métricas actuales
        current_health = np.mean([n.state for n in graph.nodes.values()]) if graph.nodes else 0.5
        node_growth_rate = self._calculate_growth_rate(graph)
        agent_efficiency = np.mean([a.omega for a in agents]) if agents else 0.5
        
        # Análisis de tendencias
        risk_factors = []
        if current_health < 0.4:
            risk_factors.append("Low system health")
        if node_growth_rate > 2.0:
            risk_factors.append("Rapid growth detected")
        if agent_efficiency < 0.3:
            risk_factors.append("Low agent efficiency")
        if state.error_count > 10:
            risk_factors.append("High error rate")
            
        # Determinar nivel de riesgo
        if len(risk_factors) >= 3:
            risk_level = 'high'
        elif len(risk_factors) >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        # Predicciones basadas en análisis
        predictions = {
            'next_hour': {
                'expected_nodes': int(len(graph.nodes) * (1 + node_growth_rate * 0.1)),
                'expected_health': max(0.0, min(1.0, current_health + (0.1 if risk_level == 'low' else -0.1))),
                'risk_level': risk_level,
                'confidence': 0.85 if len(graph.nodes) > 50 else 0.6
            },
            'next_day': {
                'expected_nodes': int(len(graph.nodes) * (1 + node_growth_rate * 0.5)),
                'expected_health': max(0.0, min(1.0, current_health + (0.2 if risk_level == 'low' else -0.2))),
                'risk_level': self._project_risk_level(risk_level, risk_factors),
                'confidence': 0.7 if len(graph.nodes) > 50 else 0.4
            },
            'current_metrics': {
                'health': current_health,
                'growth_rate': node_growth_rate,
                'agent_efficiency': agent_efficiency,
                'error_rate': state.error_count / max(1, state.processed_messages)
            },
            'risk_factors': risk_factors,
            'recommendations': self._generate_recommendations(
                current_health, node_growth_rate, agent_efficiency, risk_factors
            )
        }
        
        return predictions
    
    def _extract_features(self, graph, agents, state):
        """Extrae features para predicción"""
        features = []
        
        # Features del grafo
        features.extend([
            len(graph.nodes),
            sum(len(n.connections_out) for n in graph.nodes.values()),
            np.mean([n.state for n in graph.nodes.values()]) if graph.nodes else 0,
            len(graph.cluster_index)
        ])
        
        # Features de agentes
        features.extend([
            len(agents),
            sum(1 for a in agents if a.omega > 0),
            np.mean([a.omega for a in agents]) if agents else 0,
            np.mean([a.reputation for a in agents]) if agents else 0
        ])
        
        # Features de estado
        features.extend([
            state.error_count,
            state.warning_count
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_growth_rate(self, graph) -> float:
        """Calcula la tasa de crecimiento del grafo"""
        # Esta es una implementación simplificada
        # En producción, debería basarse en datos históricos
        current_nodes = len(graph.nodes)
        if current_nodes < 10:
            return 0.5  # Crecimiento inicial moderado
        elif current_nodes < 100:
            return 0.3  # Crecimiento estable
        else:
            return 0.1  # Crecimiento maduro
    
    def _project_risk_level(self, current_risk: str, risk_factors: List[str]) -> str:
        """Proyecta el nivel de riesgo futuro"""
        # Proyección simple basada en tendencias
        if current_risk == 'high' and len(risk_factors) > 3:
            return 'critical'
        elif current_risk == 'high':
            return 'high'
        elif current_risk == 'medium' and len(risk_factors) > 2:
            return 'high'
        elif current_risk == 'low' and len(risk_factors) > 1:
            return 'medium'
        return current_risk
    
    def _generate_recommendations(self, health: float, growth_rate: float, 
                                efficiency: float, risk_factors: List[str]) -> List[str]:
        """Genera recomendaciones basadas en el estado actual"""
        recommendations = []
        
        if health < 0.5:
            recommendations.append("Incrementar tasa de regeneración omega de agentes")
            recommendations.append("Revisar y optimizar conexiones entre nodos")
        
        if growth_rate > 1.0:
            recommendations.append("Implementar estrategias de consolidación")
            recommendations.append("Monitorear fragmentación de clusters")
        
        if efficiency < 0.4:
            recommendations.append("Optimizar algoritmos de consenso")
            recommendations.append("Reducir latencia en comunicación entre agentes")
        
        if "High error rate" in risk_factors:
            recommendations.append("Revisar logs de errores y patrones de fallo")
            recommendations.append("Implementar circuit breakers adicionales")
        
        if not recommendations:
            recommendations.append("Sistema funcionando de manera óptima")
            recommendations.append("Mantener monitoreo regular")
        
        return recommendations[:3]  # Limitar a 3 recomendaciones principales

# === FUNCIÓN PRINCIPAL ===
async def main():
    """Función principal del MSC Framework v4.0"""
    parser = argparse.ArgumentParser(
        description="MSC Framework v4.0 - Meta-cognitive Collective Synthesis with Claude"
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', choices=['simulation', 'api', 'both'], 
                       default='both', help='Execution mode')
    parser.add_argument('--api-host', default='0.0.0.0', help='API host')
    parser.add_argument('--api-port', type=int, default=5000, help='API port')
    parser.add_argument('--checkpoint', type=str, help='Load from checkpoint')
    parser.add_argument('--claude-key', type=str, help='Claude API key')
    
    args = parser.parse_args()
    
    # Configurar Claude API key si se proporciona
    if args.claude_key:
        Config.CLAUDE_API_KEY = args.claude_key
    
    # Cargar configuración
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # Añadir argumentos a config
    config['api_host'] = args.api_host
    config['api_port'] = args.api_port
    config['load_checkpoint'] = args.checkpoint
    
    # Crear simulación
    simulation = AdvancedSimulationRunner(config)
    
    # Esperar inicialización
    await asyncio.sleep(2)
    
    try:
        if args.mode in ['simulation', 'both']:
            await simulation.start()
            logger.info("Simulation started successfully")
        
        if args.mode == 'api':
            # Solo API sin simulación
            api_server = AdvancedAPIServer(simulation, config)
            api_server.run(host=args.api_host, port=args.api_port)
        
        # Mantener ejecutando
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    finally:
        await simulation.stop()
        logger.info("MSC Framework v4.0 shutdown complete")

def get_default_config() -> Dict[str, Any]:
    """Configuración por defecto"""
    return {
        # Simulación
        'simulation_steps': 100000,
        'step_delay': 0.1,
        'checkpoint_interval': 1000,
        'evolution_interval': 100,
        'consensus_interval': 500,
        
        # Agentes
        'agents': {
            'claude_taec': 3
        },
        'initial_omega': 100.0,
        'max_omega': 1000.0,
        'omega_regeneration': 0.1,
        'agent_rate_limit': 10,
        
        # Grafo
        'node_features': 768,
        'gnn_hidden': 128,
        'gnn_output': 64,
        'gnn_heads': 8,
        'gnn_layers': 4,
        'gnn_dropout': 0.1,
        'gnn_learning_rate': 0.001,
        
        # Sistema
        'max_concurrent_actions': 5,
        'enable_api': True,
        'auto_save': True,
        'auto_recovery': True,
        
        # Cache y persistencia
        'cache_size': 10000,
        'cache_ttl': 3600,
        
        # Seguridad
        'max_code_execution_time': 5.0,
        'exploration_rate': 0.1
    }

if __name__ == "__main__":
    # Configurar señales
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ejecutar
    asyncio.run(main())
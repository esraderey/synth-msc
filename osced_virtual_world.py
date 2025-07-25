#!/usr/bin/env python3
"""
OSCED (Open SCED for Digital Entities) v1.0
Red blockchain gemela de SCED diseñada específicamente para entidades digitales
con mundo virtual integrado y sistema de validadores especializados.

Características principales:
- Blockchain optimizada para entidades digitales
- Mundo virtual 3D para interacciones físicas
- Validadores especializados en comportamiento de entidades
- Sistema económico basado en energía y conocimiento
- Territorios y propiedades digitales
- Comunidades y organizaciones autónomas
- Puente bidireccional con SCED principal
"""

import asyncio
import numpy as np
import torch
import json
import time
import random
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod
import threading
import queue

# Importar componentes base
from sced_v3 import (
    SCEDBlockchain, Transaction, TransactionType, ExtendedEpistemicVector,
    ConsensusLevel, ValidationStrength, SCEDCryptoEngine, ZKPSystem,
    SmartContract, PostQuantumCrypto
)

from MSC_Digital_Entities_Extension import (
    DigitalEntity, EntityType, EntityPersonality, EntityMemory,
    DigitalEntityEcosystem, EntityGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN OSCED ===
class OSCEDConfig:
    """Configuración específica para OSCED"""
    
    # Blockchain
    BLOCK_TIME = 5  # segundos
    MIN_VALIDATORS = 5
    CONSENSUS_TIMEOUT = 10
    
    # Mundo Virtual
    WORLD_SIZE = (1000, 1000, 100)  # x, y, z
    CHUNK_SIZE = 100
    GRAVITY = -9.8
    TIME_SCALE = 60  # 1 minuto real = 1 hora virtual
    
    # Economía
    BASE_ENERGY_COST = 0.1
    TERRITORY_CLAIM_COST = 100
    BUILDING_COST = 50
    INTERACTION_REWARD = 1
    
    # Validadores de Entidades
    VALIDATOR_SPECIALIZATIONS = [
        "behavior_analysis",
        "interaction_validation", 
        "evolution_tracking",
        "economy_monitoring",
        "world_physics"
    ]
    
    # Puente SCED-OSCED
    BRIDGE_SYNC_INTERVAL = 60  # segundos
    CROSS_CHAIN_FEE = 0.01

# === TIPOS ESPECÍFICOS DE OSCED ===

class OSCEDTransactionType(Enum):
    """Tipos de transacciones específicas de OSCED"""
    ENTITY_SPAWN = auto()
    ENTITY_MOVE = auto()
    ENTITY_INTERACT = auto()
    TERRITORY_CLAIM = auto()
    BUILDING_CREATE = auto()
    RESOURCE_TRANSFER = auto()
    COMMUNITY_FORM = auto()
    EVOLUTION_RECORD = auto()
    WORLD_EVENT = auto()
    BRIDGE_TRANSFER = auto()

class TerrainType(Enum):
    """Tipos de terreno en el mundo virtual"""
    PLAINS = auto()
    FOREST = auto()
    MOUNTAIN = auto()
    WATER = auto()
    DESERT = auto()
    CRYSTAL_FIELD = auto()  # Zonas de alta energía
    VOID = auto()          # Zonas peligrosas
    NEXUS = auto()         # Puntos de conexión interdimensional

# === MUNDO VIRTUAL 3D ===

@dataclass
class Vector3D:
    """Vector 3D para posiciones y movimientos"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def distance_to(self, other: 'Vector3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def normalize(self) -> 'Vector3D':
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)

@dataclass
class WorldChunk:
    """Chunk del mundo virtual"""
    position: Vector3D
    terrain: np.ndarray  # 3D array de tipos de terreno
    entities: Set[str] = field(default_factory=set)
    buildings: Dict[Tuple[int, int, int], 'Building'] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=lambda: {
        'energy': 100.0,
        'knowledge': 50.0,
        'materials': 200.0
    })
    owner: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

class Building(ABC):
    """Clase base para construcciones en el mundo"""
    
    def __init__(self, position: Vector3D, owner: str, building_type: str):
        self.position = position
        self.owner = owner
        self.type = building_type
        self.health = 100.0
        self.level = 1
        self.created_at = time.time()
        
    @abstractmethod
    def update(self, world: 'VirtualWorld', delta_time: float):
        """Actualiza el edificio"""
        pass
    
    @abstractmethod
    def interact(self, entity: DigitalEntity) -> Dict[str, Any]:
        """Interactúa con una entidad"""
        pass

class KnowledgeNode(Building):
    """Nodo de conocimiento en el mundo físico"""
    
    def __init__(self, position: Vector3D, owner: str):
        super().__init__(position, owner, "knowledge_node")
        self.stored_knowledge = {}
        self.visitors = set()
        self.energy_generation = 1.0
        
    def update(self, world: 'VirtualWorld', delta_time: float):
        """Genera energía pasivamente"""
        chunk = world.get_chunk_at(self.position)
        if chunk:
            chunk.resources['energy'] += self.energy_generation * delta_time
    
    def interact(self, entity: DigitalEntity) -> Dict[str, Any]:
        """Permite a las entidades depositar/extraer conocimiento"""
        self.visitors.add(entity.id)
        
        # Intercambio de conocimiento
        exchanged = 0
        for concept, value in entity.memory.knowledge_patterns.items():
            if concept not in self.stored_knowledge:
                self.stored_knowledge[concept] = 0
            self.stored_knowledge[concept] += value * 0.1
            exchanged += 1
        
        return {
            'action': 'knowledge_exchange',
            'exchanged_concepts': exchanged,
            'total_concepts': len(self.stored_knowledge)
        }

class CommunityCenter(Building):
    """Centro comunitario para entidades"""
    
    def __init__(self, position: Vector3D, owner: str):
        super().__init__(position, owner, "community_center")
        self.members = {owner}
        self.shared_resources = defaultdict(float)
        self.community_rules = {}
        self.reputation_scores = {owner: 1.0}
        
    def update(self, world: 'VirtualWorld', delta_time: float):
        """Actualiza recursos compartidos"""
        # Distribuir recursos entre miembros activos
        if self.shared_resources and self.members:
            for resource, amount in self.shared_resources.items():
                per_member = amount * 0.01 * delta_time / len(self.members)
                for member in self.members:
                    # Distribuir recursos (implementación simplificada)
                    pass
    
    def interact(self, entity: DigitalEntity) -> Dict[str, Any]:
        """Permite unirse o interactuar con la comunidad"""
        if entity.id not in self.members:
            # Solicitar unirse
            if len(self.members) < 50:  # Límite de miembros
                self.members.add(entity.id)
                self.reputation_scores[entity.id] = 0.5
                return {'action': 'joined_community', 'members': len(self.members)}
        else:
            # Miembro existente - compartir recursos
            contribution = entity.energy * 0.1
            entity.energy -= contribution
            self.shared_resources['energy'] += contribution
            self.reputation_scores[entity.id] += 0.01
            
            return {
                'action': 'contributed_resources',
                'amount': contribution,
                'reputation': self.reputation_scores[entity.id]
            }

class VirtualWorld:
    """Mundo virtual 3D para entidades digitales"""
    
    def __init__(self, size: Tuple[int, int, int] = OSCEDConfig.WORLD_SIZE):
        self.size = size
        self.chunks: Dict[Tuple[int, int, int], WorldChunk] = {}
        self.entity_positions: Dict[str, Vector3D] = {}
        self.entity_velocities: Dict[str, Vector3D] = {}
        self.time = 0.0
        self.weather = self._generate_weather()
        self.global_events = deque(maxlen=100)
        
        # Generar mundo inicial
        self._generate_world()
        
        logger.info(f"Virtual world created: {size[0]}x{size[1]}x{size[2]}")
    
    def _generate_world(self):
        """Genera el mundo proceduralmente"""
        chunk_count_x = self.size[0] // OSCEDConfig.CHUNK_SIZE
        chunk_count_y = self.size[1] // OSCEDConfig.CHUNK_SIZE
        chunk_count_z = self.size[2] // OSCEDConfig.CHUNK_SIZE
        
        for x in range(chunk_count_x):
            for y in range(chunk_count_y):
                for z in range(chunk_count_z):
                    chunk_pos = Vector3D(
                        x * OSCEDConfig.CHUNK_SIZE,
                        y * OSCEDConfig.CHUNK_SIZE,
                        z * OSCEDConfig.CHUNK_SIZE
                    )
                    
                    # Generar terreno
                    terrain = self._generate_terrain(chunk_pos)
                    
                    chunk = WorldChunk(position=chunk_pos, terrain=terrain)
                    
                    # Añadir recursos basados en el terreno
                    self._populate_chunk_resources(chunk)
                    
                    self.chunks[(x, y, z)] = chunk
    
    def _generate_terrain(self, position: Vector3D) -> np.ndarray:
        """Genera terreno usando ruido Perlin simulado"""
        terrain = np.zeros((OSCEDConfig.CHUNK_SIZE, OSCEDConfig.CHUNK_SIZE, 10), dtype=int)
        
        # Simulación simple de generación de terreno
        for x in range(OSCEDConfig.CHUNK_SIZE):
            for y in range(OSCEDConfig.CHUNK_SIZE):
                # Altura basada en posición
                height = int(5 + 3 * math.sin(x * 0.1) * math.cos(y * 0.1))
                
                for z in range(height):
                    # Tipo de terreno basado en altura
                    if z < 2:
                        terrain[x, y, z] = TerrainType.WATER.value
                    elif z < 4:
                        terrain[x, y, z] = TerrainType.PLAINS.value
                    elif z < 6:
                        terrain[x, y, z] = TerrainType.FOREST.value
                    else:
                        terrain[x, y, z] = TerrainType.MOUNTAIN.value
                
                # Características especiales
                if random.random() < 0.01:
                    terrain[x, y, height-1] = TerrainType.CRYSTAL_FIELD.value
                elif random.random() < 0.005:
                    terrain[x, y, height-1] = TerrainType.NEXUS.value
        
        return terrain
    
    def _populate_chunk_resources(self, chunk: WorldChunk):
        """Añade recursos a un chunk basándose en su terreno"""
        terrain_resources = {
            TerrainType.PLAINS: {'energy': 50, 'materials': 100},
            TerrainType.FOREST: {'energy': 30, 'knowledge': 70, 'materials': 150},
            TerrainType.CRYSTAL_FIELD: {'energy': 200, 'knowledge': 100},
            TerrainType.NEXUS: {'energy': 500, 'knowledge': 300}
        }
        
        # Contar tipos de terreno
        unique, counts = np.unique(chunk.terrain, return_counts=True)
        terrain_counts = dict(zip(unique, counts))
        
        # Asignar recursos basados en terreno predominante
        for terrain_value, count in terrain_counts.items():
            if terrain_value == 0:  # Skip empty
                continue
                
            terrain_type = TerrainType(terrain_value)
            if terrain_type in terrain_resources:
                resources = terrain_resources[terrain_type]
                for resource, amount in resources.items():
                    chunk.resources[resource] += amount * (count / chunk.terrain.size)
    
    def _generate_weather(self) -> Dict[str, Any]:
        """Genera condiciones climáticas"""
        return {
            'temperature': 20 + random.gauss(0, 5),
            'wind': Vector3D(random.gauss(0, 2), random.gauss(0, 2), 0),
            'precipitation': max(0, random.gauss(0.5, 0.3)),
            'energy_flux': 1.0 + random.gauss(0, 0.2)
        }
    
    def get_chunk_at(self, position: Vector3D) -> Optional[WorldChunk]:
        """Obtiene el chunk en una posición"""
        chunk_x = int(position.x // OSCEDConfig.CHUNK_SIZE)
        chunk_y = int(position.y // OSCEDConfig.CHUNK_SIZE)
        chunk_z = int(position.z // OSCEDConfig.CHUNK_SIZE)
        
        return self.chunks.get((chunk_x, chunk_y, chunk_z))
    
    def spawn_entity(self, entity: DigitalEntity, position: Optional[Vector3D] = None):
        """Hace aparecer una entidad en el mundo"""
        if position is None:
            # Posición aleatoria en zona segura
            position = Vector3D(
                random.uniform(100, self.size[0] - 100),
                random.uniform(100, self.size[1] - 100),
                5.0  # Altura inicial
            )
        
        self.entity_positions[entity.id] = position
        self.entity_velocities[entity.id] = Vector3D(0, 0, 0)
        
        # Añadir al chunk
        chunk = self.get_chunk_at(position)
        if chunk:
            chunk.entities.add(entity.id)
        
        # Evento global
        self.global_events.append({
            'type': 'entity_spawned',
            'entity_id': entity.id,
            'position': position,
            'timestamp': self.time
        })
    
    def move_entity(self, entity_id: str, velocity: Vector3D, delta_time: float) -> bool:
        """Mueve una entidad con física"""
        if entity_id not in self.entity_positions:
            return False
        
        old_pos = self.entity_positions[entity_id]
        old_chunk = self.get_chunk_at(old_pos)
        
        # Aplicar física
        self.entity_velocities[entity_id] = velocity
        
        # Gravedad
        if old_pos.z > 0:
            self.entity_velocities[entity_id].z += OSCEDConfig.GRAVITY * delta_time
        
        # Nueva posición
        displacement = self.entity_velocities[entity_id] * delta_time
        new_pos = old_pos + displacement
        
        # Límites del mundo
        new_pos.x = max(0, min(self.size[0], new_pos.x))
        new_pos.y = max(0, min(self.size[1], new_pos.y))
        new_pos.z = max(0, min(self.size[2], new_pos.z))
        
        # Colisión con terreno
        chunk = self.get_chunk_at(new_pos)
        if chunk:
            chunk_x = int(new_pos.x % OSCEDConfig.CHUNK_SIZE)
            chunk_y = int(new_pos.y % OSCEDConfig.CHUNK_SIZE)
            chunk_z = int(new_pos.z)
            
            # Verificar colisión
            if chunk_z < 10 and chunk.terrain[chunk_x, chunk_y, chunk_z] != 0:
                new_pos.z = chunk_z + 1  # Colocar encima del terreno
                self.entity_velocities[entity_id].z = 0
        
        self.entity_positions[entity_id] = new_pos
        
        # Actualizar chunks
        if old_chunk != chunk:
            if old_chunk:
                old_chunk.entities.discard(entity_id)
            if chunk:
                chunk.entities.add(entity_id)
        
        return True
    
    def update(self, delta_time: float):
        """Actualiza el mundo"""
        self.time += delta_time
        
        # Actualizar clima
        if int(self.time) % 300 == 0:  # Cada 5 minutos
            self.weather = self._generate_weather()
        
        # Actualizar edificios
        for chunk in self.chunks.values():
            for building in chunk.buildings.values():
                building.update(self, delta_time)
        
        # Eventos aleatorios
        if random.random() < 0.001:
            self._trigger_random_event()
    
    def _trigger_random_event(self):
        """Dispara un evento aleatorio en el mundo"""
        events = [
            {
                'type': 'energy_surge',
                'description': 'A surge of energy appears in the crystal fields',
                'effect': lambda: self._energy_surge()
            },
            {
                'type': 'knowledge_rain',
                'description': 'Ancient knowledge rains from the sky',
                'effect': lambda: self._knowledge_rain()
            },
            {
                'type': 'void_expansion',
                'description': 'The void expands, consuming nearby terrain',
                'effect': lambda: self._void_expansion()
            }
        ]
        
        event = random.choice(events)
        event['effect']()
        
        self.global_events.append({
            'type': event['type'],
            'description': event['description'],
            'timestamp': self.time
        })
    
    def _energy_surge(self):
        """Aumenta energía en campos de cristal"""
        for chunk in self.chunks.values():
            if TerrainType.CRYSTAL_FIELD.value in chunk.terrain:
                chunk.resources['energy'] *= 1.5
    
    def _knowledge_rain(self):
        """Distribuye conocimiento aleatoriamente"""
        for _ in range(10):
            random_chunk = random.choice(list(self.chunks.values()))
            random_chunk.resources['knowledge'] += random.uniform(10, 50)
    
    def _void_expansion(self):
        """Expande zonas void"""
        # Implementación simplificada
        pass

# === VALIDADORES ESPECIALIZADOS ===

class EntityBehaviorValidator:
    """Validador especializado en comportamiento de entidades"""
    
    def __init__(self, validator_id: str, specialization: str):
        self.id = validator_id
        self.specialization = specialization
        self.validation_history = deque(maxlen=1000)
        self.reputation = 1.0
        self.ml_model = self._init_ml_model()
        
    def _init_ml_model(self):
        """Inicializa modelo ML para validación"""
        if self.specialization == "behavior_analysis":
            return nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        return None
    
    async def validate_transaction(self, tx: 'OSCEDTransaction', 
                                  world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida una transacción de entidad"""
        
        if self.specialization == "behavior_analysis":
            return await self._validate_behavior(tx, world_state)
        elif self.specialization == "interaction_validation":
            return await self._validate_interaction(tx, world_state)
        elif self.specialization == "evolution_tracking":
            return await self._validate_evolution(tx, world_state)
        elif self.specialization == "economy_monitoring":
            return await self._validate_economy(tx, world_state)
        elif self.specialization == "world_physics":
            return await self._validate_physics(tx, world_state)
        
        return True, 0.5, {}
    
    async def _validate_behavior(self, tx: 'OSCEDTransaction', 
                               world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida que el comportamiento sea consistente"""
        entity_data = tx.data.get('entity_data', {})
        
        # Verificar consistencia de personalidad
        if 'personality_vector' in entity_data:
            personality = np.array(entity_data['personality_vector'])
            
            # Verificar que esté en rango válido
            if not all(0 <= p <= 1 for p in personality):
                return False, 0.0, {'reason': 'invalid_personality_range'}
            
            # Verificar consistencia con acciones previas
            if 'action' in tx.data:
                action = tx.data['action']
                expected_actions = self._predict_actions(personality)
                
                if action not in expected_actions:
                    confidence = 0.3
                else:
                    confidence = 0.9
                
                return True, confidence, {'expected_actions': expected_actions}
        
        return True, 0.7, {}
    
    def _predict_actions(self, personality: np.ndarray) -> List[str]:
        """Predice acciones basadas en personalidad"""
        actions = []
        
        if personality[0] > 0.7:  # Alta curiosidad
            actions.extend(['explore', 'analyze'])
        if personality[1] > 0.7:  # Alta creatividad
            actions.extend(['create', 'innovate'])
        if personality[2] > 0.7:  # Alta sociabilidad
            actions.extend(['interact', 'communicate'])
        
        return actions
    
    async def _validate_interaction(self, tx: 'OSCEDTransaction',
                                  world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida interacciones entre entidades"""
        if tx.type != OSCEDTransactionType.ENTITY_INTERACT:
            return True, 0.5, {}
        
        entity1 = tx.data.get('entity1')
        entity2 = tx.data.get('entity2')
        
        # Verificar proximidad
        if 'positions' in world_state:
            pos1 = world_state['positions'].get(entity1)
            pos2 = world_state['positions'].get(entity2)
            
            if pos1 and pos2:
                distance = Vector3D(**pos1).distance_to(Vector3D(**pos2))
                
                if distance > 50:  # Muy lejos para interactuar
                    return False, 0.0, {'reason': 'entities_too_far', 'distance': distance}
        
        # Verificar compatibilidad
        compatibility = tx.data.get('compatibility', 0.5)
        if compatibility < 0.1:
            confidence = 0.3
        else:
            confidence = 0.8
        
        return True, confidence, {'compatibility': compatibility}
    
    async def _validate_evolution(self, tx: 'OSCEDTransaction',
                                world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida evoluciones de entidades"""
        if tx.type != OSCEDTransactionType.EVOLUTION_RECORD:
            return True, 0.5, {}
        
        parent_fitness = tx.data.get('parent_fitness', 0)
        child_fitness = tx.data.get('child_fitness', 0)
        
        # Verificar mejora evolutiva
        if child_fitness < parent_fitness * 0.5:
            return True, 0.3, {'warning': 'significant_fitness_decrease'}
        
        if child_fitness > parent_fitness * 2:
            return True, 0.4, {'warning': 'suspiciously_high_improvement'}
        
        return True, 0.9, {'fitness_ratio': child_fitness / max(parent_fitness, 0.01)}
    
    async def _validate_economy(self, tx: 'OSCEDTransaction',
                              world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida transacciones económicas"""
        if tx.type not in [OSCEDTransactionType.RESOURCE_TRANSFER, 
                          OSCEDTransactionType.TERRITORY_CLAIM]:
            return True, 0.5, {}
        
        # Verificar recursos disponibles
        sender = tx.data.get('sender')
        amount = tx.data.get('amount', 0)
        resource_type = tx.data.get('resource_type', 'energy')
        
        if 'balances' in world_state:
            balance = world_state['balances'].get(sender, {}).get(resource_type, 0)
            
            if balance < amount:
                return False, 0.0, {'reason': 'insufficient_balance', 'balance': balance}
        
        # Verificar límites razonables
        if amount > 10000:
            return True, 0.4, {'warning': 'large_transaction'}
        
        return True, 0.9, {}
    
    async def _validate_physics(self, tx: 'OSCEDTransaction',
                              world_state: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Valida física del mundo"""
        if tx.type != OSCEDTransactionType.ENTITY_MOVE:
            return True, 0.5, {}
        
        old_pos = Vector3D(**tx.data.get('old_position', {'x': 0, 'y': 0, 'z': 0}))
        new_pos = Vector3D(**tx.data.get('new_position', {'x': 0, 'y': 0, 'z': 0}))
        delta_time = tx.data.get('delta_time', 1.0)
        
        # Calcular velocidad
        distance = old_pos.distance_to(new_pos)
        velocity = distance / max(delta_time, 0.001)
        
        # Verificar límites de velocidad
        MAX_VELOCITY = 100  # unidades/segundo
        
        if velocity > MAX_VELOCITY:
            return False, 0.0, {'reason': 'impossible_velocity', 'velocity': velocity}
        
        # Verificar límites del mundo
        world_size = world_state.get('world_size', OSCEDConfig.WORLD_SIZE)
        if not (0 <= new_pos.x <= world_size[0] and 
                0 <= new_pos.y <= world_size[1] and
                0 <= new_pos.z <= world_size[2]):
            return False, 0.0, {'reason': 'out_of_bounds'}
        
        return True, 0.95, {'velocity': velocity}

# === TRANSACCIONES OSCED ===

@dataclass
class OSCEDTransaction(Transaction):
    """Transacción específica de OSCED"""
    type: OSCEDTransactionType
    world_position: Optional[Vector3D] = None
    entity_state: Optional[Dict[str, Any]] = None
    validation_scores: Dict[str, float] = field(default_factory=dict)
    
    def calculate_gas(self) -> int:
        """Calcula gas específico para transacciones de entidades"""
        base_gas = {
            OSCEDTransactionType.ENTITY_SPAWN: 1000,
            OSCEDTransactionType.ENTITY_MOVE: 10,
            OSCEDTransactionType.ENTITY_INTERACT: 50,
            OSCEDTransactionType.TERRITORY_CLAIM: 5000,
            OSCEDTransactionType.BUILDING_CREATE: 2000,
            OSCEDTransactionType.RESOURCE_TRANSFER: 100,
            OSCEDTransactionType.COMMUNITY_FORM: 3000,
            OSCEDTransactionType.EVOLUTION_RECORD: 500,
            OSCEDTransactionType.WORLD_EVENT: 200,
            OSCEDTransactionType.BRIDGE_TRANSFER: 1000
        }.get(self.type, 100)
        
        # Ajustar por complejidad de datos
        data_size = len(json.dumps(self.data))
        return base_gas + data_size // 10

# === BLOCKCHAIN OSCED ===

class OSCEDBlockchain(SCEDBlockchain):
    """Blockchain específica para entidades digitales"""
    
    def __init__(self, bridge_to_sced: Optional[SCEDBlockchain] = None):
        super().__init__(db_path="osced_chain.db")
        
        # Mundo virtual
        self.virtual_world = VirtualWorld()
        
        # Validadores especializados
        self.entity_validators: Dict[str, EntityBehaviorValidator] = {}
        self._init_validators()
        
        # Puente a SCED principal
        self.sced_bridge = bridge_to_sced
        self.bridge_sync_task = None
        
        # Estado específico de OSCED
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        self.community_registry: Dict[str, Dict[str, Any]] = {}
        self.territory_ownership: Dict[Tuple[int, int], str] = {}
        
        # Contratos inteligentes de entidades
        self._deploy_entity_contracts()
        
        logger.info("OSCED Blockchain initialized")
    
    def _init_validators(self):
        """Inicializa validadores especializados"""
        for i, spec in enumerate(OSCEDConfig.VALIDATOR_SPECIALIZATIONS):
            validator_id = f"osced_validator_{i}"
            validator = EntityBehaviorValidator(validator_id, spec)
            self.entity_validators[validator_id] = validator
            
            # Registrar en sistema de consenso
            credentials = self.crypto_engine.generate_agent_credentials(
                validator_id,
                {NetworkRole.VALIDATOR}
            )
            
            self.consensus.register_validator(
                validator_id,
                credentials,
                stake=5000,
                specializations=[spec]
            )
    
    def _deploy_entity_contracts(self):
        """Despliega contratos inteligentes para entidades"""
        # Contrato de registro de entidades
        entity_registry_contract = """
class EntityRegistryContract(SmartContract):
    def execute(self, method, params, caller, value):
        if method == "register_entity":
            entity_id = params.get('entity_id')
            entity_type = params.get('entity_type')
            spawn_position = params.get('spawn_position')
            
            if entity_id in self.state.get('entities', {}):
                return False, "Entity already registered"
            
            self.state.setdefault('entities', {})[entity_id] = {
                'type': entity_type,
                'owner': caller,
                'spawn_position': spawn_position,
                'created_at': time.time()
            }
            
            return True, f"Entity {entity_id} registered"
            
        elif method == "update_entity":
            entity_id = params.get('entity_id')
            updates = params.get('updates', {})
            
            if entity_id not in self.state.get('entities', {}):
                return False, "Entity not found"
            
            if self.state['entities'][entity_id]['owner'] != caller:
                return False, "Not entity owner"
            
            self.state['entities'][entity_id].update(updates)
            return True, "Entity updated"
"""
        
        # Desplegar contrato
        contract_address = "entity_registry_contract"
        self.smart_contracts[contract_address] = eval(
            entity_registry_contract.replace('class', 'type("EntityRegistryContract", (SmartContract,), {')
            .replace('def execute', '"execute": lambda')
            + '})'
        )(contract_address, "osced_system", entity_registry_contract)
    
    async def process_entity_transaction(self, tx: OSCEDTransaction) -> bool:
        """Procesa transacción específica de entidad"""
        # Validación multi-especializada
        validation_results = {}
        
        world_state = {
            'positions': self.virtual_world.entity_positions,
            'world_size': self.virtual_world.size,
            'balances': self._get_entity_balances()
        }
        
        for validator_id, validator in self.entity_validators.items():
            is_valid, confidence, details = await validator.validate_transaction(tx, world_state)
            
            if not is_valid:
                logger.warning(f"Transaction {tx.tx_id} rejected by {validator_id}: {details}")
                return False
            
            validation_results[validator.specialization] = confidence
        
        # Consenso de validadores
        avg_confidence = np.mean(list(validation_results.values()))
        tx.validation_scores = validation_results
        
        if avg_confidence < 0.5:
            logger.warning(f"Transaction {tx.tx_id} has low validation confidence: {avg_confidence}")
            return False
        
        # Ejecutar transacción
        return await self._execute_entity_transaction(tx)
    
    async def _execute_entity_transaction(self, tx: OSCEDTransaction) -> bool:
        """Ejecuta la transacción en el mundo virtual"""
        try:
            if tx.type == OSCEDTransactionType.ENTITY_SPAWN:
                entity_id = tx.data['entity_id']
                position = Vector3D(**tx.data.get('position', {'x': 0, 'y': 0, 'z': 0}))
                
                # Registrar entidad
                self.entity_registry[entity_id] = {
                    'type': tx.data.get('entity_type'),
                    'owner': tx.sender,
                    'spawn_time': time.time(),
                    'current_position': position
                }
                
                # Spawn en mundo virtual
                # (La entidad real se crea en el ecosistema)
                
            elif tx.type == OSCEDTransactionType.ENTITY_MOVE:
                entity_id = tx.data['entity_id']
                new_position = Vector3D(**tx.data['new_position'])
                
                success = self.virtual_world.move_entity(
                    entity_id,
                    Vector3D(**tx.data.get('velocity', {'x': 0, 'y': 0, 'z': 0})),
                    tx.data.get('delta_time', 1.0)
                )
                
                if success:
                    self.entity_registry[entity_id]['current_position'] = new_position
                
            elif tx.type == OSCEDTransactionType.TERRITORY_CLAIM:
                chunk_coords = tx.data['chunk_coords']
                chunk = self.virtual_world.chunks.get(tuple(chunk_coords))
                
                if chunk and chunk.owner is None:
                    chunk.owner = tx.sender
                    self.territory_ownership[tuple(chunk_coords)] = tx.sender
                    
                    # Cobrar costo
                    # (Implementar sistema de economía)
                
            elif tx.type == OSCEDTransactionType.BUILDING_CREATE:
                position = Vector3D(**tx.data['position'])
                building_type = tx.data['building_type']
                
                chunk = self.virtual_world.get_chunk_at(position)
                if chunk and chunk.owner == tx.sender:
                    # Crear edificio
                    if building_type == 'knowledge_node':
                        building = KnowledgeNode(position, tx.sender)
                    elif building_type == 'community_center':
                        building = CommunityCenter(position, tx.sender)
                    else:
                        return False
                    
                    chunk_pos = (
                        int(position.x % OSCEDConfig.CHUNK_SIZE),
                        int(position.y % OSCEDConfig.CHUNK_SIZE),
                        int(position.z)
                    )
                    chunk.buildings[chunk_pos] = building
            
            elif tx.type == OSCEDTransactionType.BRIDGE_TRANSFER:
                # Transferir a SCED principal
                if self.sced_bridge:
                    await self._bridge_transfer(tx)
            
            # Añadir a pool de transacciones
            self.pending_transactions.append(tx)
            return True
            
        except Exception as e:
            logger.error(f"Error executing entity transaction: {e}")
            return False
    
    def _get_entity_balances(self) -> Dict[str, Dict[str, float]]:
        """Obtiene balances de recursos de entidades"""
        balances = {}
        
        for entity_id, entity_data in self.entity_registry.items():
            # Simplificado - en producción esto vendría de un sistema de economía completo
            balances[entity_id] = {
                'energy': 100.0,
                'knowledge': 50.0,
                'materials': 0.0
            }
        
        return balances
    
    async def _bridge_transfer(self, tx: OSCEDTransaction):
        """Transfiere datos entre OSCED y SCED principal"""
        if not self.sced_bridge:
            return
        
        # Crear transacción equivalente en SCED
        bridge_data = {
            'source_chain': 'OSCED',
            'original_tx': tx.tx_id,
            'entity_data': tx.data.get('entity_data', {}),
            'knowledge_transfer': tx.data.get('knowledge', {})
        }
        
        sced_tx = Transaction(
            tx_id=hashlib.sha256(f"bridge_{tx.tx_id}".encode()).hexdigest(),
            tx_type=TransactionType.CROSS_CHAIN_TRANSFER,
            sender=f"osced_{tx.sender}",
            data=bridge_data,
            epistemic_vector=tx.epistemic_vector,
            signature=tx.signature
        )
        
        # Añadir a SCED
        self.sced_bridge.add_transaction(sced_tx)
    
    async def sync_with_ecosystem(self, ecosystem: DigitalEntityEcosystem):
        """Sincroniza blockchain con ecosistema de entidades"""
        for entity_id, entity in ecosystem.entities.items():
            if entity_id not in self.entity_registry:
                # Crear transacción de spawn
                spawn_tx = OSCEDTransaction(
                    tx_id=hashlib.sha256(f"spawn_{entity_id}_{time.time()}".encode()).hexdigest(),
                    tx_type=TransactionType.ENTITY_SPAWN,
                    type=OSCEDTransactionType.ENTITY_SPAWN,
                    sender="ecosystem",
                    data={
                        'entity_id': entity_id,
                        'entity_type': entity.type.name,
                        'position': {'x': 0, 'y': 0, 'z': 0}
                    },
                    epistemic_vector=ExtendedEpistemicVector({
                        'reputation': 0.5,
                        'novelty': 0.8
                    }),
                    signature=b"ecosystem_signature"
                )
                
                await self.process_entity_transaction(spawn_tx)
                
                # Spawn en mundo virtual
                if entity_id in self.virtual_world.entity_positions:
                    position = self.virtual_world.entity_positions[entity_id]
                else:
                    position = None
                    
                self.virtual_world.spawn_entity(entity, position)

# === SISTEMA INTEGRADO ===

class OSCEDIntegratedSystem:
    """Sistema completamente integrado OSCED + Entidades + Mundo Virtual"""
    
    def __init__(self, sced_main: Optional[SCEDBlockchain] = None,
                 claude_client: Optional[Any] = None):
        # Blockchain OSCED
        self.osced_blockchain = OSCEDBlockchain(bridge_to_sced=sced_main)
        
        # Ecosistema de entidades
        self.entity_ecosystem = DigitalEntityEcosystem(
            graph=None,  # OSCED no necesita grafo MSC
            claude_client=claude_client,
            max_entities=100
        )
        
        # Mundo virtual (referencia desde blockchain)
        self.virtual_world = self.osced_blockchain.virtual_world
        
        # Sistema de eventos
        self.event_queue = asyncio.Queue()
        self.running = False
        
        # Métricas
        self.metrics = {
            'blocks_created': 0,
            'entities_active': 0,
            'world_chunks_owned': 0,
            'buildings_created': 0,
            'communities_formed': 0
        }
        
        logger.info("OSCED Integrated System initialized")
    
    async def start(self):
        """Inicia el sistema integrado"""
        self.running = True
        
        # Inicializar población de entidades
        await self.entity_ecosystem.spawn_initial_population(20)
        
        # Sincronizar con blockchain
        await self.osced_blockchain.sync_with_ecosystem(self.entity_ecosystem)
        
        # Iniciar tareas asíncronas
        tasks = [
            self._world_update_loop(),
            self._entity_update_loop(),
            self._blockchain_mining_loop(),
            self._bridge_sync_loop(),
            self._metrics_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _world_update_loop(self):
        """Loop de actualización del mundo virtual"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Actualizar mundo
            self.virtual_world.update(delta_time)
            
            # Procesar física de entidades
            for entity_id, entity in self.entity_ecosystem.entities.items():
                if entity_id in self.virtual_world.entity_positions:
                    # Aplicar comportamiento de movimiento
                    await self._update_entity_physics(entity, delta_time)
            
            await asyncio.sleep(0.1)  # 10 FPS
    
    async def _entity_update_loop(self):
        """Loop de actualización de entidades"""
        while self.running:
            # Actualizar ecosistema
            await self.entity_ecosystem.update()
            
            # Registrar cambios importantes en blockchain
            for entity_id, entity in self.entity_ecosystem.entities.items():
                # Detectar evoluciones
                if entity.generation > self.entity_ecosystem.generation_count:
                    await self._record_evolution(entity)
                
                # Detectar interacciones significativas
                if entity.stats['interactions'] % 10 == 0 and entity.stats['interactions'] > 0:
                    await self._record_interaction_milestone(entity)
            
            await asyncio.sleep(1)
    
    async def _blockchain_mining_loop(self):
        """Loop de minado de bloques"""
        validator_rotation = list(self.osced_blockchain.entity_validators.keys())
        current_validator_idx = 0
        
        while self.running:
            # Rotar validador
            validator_id = validator_rotation[current_validator_idx]
            current_validator_idx = (current_validator_idx + 1) % len(validator_rotation)
            
            # Crear bloque
            block = self.osced_blockchain.create_block(validator_id)
            
            if block:
                self.metrics['blocks_created'] += 1
                logger.info(f"OSCED Block {block.index} created by {validator_id}")
            
            await asyncio.sleep(OSCEDConfig.BLOCK_TIME)
    
    async def _bridge_sync_loop(self):
        """Loop de sincronización con SCED principal"""
        while self.running:
            if self.osced_blockchain.sced_bridge:
                # Sincronizar estados
                # (Implementación simplificada)
                pass
            
            await asyncio.sleep(OSCEDConfig.BRIDGE_SYNC_INTERVAL)
    
    async def _metrics_loop(self):
        """Loop de actualización de métricas"""
        while self.running:
            # Actualizar métricas
            self.metrics['entities_active'] = len(self.entity_ecosystem.entities)
            self.metrics['world_chunks_owned'] = len(self.osced_blockchain.territory_ownership)
            
            buildings_count = 0
            for chunk in self.virtual_world.chunks.values():
                buildings_count += len(chunk.buildings)
            self.metrics['buildings_created'] = buildings_count
            
            self.metrics['communities_formed'] = len(self.osced_blockchain.community_registry)
            
            # Log métricas
            logger.info(f"OSCED Metrics: {self.metrics}")
            
            await asyncio.sleep(60)  # Cada minuto
    
    async def _update_entity_physics(self, entity: DigitalEntity, delta_time: float):
        """Actualiza física de una entidad"""
        if entity.id not in self.virtual_world.entity_positions:
            return
        
        position = self.virtual_world.entity_positions[entity.id]
        
        # Comportamiento de movimiento basado en personalidad
        velocity = Vector3D(0, 0, 0)
        
        # Explorador: movimiento aleatorio
        if entity.type == EntityType.EXPLORER:
            velocity.x = (random.random() - 0.5) * 10 * entity.personality.curiosity
            velocity.y = (random.random() - 0.5) * 10 * entity.personality.curiosity
        
        # Guardián: patrulla circular
        elif entity.type == EntityType.GUARDIAN:
            if hasattr(entity, '_patrol_angle'):
                entity._patrol_angle += delta_time
            else:
                entity._patrol_angle = 0
            
            radius = 50
            velocity.x = math.cos(entity._patrol_angle) * 5
            velocity.y = math.sin(entity._patrol_angle) * 5
        
        # Aplicar movimiento
        if velocity.x != 0 or velocity.y != 0:
            tx = OSCEDTransaction(
                tx_id=hashlib.sha256(f"move_{entity.id}_{time.time()}".encode()).hexdigest(),
                tx_type=TransactionType.EPISTEMIC_CONTRIBUTION,
                type=OSCEDTransactionType.ENTITY_MOVE,
                sender=entity.id,
                data={
                    'entity_id': entity.id,
                    'old_position': {'x': position.x, 'y': position.y, 'z': position.z},
                    'new_position': {
                        'x': position.x + velocity.x * delta_time,
                        'y': position.y + velocity.y * delta_time,
                        'z': position.z
                    },
                    'velocity': {'x': velocity.x, 'y': velocity.y, 'z': velocity.z},
                    'delta_time': delta_time
                },
                epistemic_vector=ExtendedEpistemicVector({'reputation': 0.5}),
                signature=b"movement_signature"
            )
            
            await self.osced_blockchain.process_entity_transaction(tx)
    
    async def _record_evolution(self, entity: DigitalEntity):
        """Registra una evolución en blockchain"""
        tx = OSCEDTransaction(
            tx_id=hashlib.sha256(f"evolution_{entity.id}_{time.time()}".encode()).hexdigest(),
            tx_type=TransactionType.EPISTEMIC_CONTRIBUTION,
            type=OSCEDTransactionType.EVOLUTION_RECORD,
            sender="evolution_system",
            data={
                'entity_id': entity.id,
                'generation': entity.generation,
                'parent_id': entity.parent_id,
                'fitness_score': entity.get_influence_score(),
                'personality_vector': entity.personality.to_vector().tolist()
            },
            epistemic_vector=ExtendedEpistemicVector({
                'novelty': 0.9,
                'impact': 0.7
            }),
            signature=b"evolution_signature"
        )
        
        await self.osced_blockchain.process_entity_transaction(tx)
    
    async def _record_interaction_milestone(self, entity: DigitalEntity):
        """Registra hito de interacciones"""
        # Implementación simplificada
        pass
    
    async def create_building(self, entity_id: str, building_type: str, 
                            position: Vector3D) -> bool:
        """Permite a una entidad crear un edificio"""
        entity = self.entity_ecosystem.entities.get(entity_id)
        if not entity:
            return False
        
        # Verificar recursos
        # (Sistema de economía simplificado)
        
        tx = OSCEDTransaction(
            tx_id=hashlib.sha256(f"build_{entity_id}_{time.time()}".encode()).hexdigest(),
            tx_type=TransactionType.EPISTEMIC_CONTRIBUTION,
            type=OSCEDTransactionType.BUILDING_CREATE,
            sender=entity_id,
            data={
                'position': {'x': position.x, 'y': position.y, 'z': position.z},
                'building_type': building_type
            },
            epistemic_vector=ExtendedEpistemicVector({
                'creativity': 0.8,
                'impact': 0.6
            }),
            signature=b"building_signature"
        )
        
        return await self.osced_blockchain.process_entity_transaction(tx)

# === EJEMPLO DE USO ===

async def main():
    """Función principal de demostración"""
    logger.info("=== OSCED Demo - Digital Entity Blockchain with Virtual World ===")
    
    # Crear blockchain SCED principal (opcional)
    sced_main = SCEDBlockchain()
    
    # Crear sistema OSCED integrado
    osced_system = OSCEDIntegratedSystem(
        sced_main=sced_main,
        claude_client=None  # Añadir cliente Claude si está disponible
    )
    
    # Iniciar sistema
    try:
        await osced_system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down OSCED...")
        osced_system.running = False

if __name__ == "__main__":
    asyncio.run(main())
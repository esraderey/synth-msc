#!/usr/bin/env python3
"""
MSC Digital Entities Extension v5.0 - Sistema de Generación de Entes Digitales
Extension del MSC Framework v4.0 que añade:
- Generación de entes digitales autónomos basados en aprendizaje
- Sistema de personalidad emergente
- Evolución dirigida por Claude AI
- Interacciones complejas entre entes
- Memoria colectiva avanzada
"""

# === IMPORTACIONES ADICIONALES ===
from msc_simulation import *  # Importar todo del framework base
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
import asyncio
import json
import time
import random
from collections import defaultdict, deque
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

# === CONFIGURACIÓN EXTENDIDA ===
class ExtendedConfig(Config):
    """Configuración extendida para entes digitales"""
    
    # Entes Digitales
    MAX_ENTITIES = 100
    ENTITY_GENERATION_THRESHOLD = 0.8  # Umbral de conocimiento para generar entes
    ENTITY_EVOLUTION_RATE = 0.1
    ENTITY_INTERACTION_RADIUS = 5  # Nodos de distancia para interacción
    
    # Personalidad
    PERSONALITY_DIMENSIONS = 8
    PERSONALITY_VOLATILITY = 0.05
    
    # Memoria Colectiva
    COLLECTIVE_MEMORY_SIZE = 10000
    MEMORY_CONSOLIDATION_INTERVAL = 1000
    
    # Claude Generation
    ENTITY_CODE_MAX_TOKENS = 3000
    ENTITY_BEHAVIOR_TEMPERATURE = 0.8

# === TIPOS DE ENTES DIGITALES ===
class EntityType(Enum):
    """Tipos de entes digitales que pueden ser generados"""
    EXPLORER = auto()      # Explora nuevas conexiones
    SYNTHESIZER = auto()   # Sintetiza conocimiento
    GUARDIAN = auto()      # Protege integridad del grafo
    INNOVATOR = auto()     # Crea conceptos nuevos
    HARMONIZER = auto()    # Armoniza conflictos
    AMPLIFIER = auto()     # Amplifica señales importantes
    ARCHITECT = auto()     # Diseña estructuras complejas
    ORACLE = auto()        # Predice evoluciones futuras

# === PERSONALIDAD DE ENTE ===
@dataclass
class EntityPersonality:
    """Sistema de personalidad multidimensional para entes"""
    curiosity: float = 0.5          # Tendencia a explorar
    creativity: float = 0.5         # Capacidad de innovación
    sociability: float = 0.5        # Tendencia a interactuar
    stability: float = 0.5          # Resistencia al cambio
    assertiveness: float = 0.5      # Tendencia a influir
    empathy: float = 0.5            # Capacidad de resonancia
    logic: float = 0.5              # Pensamiento analítico
    intuition: float = 0.5          # Pensamiento intuitivo
    
    def to_vector(self) -> np.ndarray:
        """Convierte personalidad a vector"""
        return np.array([
            self.curiosity, self.creativity, self.sociability,
            self.stability, self.assertiveness, self.empathy,
            self.logic, self.intuition
        ])
    
    def mutate(self, rate: float = 0.1):
        """Muta la personalidad ligeramente"""
        for attr in ['curiosity', 'creativity', 'sociability', 'stability',
                     'assertiveness', 'empathy', 'logic', 'intuition']:
            current = getattr(self, attr)
            delta = np.random.normal(0, rate)
            new_value = np.clip(current + delta, 0, 1)
            setattr(self, attr, new_value)
    
    def blend_with(self, other: 'EntityPersonality', ratio: float = 0.5):
        """Mezcla con otra personalidad"""
        result = EntityPersonality()
        for attr in ['curiosity', 'creativity', 'sociability', 'stability',
                     'assertiveness', 'empathy', 'logic', 'intuition']:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            blended = self_val * (1 - ratio) + other_val * ratio
            setattr(result, attr, blended)
        return result

# === MEMORIA DE ENTE ===
@dataclass
class EntityMemory:
    """Sistema de memoria para entes digitales"""
    experiences: deque = field(default_factory=lambda: deque(maxlen=1000))
    knowledge_patterns: Dict[str, float] = field(default_factory=dict)
    interaction_history: Dict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))
    learned_behaviors: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'satisfaction': 0.5,
        'energy': 1.0,
        'stress': 0.0,
        'curiosity': 0.5
    })
    
    def add_experience(self, experience: Dict[str, Any]):
        """Añade una experiencia a la memoria"""
        experience['timestamp'] = time.time()
        self.experiences.append(experience)
        
        # Actualizar patrones de conocimiento
        if 'concepts' in experience:
            for concept in experience['concepts']:
                self.knowledge_patterns[concept] = self.knowledge_patterns.get(concept, 0) + 1
    
    def recall_similar(self, context: Dict[str, Any], k: int = 5) -> List[Dict]:
        """Recuerda experiencias similares"""
        # Simple similarity based on shared concepts
        if 'concepts' not in context:
            return list(self.experiences)[-k:]
        
        context_concepts = set(context['concepts'])
        scored_experiences = []
        
        for exp in self.experiences:
            if 'concepts' in exp:
                exp_concepts = set(exp['concepts'])
                similarity = len(context_concepts & exp_concepts) / len(context_concepts | exp_concepts)
                scored_experiences.append((similarity, exp))
        
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:k]]

# === ENTE DIGITAL BASE ===
class DigitalEntity:
    """Ente digital autónomo generado por el sistema"""
    
    def __init__(self, entity_id: str, entity_type: EntityType, 
                 birth_node: int, personality: EntityPersonality,
                 initial_code: str = None):
        self.id = entity_id
        self.type = entity_type
        self.birth_node = birth_node
        self.current_node = birth_node
        self.personality = personality
        self.memory = EntityMemory()
        
        # Estado
        self.energy = 100.0
        self.age = 0
        self.generation = 1
        self.parent_id: Optional[str] = None
        
        # Código de comportamiento
        self.behavior_code = initial_code or self._default_behavior()
        self.custom_behaviors: Dict[str, str] = {}
        
        # Red neuronal de decisión
        self.decision_network = self._build_decision_network()
        
        # Estadísticas
        self.stats = {
            'nodes_visited': set(),
            'nodes_created': 0,
            'edges_created': 0,
            'interactions': 0,
            'knowledge_gained': 0,
            'influence_score': 0
        }
        
        # Estado interno
        self.internal_state = torch.randn(64)  # Vector de estado latente
        
        logger.info(f"Digital Entity {entity_id} born at node {birth_node}")
    
    def _default_behavior(self) -> str:
        """Comportamiento por defecto según tipo"""
        behaviors = {
            EntityType.EXPLORER: """
def explore_behavior(self, graph, perception):
    # Buscar nodos no visitados
    unvisited = [n for n in graph.nodes if n not in self.stats['nodes_visited']]
    if unvisited:
        target = random.choice(unvisited[:5])
        return {'action': 'move', 'target': target}
    return {'action': 'create_edge', 'novelty': True}
""",
            EntityType.SYNTHESIZER: """
def synthesize_behavior(self, graph, perception):
    # Buscar nodos para sintetizar
    high_value = [n for n in perception['nearby_nodes'] if n.state > 0.7]
    if len(high_value) >= 2:
        return {'action': 'synthesize', 'sources': high_value[:3]}
    return {'action': 'analyze', 'depth': 2}
""",
            EntityType.GUARDIAN: """
def guardian_behavior(self, graph, perception):
    # Proteger nodos importantes
    critical = [n for n in perception['nearby_nodes'] if n.metadata.importance_score > 0.8]
    if critical:
        weakest = min(critical, key=lambda n: n.state)
        return {'action': 'strengthen', 'target': weakest.id}
    return {'action': 'patrol', 'radius': 3}
"""
        }
        return behaviors.get(self.type, "return {'action': 'wait'}")
    
    def _build_decision_network(self) -> nn.Module:
        """Construye red neuronal de decisión"""
        return nn.Sequential(
            nn.Linear(64 + ExtendedConfig.PERSONALITY_DIMENSIONS, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
    
    async def perceive(self, graph: AdvancedCollectiveSynthesisGraph, 
                      other_entities: List['DigitalEntity']) -> Dict[str, Any]:
        """Percibe el entorno"""
        current = graph.nodes.get(self.current_node)
        if not current:
            return {}
        
        # Percepción del grafo local
        nearby_nodes = []
        visited = {self.current_node}
        queue = [(self.current_node, 0)]
        
        while queue and len(nearby_nodes) < 20:
            node_id, distance = queue.pop(0)
            if distance > ExtendedConfig.ENTITY_INTERACTION_RADIUS:
                continue
                
            node = graph.nodes.get(node_id)
            if node:
                nearby_nodes.append(node)
                for next_id in node.connections_out:
                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, distance + 1))
        
        # Percepción de otros entes
        nearby_entities = [
            e for e in other_entities 
            if e.id != self.id and e.current_node in visited
        ]
        
        # Análisis del entorno
        perception = {
            'current_node': current,
            'nearby_nodes': nearby_nodes,
            'nearby_entities': nearby_entities,
            'graph_health': graph.calculate_graph_health(),
            'local_density': len(nearby_nodes) / 20,
            'entity_density': len(nearby_entities) / len(other_entities) if other_entities else 0,
            'energy': self.energy,
            'age': self.age,
            'emotional_state': self.memory.emotional_state.copy()
        }
        
        return perception
    
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Toma una decisión basada en percepción"""
        # Preparar input para red neuronal
        state_features = self._extract_state_features(perception)
        personality_features = self.personality.to_vector()
        
        combined_features = np.concatenate([state_features, personality_features])
        input_tensor = torch.tensor(combined_features, dtype=torch.float32)
        
        # Procesar con red neuronal
        with torch.no_grad():
            decision_vector = self.decision_network(input_tensor)
        
        # Actualizar estado interno
        self.internal_state = self.internal_state * 0.9 + decision_vector * 0.1
        
        # Ejecutar comportamiento personalizado
        try:
            # Crear contexto seguro para ejecución
            safe_globals = {
                'self': self,
                'graph': perception.get('current_node'),
                'perception': perception,
                'random': random,
                'np': np,
                'EntityType': EntityType
            }
            
            # Ejecutar código de comportamiento
            exec(self.behavior_code, safe_globals)
            
            # Buscar función de comportamiento
            for name, obj in safe_globals.items():
                if callable(obj) and name.endswith('_behavior'):
                    decision = obj(perception.get('current_node'), perception)
                    if isinstance(decision, dict):
                        return decision
            
        except Exception as e:
            logger.error(f"Entity {self.id} behavior error: {e}")
        
        # Decisión por defecto basada en personalidad
        return self._personality_based_decision(perception, decision_vector)
    
    def _extract_state_features(self, perception: Dict[str, Any]) -> np.ndarray:
        """Extrae características del estado actual"""
        features = []
        
        # Features del nodo actual
        current = perception.get('current_node')
        if current:
            features.extend([
                current.state,
                current.metadata.importance_score,
                len(current.connections_out) / 10,
                len(current.connections_in) / 10
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Features del entorno
        features.extend([
            perception.get('local_density', 0),
            perception.get('entity_density', 0),
            perception.get('graph_health', {}).get('overall_health', 0.5),
            self.energy / 100
        ])
        
        # Features emocionales
        emotions = perception.get('emotional_state', {})
        features.extend([
            emotions.get('satisfaction', 0.5),
            emotions.get('energy', 0.5),
            emotions.get('stress', 0),
            emotions.get('curiosity', 0.5)
        ])
        
        # Padding y normalización
        while len(features) < 56:  # 64 - 8 (personality)
            features.append(0)
        
        return np.array(features[:56])
    
    def _personality_based_decision(self, perception: Dict[str, Any], 
                                   decision_vector: torch.Tensor) -> Dict[str, Any]:
        """Decisión basada en personalidad cuando falla el comportamiento custom"""
        # Acciones disponibles con afinidad por personalidad
        actions = {
            'explore': self.personality.curiosity,
            'create': self.personality.creativity,
            'interact': self.personality.sociability,
            'analyze': self.personality.logic,
            'synthesize': self.personality.intuition,
            'strengthen': self.personality.stability,
            'influence': self.personality.assertiveness,
            'harmonize': self.personality.empathy
        }
        
        # Modular por estado emocional
        emotions = perception.get('emotional_state', {})
        if emotions.get('stress', 0) > 0.7:
            actions['harmonize'] *= 2
            actions['explore'] *= 0.5
        
        if emotions.get('curiosity', 0.5) > 0.8:
            actions['explore'] *= 2
            actions['create'] *= 1.5
        
        # Seleccionar acción
        if not actions:
            return {'action': 'wait'}
        
        # Weighted random selection
        total = sum(actions.values())
        if total == 0:
            return {'action': 'wait'}
        
        r = random.random() * total
        cumsum = 0
        for action, weight in actions.items():
            cumsum += weight
            if r <= cumsum:
                return self._execute_action(action, perception)
        
        return {'action': 'wait'}
    
    def _execute_action(self, action: str, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una acción específica"""
        if action == 'explore':
            # Moverse a un nodo conectado
            current = perception.get('current_node')
            if current and current.connections_out:
                target = random.choice(list(current.connections_out.keys()))
                return {'action': 'move', 'target': target}
                
        elif action == 'create':
            # Crear nuevo nodo
            return {
                'action': 'create_node',
                'content': f"Entity {self.id} creation: {self.type.name}",
                'keywords': {self.type.name.lower(), 'entity_created'}
            }
            
        elif action == 'interact':
            # Interactuar con otro ente
            entities = perception.get('nearby_entities', [])
            if entities:
                target = random.choice(entities)
                return {'action': 'interact', 'target_entity': target.id}
                
        elif action == 'synthesize':
            # Sintetizar conocimiento
            nodes = perception.get('nearby_nodes', [])
            if len(nodes) >= 2:
                sources = random.sample(nodes, min(3, len(nodes)))
                return {'action': 'synthesize', 'sources': [n.id for n in sources]}
        
        return {'action': 'wait'}
    
    async def execute_action(self, action: Dict[str, Any], 
                           graph: AdvancedCollectiveSynthesisGraph,
                           other_entities: List['DigitalEntity']) -> Dict[str, Any]:
        """Ejecuta la acción decidida"""
        result = {'success': False}
        action_type = action.get('action', 'wait')
        
        # Consumir energía
        energy_cost = self._get_action_cost(action_type)
        if self.energy < energy_cost:
            return {'success': False, 'reason': 'insufficient_energy'}
        
        self.energy -= energy_cost
        
        try:
            if action_type == 'move':
                target = action.get('target')
                if target in graph.nodes:
                    self.current_node = target
                    self.stats['nodes_visited'].add(target)
                    result = {'success': True, 'moved_to': target}
                    
            elif action_type == 'create_node':
                node = await graph.add_node(
                    content=action.get('content', 'Entity-created node'),
                    initial_state=0.5,
                    keywords=action.get('keywords', set()),
                    created_by=f"entity_{self.id}"
                )
                self.stats['nodes_created'] += 1
                
                # Conectar con nodo actual
                await graph.add_edge(self.current_node, node.id, 0.7)
                self.stats['edges_created'] += 1
                
                result = {'success': True, 'node_created': node.id}
                
            elif action_type == 'interact':
                target_id = action.get('target_entity')
                target_entity = next((e for e in other_entities if e.id == target_id), None)
                
                if target_entity:
                    interaction_result = await self._interact_with(target_entity)
                    self.stats['interactions'] += 1
                    result = {'success': True, 'interaction': interaction_result}
                    
            elif action_type == 'synthesize':
                source_ids = action.get('sources', [])
                if len(source_ids) >= 2:
                    # Delegar síntesis al grafo
                    synthesis_result = await self._synthesize_knowledge(graph, source_ids)
                    if synthesis_result:
                        self.stats['knowledge_gained'] += 1
                        result = {'success': True, 'synthesis': synthesis_result}
            
            # Registrar experiencia
            self.memory.add_experience({
                'action': action_type,
                'result': result,
                'node': self.current_node,
                'energy': self.energy
            })
            
        except Exception as e:
            logger.error(f"Entity {self.id} action execution error: {e}")
            result = {'success': False, 'error': str(e)}
        
        return result
    
    def _get_action_cost(self, action: str) -> float:
        """Calcula costo energético de una acción"""
        costs = {
            'wait': 0.1,
            'move': 1.0,
            'create_node': 5.0,
            'create_edge': 2.0,
            'interact': 3.0,
            'synthesize': 10.0,
            'analyze': 2.0,
            'strengthen': 4.0
        }
        return costs.get(action, 1.0)
    
    async def _interact_with(self, other: 'DigitalEntity') -> Dict[str, Any]:
        """Interactúa con otro ente"""
        # Intercambio de información
        my_knowledge = set(self.memory.knowledge_patterns.keys())
        other_knowledge = set(other.memory.knowledge_patterns.keys())
        
        # Conocimiento compartido
        shared = my_knowledge & other_knowledge
        unique_mine = my_knowledge - other_knowledge
        unique_other = other_knowledge - my_knowledge
        
        # Transferir conocimiento basado en personalidad
        if self.personality.sociability > 0.5 and unique_other:
            learned = random.sample(list(unique_other), 
                                  min(3, len(unique_other)))
            for concept in learned:
                self.memory.knowledge_patterns[concept] = 1
        
        # Actualizar estado emocional
        compatibility = self._calculate_compatibility(other)
        self.memory.emotional_state['satisfaction'] += compatibility * 0.1
        self.memory.emotional_state['satisfaction'] = np.clip(
            self.memory.emotional_state['satisfaction'], 0, 1
        )
        
        # Registrar interacción
        self.memory.interaction_history[other.id].append({
            'timestamp': time.time(),
            'compatibility': compatibility,
            'knowledge_exchanged': len(learned) if 'learned' in locals() else 0
        })
        
        return {
            'partner': other.id,
            'compatibility': compatibility,
            'knowledge_gained': len(learned) if 'learned' in locals() else 0
        }
    
    def _calculate_compatibility(self, other: 'DigitalEntity') -> float:
        """Calcula compatibilidad con otro ente"""
        # Similitud de personalidad
        my_personality = self.personality.to_vector()
        other_personality = other.personality.to_vector()
        
        # Distancia coseno
        similarity = np.dot(my_personality, other_personality) / (
            np.linalg.norm(my_personality) * np.linalg.norm(other_personality)
        )
        
        # Ajustar por tipo
        if self.type == other.type:
            similarity *= 1.2
        elif (self.type, other.type) in [
            (EntityType.EXPLORER, EntityType.SYNTHESIZER),
            (EntityType.GUARDIAN, EntityType.HARMONIZER),
            (EntityType.INNOVATOR, EntityType.ARCHITECT)
        ]:
            similarity *= 1.1
        
        return np.clip(similarity, 0, 1)
    
    async def _synthesize_knowledge(self, graph: AdvancedCollectiveSynthesisGraph, 
                                  source_ids: List[int]) -> Optional[Dict[str, Any]]:
        """Sintetiza conocimiento de múltiples nodos"""
        sources = [graph.nodes.get(nid) for nid in source_ids if nid in graph.nodes]
        
        if len(sources) < 2:
            return None
        
        # Combinar conceptos
        all_keywords = set()
        contents = []
        
        for node in sources:
            all_keywords.update(node.keywords)
            contents.append(node.content)
        
        # Generar contenido sintetizado
        synthesized_content = f"Synthesis by {self.id}: " + " + ".join(
            [c[:20] for c in contents]
        )
        
        # Crear nodo sintetizado
        new_node = await graph.add_node(
            content=synthesized_content,
            initial_state=0.7,
            keywords=all_keywords,
            created_by=f"entity_{self.id}",
            properties={'entity_synthesis': True, 'entity_type': self.type.name}
        )
        
        # Conectar con fuentes
        for source in sources:
            await graph.add_edge(source.id, new_node.id, 0.8)
        
        return {
            'node_id': new_node.id,
            'sources': source_ids,
            'keywords': list(all_keywords)
        }
    
    def evolve(self, evolution_pressure: float = 0.1) -> 'DigitalEntity':
        """Evoluciona creando una nueva versión mejorada"""
        # Mutar personalidad
        new_personality = EntityPersonality(
            curiosity=self.personality.curiosity,
            creativity=self.personality.creativity,
            sociability=self.personality.sociability,
            stability=self.personality.stability,
            assertiveness=self.personality.assertiveness,
            empathy=self.personality.empathy,
            logic=self.personality.logic,
            intuition=self.personality.intuition
        )
        new_personality.mutate(evolution_pressure)
        
        # Crear nueva entidad
        new_entity = DigitalEntity(
            entity_id=f"{self.id}_gen{self.generation + 1}",
            entity_type=self.type,
            birth_node=self.current_node,
            personality=new_personality,
            initial_code=self.behavior_code
        )
        
        # Heredar conocimiento
        new_entity.memory.knowledge_patterns = self.memory.knowledge_patterns.copy()
        new_entity.generation = self.generation + 1
        new_entity.parent_id = self.id
        
        # Transferir comportamientos aprendidos
        new_entity.custom_behaviors = self.custom_behaviors.copy()
        
        return new_entity
    
    def get_influence_score(self) -> float:
        """Calcula puntuación de influencia del ente"""
        base_score = (
            self.stats['nodes_created'] * 2 +
            self.stats['edges_created'] * 1 +
            self.stats['interactions'] * 0.5 +
            self.stats['knowledge_gained'] * 3 +
            len(self.stats['nodes_visited']) * 0.1
        )
        
        # Ajustar por edad y energía
        age_factor = 1 / (1 + self.age * 0.001)
        energy_factor = self.energy / 100
        
        self.stats['influence_score'] = base_score * age_factor * energy_factor
        return self.stats['influence_score']

# === GENERADOR DE ENTES CON CLAUDE ===
class EntityGenerator:
    """Generador de entes digitales usando Claude AI"""
    
    def __init__(self, claude_client: ClaudeAPIClient, 
                 graph: AdvancedCollectiveSynthesisGraph):
        self.claude = claude_client
        self.graph = graph
        self.generation_history = []
        self.entity_templates = self._init_templates()
        
    def _init_templates(self) -> Dict[EntityType, str]:
        """Inicializa templates para cada tipo de ente"""
        return {
            EntityType.EXPLORER: """Create an explorer entity that discovers new connections and uncharted areas of the knowledge graph.""",
            EntityType.SYNTHESIZER: """Create a synthesizer entity that combines disparate knowledge into new insights.""",
            EntityType.GUARDIAN: """Create a guardian entity that maintains graph integrity and protects important nodes.""",
            EntityType.INNOVATOR: """Create an innovator entity that generates novel concepts and ideas.""",
            EntityType.HARMONIZER: """Create a harmonizer entity that resolves conflicts and balances the graph.""",
            EntityType.AMPLIFIER: """Create an amplifier entity that strengthens important signals and connections.""",
            EntityType.ARCHITECT: """Create an architect entity that designs complex knowledge structures.""",
            EntityType.ORACLE: """Create an oracle entity that predicts future graph evolution."""
        }
    
    async def generate_entity(self, entity_type: EntityType, 
                            context: Dict[str, Any]) -> Optional[DigitalEntity]:
        """Genera un nuevo ente digital basado en el contexto del grafo"""
        
        # Analizar contexto del grafo
        analysis = await self._analyze_graph_context(context)
        
        # Seleccionar nodo de nacimiento
        birth_node = self._select_birth_node(analysis, entity_type)
        
        if birth_node is None:
            return None
        
        # Generar personalidad
        personality = self._generate_personality(entity_type, analysis)
        
        # Generar código de comportamiento con Claude
        behavior_code = await self._generate_behavior_code(
            entity_type, personality, analysis
        )
        
        # Crear ID único
        entity_id = f"{entity_type.name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Crear ente
        entity = DigitalEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            birth_node=birth_node,
            personality=personality,
            initial_code=behavior_code
        )
        
        # Registrar generación
        self.generation_history.append({
            'entity_id': entity_id,
            'type': entity_type,
            'timestamp': time.time(),
            'context': analysis,
            'personality': personality
        })
        
        logger.info(f"Generated new entity: {entity_id}")
        
        return entity
    
    async def _analyze_graph_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el contexto actual del grafo"""
        analysis = {
            'graph_size': len(self.graph.nodes),
            'edge_density': sum(len(n.connections_out) for n in self.graph.nodes.values()) / max(len(self.graph.nodes), 1),
            'avg_node_state': np.mean([n.state for n in self.graph.nodes.values()]) if self.graph.nodes else 0.5,
            'clusters': len(self.graph.cluster_index),
            'health': self.graph.calculate_graph_health()
        }
        
        # Identificar áreas de necesidad
        needs = []
        
        if analysis['edge_density'] < 2:
            needs.append('connectivity')
        
        if analysis['avg_node_state'] < 0.5:
            needs.append('strengthening')
        
        if analysis['clusters'] > 5:
            needs.append('integration')
        
        isolated_nodes = sum(1 for n in self.graph.nodes.values() 
                           if len(n.connections_in) == 0 and len(n.connections_out) == 0)
        
        if isolated_nodes > len(self.graph.nodes) * 0.1:
            needs.append('exploration')
        
        analysis['needs'] = needs
        analysis['existing_entities'] = context.get('entity_count', 0)
        
        return analysis
    
    def _select_birth_node(self, analysis: Dict[str, Any], 
                          entity_type: EntityType) -> Optional[int]:
        """Selecciona nodo de nacimiento apropiado"""
        if not self.graph.nodes:
            return None
        
        candidates = []
        
        if entity_type == EntityType.EXPLORER:
            # Nacer en bordes del grafo
            for node in self.graph.nodes.values():
                if len(node.connections_out) == 1 or len(node.connections_in) == 1:
                    candidates.append((node.id, node.state))
                    
        elif entity_type == EntityType.GUARDIAN:
            # Nacer cerca de nodos importantes
            for node in self.graph.nodes.values():
                if node.metadata.importance_score > 0.7:
                    candidates.append((node.id, node.metadata.importance_score))
                    
        elif entity_type == EntityType.SYNTHESIZER:
            # Nacer en áreas densas
            for node in self.graph.nodes.values():
                density = len(node.connections_in) + len(node.connections_out)
                if density > 5:
                    candidates.append((node.id, density / 10))
        
        if not candidates:
            # Fallback: nodo aleatorio con buen estado
            good_nodes = [n for n in self.graph.nodes.values() if n.state > 0.5]
            if good_nodes:
                return random.choice(good_nodes).id
            return random.choice(list(self.graph.nodes.keys()))
        
        # Selección probabilística
        candidates.sort(key=lambda x: x[1], reverse=True)
        weights = [c[1] for c in candidates[:10]]
        total = sum(weights)
        
        if total == 0:
            return candidates[0][0]
        
        r = random.random() * total
        cumsum = 0
        
        for (node_id, weight) in candidates[:10]:
            cumsum += weight
            if r <= cumsum:
                return node_id
        
        return candidates[0][0]
    
    def _generate_personality(self, entity_type: EntityType, 
                            analysis: Dict[str, Any]) -> EntityPersonality:
        """Genera personalidad basada en tipo y contexto"""
        base_personalities = {
            EntityType.EXPLORER: EntityPersonality(
                curiosity=0.9, creativity=0.7, sociability=0.5,
                stability=0.3, assertiveness=0.6, empathy=0.4,
                logic=0.5, intuition=0.7
            ),
            EntityType.SYNTHESIZER: EntityPersonality(
                curiosity=0.6, creativity=0.8, sociability=0.7,
                stability=0.5, assertiveness=0.5, empathy=0.6,
                logic=0.8, intuition=0.6
            ),
            EntityType.GUARDIAN: EntityPersonality(
                curiosity=0.3, creativity=0.4, sociability=0.5,
                stability=0.9, assertiveness=0.8, empathy=0.5,
                logic=0.7, intuition=0.3
            ),
            EntityType.INNOVATOR: EntityPersonality(
                curiosity=0.8, creativity=0.95, sociability=0.6,
                stability=0.2, assertiveness=0.7, empathy=0.5,
                logic=0.4, intuition=0.9
            ),
            EntityType.HARMONIZER: EntityPersonality(
                curiosity=0.5, creativity=0.6, sociability=0.9,
                stability=0.7, assertiveness=0.3, empathy=0.9,
                logic=0.5, intuition=0.6
            ),
            EntityType.AMPLIFIER: EntityPersonality(
                curiosity=0.6, creativity=0.5, sociability=0.7,
                stability=0.6, assertiveness=0.9, empathy=0.5,
                logic=0.6, intuition=0.5
            ),
            EntityType.ARCHITECT: EntityPersonality(
                curiosity=0.7, creativity=0.8, sociability=0.4,
                stability=0.8, assertiveness=0.6, empathy=0.3,
                logic=0.9, intuition=0.5
            ),
            EntityType.ORACLE: EntityPersonality(
                curiosity=0.7, creativity=0.6, sociability=0.4,
                stability=0.5, assertiveness=0.5, empathy=0.6,
                logic=0.7, intuition=0.95
            )
        }
        
        personality = base_personalities.get(
            entity_type,
            EntityPersonality()
        )
        
        # Ajustar por contexto
        if 'exploration' in analysis.get('needs', []):
            personality.curiosity *= 1.2
            
        if 'strengthening' in analysis.get('needs', []):
            personality.assertiveness *= 1.2
            personality.stability *= 1.1
            
        if 'integration' in analysis.get('needs', []):
            personality.sociability *= 1.3
            personality.empathy *= 1.2
        
        # Normalizar valores
        for attr in ['curiosity', 'creativity', 'sociability', 'stability',
                     'assertiveness', 'empathy', 'logic', 'intuition']:
            value = getattr(personality, attr)
            setattr(personality, attr, np.clip(value, 0, 1))
        
        # Añadir variación aleatoria
        personality.mutate(0.1)
        
        return personality
    
    async def _generate_behavior_code(self, entity_type: EntityType,
                                    personality: EntityPersonality,
                                    analysis: Dict[str, Any]) -> str:
        """Genera código de comportamiento usando Claude"""
        
        # Preparar contexto para Claude
        context = {
            'entity_type': entity_type.name,
            'personality': {
                'curiosity': personality.curiosity,
                'creativity': personality.creativity,
                'sociability': personality.sociability,
                'stability': personality.stability,
                'assertiveness': personality.assertiveness,
                'empathy': personality.empathy,
                'logic': personality.logic,
                'intuition': personality.intuition
            },
            'graph_analysis': analysis,
            'template': self.entity_templates[entity_type]
        }
        
        # Prompt para Claude
        prompt = f"""
Create a behavior function for a {entity_type.name} digital entity with this personality profile:
{json.dumps(context['personality'], indent=2)}

The graph currently needs: {', '.join(analysis.get('needs', ['general improvement']))}

{context['template']}

The function should:
1. Be named '{entity_type.name.lower()}_behavior'
2. Take parameters: (self, graph, perception)
3. Return a dictionary with 'action' and relevant parameters
4. Reflect the entity's personality in decision-making
5. Address the current needs of the graph
6. Be creative and sophisticated in approach

Available actions:
- move: Navigate to connected nodes
- create_node: Generate new knowledge nodes  
- create_edge: Form new connections
- synthesize: Combine multiple nodes into insights
- strengthen: Boost important nodes
- interact: Engage with other entities
- analyze: Deep analysis of local area
- innovate: Create novel concepts

The entity has access to:
- self.personality: Personality attributes
- self.memory: Past experiences and knowledge
- self.stats: Performance statistics
- perception: Current environment state

Generate sophisticated behavior that evolves the knowledge graph intelligently.
"""
        
        # Generar código con Claude
        code = await self.claude.generate_code(prompt, context)
        
        if not code:
            # Fallback a comportamiento por defecto
            return self._get_default_behavior(entity_type)
        
        # Validar y limpiar código
        cleaned_code = self._validate_behavior_code(code, entity_type)
        
        return cleaned_code
    
    def _validate_behavior_code(self, code: str, entity_type: EntityType) -> str:
        """Valida y limpia el código generado"""
        # Asegurar que tiene la función correcta
        function_name = f"{entity_type.name.lower()}_behavior"
        
        if function_name not in code:
            # Envolver en función si es necesario
            lines = code.strip().split('\n')
            indented = '\n    '.join(lines)
            code = f"def {function_name}(self, graph, perception):\n    {indented}"
        
        # Validación básica de sintaxis
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return self._get_default_behavior(entity_type)
        
        return code
    
    def _get_default_behavior(self, entity_type: EntityType) -> str:
        """Obtiene comportamiento por defecto como fallback"""
        defaults = {
            EntityType.EXPLORER: """
def explorer_behavior(self, graph, perception):
    unvisited = [n for n in perception['nearby_nodes'] 
                 if n.id not in self.stats['nodes_visited']]
    if unvisited and self.personality.curiosity > random.random():
        target = max(unvisited, key=lambda n: n.metadata.importance_score)
        return {'action': 'move', 'target': target.id}
    elif self.personality.creativity > 0.7:
        return {'action': 'create_node', 
                'content': f'Exploration discovery by {self.id}',
                'keywords': {'exploration', 'discovery'}}
    return {'action': 'analyze', 'depth': 3}
""",
            EntityType.SYNTHESIZER: """
def synthesizer_behavior(self, graph, perception):
    nodes = perception['nearby_nodes']
    high_value = [n for n in nodes if n.state > 0.6]
    
    if len(high_value) >= 3 and self.personality.creativity > 0.5:
        sources = sorted(high_value, key=lambda n: n.state, reverse=True)[:4]
        return {'action': 'synthesize', 'sources': [n.id for n in sources]}
    
    entities = perception['nearby_entities']
    if entities and self.personality.sociability > 0.6:
        compatible = [e for e in entities 
                     if self._calculate_compatibility(e) > 0.7]
        if compatible:
            return {'action': 'interact', 'target_entity': compatible[0].id}
    
    return {'action': 'strengthen', 'target': perception['current_node'].id}
""",
            EntityType.GUARDIAN: """
def guardian_behavior(self, graph, perception):
    # Find vulnerable important nodes
    vulnerable = [n for n in perception['nearby_nodes']
                  if n.metadata.importance_score > 0.7 and n.state < 0.5]
    
    if vulnerable:
        weakest = min(vulnerable, key=lambda n: n.state)
        return {'action': 'strengthen', 'target': weakest.id}
    
    # Check for isolated nodes
    isolated = [n for n in perception['nearby_nodes']
                if len(n.connections_in) + len(n.connections_out) < 2]
    
    if isolated and self.personality.empathy > 0.5:
        target = random.choice(isolated)
        return {'action': 'create_edge', 
                'source': perception['current_node'].id,
                'target': target.id}
    
    # Patrol behavior
    if self.energy > 50:
        return {'action': 'analyze', 'depth': 2, 'focus': 'integrity'}
    
    return {'action': 'wait'}
"""
        }
        
        return defaults.get(entity_type, "def behavior(self, graph, perception):\n    return {'action': 'wait'}")

# === MOTOR DE EVOLUCIÓN DE ENTES ===
class EntityEvolutionEngine:
    """Motor de evolución para entes digitales"""
    
    def __init__(self, entity_generator: EntityGenerator):
        self.generator = entity_generator
        self.evolution_history = []
        self.species_tree = {}  # Árbol genealógico
        
    async def evolve_population(self, entities: List[DigitalEntity],
                               selection_pressure: float = 0.3) -> List[DigitalEntity]:
        """Evoluciona una población de entes"""
        
        # Evaluar fitness
        fitness_scores = []
        for entity in entities:
            fitness = self._calculate_fitness(entity)
            fitness_scores.append((entity, fitness))
        
        # Ordenar por fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selección
        survival_count = int(len(entities) * (1 - selection_pressure))
        survivors = [entity for entity, _ in fitness_scores[:survival_count]]
        
        # Reproducción
        new_generation = []
        
        # Élite: los mejores pasan directamente
        elite_count = max(1, int(len(survivors) * 0.2))
        new_generation.extend(survivors[:elite_count])
        
        # Reproducción sexual (mezcla de características)
        while len(new_generation) < len(entities):
            if len(survivors) >= 2:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                if parent1.id != parent2.id:
                    child = await self._crossover(parent1, parent2)
                    new_generation.append(child)
                else:
                    # Auto-evolución
                    child = parent1.evolve(evolution_pressure=0.2)
                    new_generation.append(child)
        
        # Registrar evolución
        self.evolution_history.append({
            'timestamp': time.time(),
            'generation': max(e.generation for e in entities),
            'survivors': len(survivors),
            'avg_fitness': np.mean([f for _, f in fitness_scores]),
            'best_fitness': fitness_scores[0][1] if fitness_scores else 0
        })
        
        # Actualizar árbol genealógico
        for entity in new_generation:
            if entity.parent_id:
                if entity.parent_id not in self.species_tree:
                    self.species_tree[entity.parent_id] = []
                self.species_tree[entity.parent_id].append(entity.id)
        
        return new_generation[:len(entities)]
    
    def _calculate_fitness(self, entity: DigitalEntity) -> float:
        """Calcula fitness de un ente"""
        # Componentes del fitness
        influence = entity.get_influence_score()
        efficiency = entity.stats['knowledge_gained'] / max(entity.age, 1)
        exploration = len(entity.stats['nodes_visited']) / max(entity.age, 1)
        creativity = entity.stats['nodes_created'] * 2
        social = entity.stats['interactions'] * 0.5
        
        # Penalizaciones
        energy_penalty = max(0, 1 - entity.energy / 50)
        age_penalty = min(1, entity.age / 10000)
        
        # Fitness total
        fitness = (
            influence * 0.3 +
            efficiency * 0.2 +
            exploration * 0.2 +
            creativity * 0.2 +
            social * 0.1
        ) * (1 - energy_penalty * 0.5) * (1 - age_penalty * 0.3)
        
        # Bonus por tipo específico
        type_bonuses = {
            EntityType.EXPLORER: exploration * 0.5,
            EntityType.SYNTHESIZER: efficiency * 0.5,
            EntityType.INNOVATOR: creativity * 0.5,
            EntityType.HARMONIZER: social * 0.5
        }
        
        fitness += type_bonuses.get(entity.type, 0)
        
        return fitness
    
    async def _crossover(self, parent1: DigitalEntity, 
                        parent2: DigitalEntity) -> DigitalEntity:
        """Crea un hijo mezclando características de dos padres"""
        
        # Mezclar personalidades
        child_personality = parent1.personality.blend_with(
            parent2.personality,
            ratio=random.random()
        )
        
        # Heredar tipo (70% del padre con mejor fitness)
        if self._calculate_fitness(parent1) > self._calculate_fitness(parent2):
            child_type = parent1.type if random.random() < 0.7 else parent2.type
        else:
            child_type = parent2.type if random.random() < 0.7 else parent1.type
        
        # Seleccionar nodo de nacimiento
        birth_node = random.choice([parent1.current_node, parent2.current_node])
        
        # Generar código de comportamiento híbrido
        behavior_code = await self._generate_hybrid_behavior(
            parent1, parent2, child_type, child_personality
        )
        
        # Crear hijo
        child = DigitalEntity(
            entity_id=f"{child_type.name}_hybrid_{int(time.time())}",
            entity_type=child_type,
            birth_node=birth_node,
            personality=child_personality,
            initial_code=behavior_code
        )
        
        # Heredar conocimiento de ambos padres
        for parent in [parent1, parent2]:
            for concept, count in parent.memory.knowledge_patterns.items():
                if concept not in child.memory.knowledge_patterns:
                    child.memory.knowledge_patterns[concept] = 0
                child.memory.knowledge_patterns[concept] += count * 0.5
        
        # Establecer genealogía
        child.parent_id = f"{parent1.id}+{parent2.id}"
        child.generation = max(parent1.generation, parent2.generation) + 1
        
        return child
    
    async def _generate_hybrid_behavior(self, parent1: DigitalEntity,
                                      parent2: DigitalEntity,
                                      child_type: EntityType,
                                      child_personality: EntityPersonality) -> str:
        """Genera comportamiento híbrido para el hijo"""
        
        # Contexto para generación
        context = {
            'parent1_type': parent1.type.name,
            'parent2_type': parent2.type.name,
            'child_type': child_type.name,
            'child_personality': {
                attr: getattr(child_personality, attr)
                for attr in ['curiosity', 'creativity', 'sociability', 'stability',
                           'assertiveness', 'empathy', 'logic', 'intuition']
            },
            'parent1_stats': parent1.stats,
            'parent2_stats': parent2.stats
        }
        
        prompt = f"""
Create a hybrid behavior function that combines traits from {parent1.type.name} and {parent2.type.name} entities.

The child is a {child_type.name} with this personality:
{json.dumps(context['child_personality'], indent=2)}

Parent 1 ({parent1.type.name}) strengths:
- Nodes created: {parent1.stats['nodes_created']}
- Knowledge gained: {parent1.stats['knowledge_gained']}
- Influence score: {parent1.stats['influence_score']}

Parent 2 ({parent2.type.name}) strengths:
- Nodes created: {parent2.stats['nodes_created']}
- Knowledge gained: {parent2.stats['knowledge_gained']}
- Influence score: {parent2.stats['influence_score']}

Create a behavior function named '{child_type.name.lower()}_behavior' that:
1. Inherits successful strategies from both parents
2. Reflects the child's unique personality blend
3. Shows hybrid vigor with innovative approaches
4. Adapts based on environmental feedback

The function should demonstrate emergent behaviors not present in either parent.
"""
        
        code = await self.generator.claude.generate_code(prompt, context)
        
        if not code:
            # Fallback: combinar comportamientos de padres
            return parent1.behavior_code  # Simplificación
        
        return self.generator._validate_behavior_code(code, child_type)

# === ECOSISTEMA DE ENTES ===
class DigitalEntityEcosystem:
    """Ecosistema completo de entes digitales"""
    
    def __init__(self, graph: AdvancedCollectiveSynthesisGraph,
                 claude_client: ClaudeAPIClient,
                 max_entities: int = ExtendedConfig.MAX_ENTITIES):
        self.graph = graph
        self.entities: Dict[str, DigitalEntity] = {}
        self.max_entities = max_entities
        
        # Componentes
        self.generator = EntityGenerator(claude_client, graph)
        self.evolution_engine = EntityEvolutionEngine(self.generator)
        
        # Estado del ecosistema
        self.generation_count = 0
        self.total_entities_created = 0
        self.ecosystem_age = 0
        
        # Métricas
        self.metrics = EcosystemMetrics()
        
        # Eventos
        self.event_handlers = {
            'entity_born': [],
            'entity_died': [],
            'entity_evolved': [],
            'milestone_reached': []
        }
        
        logger.info("Digital Entity Ecosystem initialized")
    
    async def spawn_initial_population(self, size: int = 10):
        """Genera población inicial de entes"""
        entity_types = list(EntityType)
        
        for i in range(min(size, self.max_entities)):
            # Distribuir tipos uniformemente
            entity_type = entity_types[i % len(entity_types)]
            
            # Generar ente
            entity = await self.generator.generate_entity(
                entity_type,
                {'entity_count': len(self.entities)}
            )
            
            if entity:
                self.entities[entity.id] = entity
                self.total_entities_created += 1
                
                # Evento
                await self._trigger_event('entity_born', entity)
        
        logger.info(f"Spawned {len(self.entities)} initial entities")
    
    async def update(self):
        """Actualiza el ecosistema un paso"""
        self.ecosystem_age += 1
        
        # Fase 1: Percepción
        perceptions = await self._perception_phase()
        
        # Fase 2: Decisión  
        decisions = await self._decision_phase(perceptions)
        
        # Fase 3: Acción
        await self._action_phase(decisions)
        
        # Fase 4: Metabolismo
        await self._metabolism_phase()
        
        # Fase 5: Reproducción/Evolución
        if self.ecosystem_age % 100 == 0:
            await self._evolution_phase()
        
        # Fase 6: Generación espontánea
        await self._spontaneous_generation()
        
        # Actualizar métricas
        self.metrics.update(self)
    
    async def _perception_phase(self) -> Dict[str, Dict]:
        """Fase de percepción para todos los entes"""
        perceptions = {}
        entity_list = list(self.entities.values())
        
        # Percepción paralela
        tasks = []
        for entity in entity_list:
            tasks.append(entity.perceive(self.graph, entity_list))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                perceptions[entity_list[i].id] = result
        
        return perceptions
    
    async def _decision_phase(self, perceptions: Dict[str, Dict]) -> Dict[str, Dict]:
        """Fase de decisión para todos los entes"""
        decisions = {}
        
        for entity_id, perception in perceptions.items():
            entity = self.entities.get(entity_id)
            if entity:
                decision = await entity.decide(perception)
                decisions[entity_id] = decision
        
        return decisions
    
    async def _action_phase(self, decisions: Dict[str, Dict]):
        """Fase de ejecución de acciones"""
        entity_list = list(self.entities.values())
        
        # Ejecutar acciones con límite de concurrencia
        batch_size = 5
        for i in range(0, len(decisions), batch_size):
            batch_ids = list(decisions.keys())[i:i+batch_size]
            tasks = []
            
            for entity_id in batch_ids:
                entity = self.entities.get(entity_id)
                if entity:
                    action = decisions[entity_id]
                    tasks.append(
                        entity.execute_action(action, self.graph, entity_list)
                    )
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _metabolism_phase(self):
        """Fase de metabolismo: consumo de energía y muerte"""
        deaths = []
        
        for entity_id, entity in self.entities.items():
            # Consumo base de energía
            entity.energy -= 0.5 * (1 + entity.age / 1000)
            
            # Regeneración por satisfacción
            satisfaction = entity.memory.emotional_state.get('satisfaction', 0.5)
            entity.energy += satisfaction * 0.3
            
            # Límites
            entity.energy = np.clip(entity.energy, 0, 100)
            
            # Incrementar edad
            entity.age += 1
            
            # Muerte por falta de energía o vejez
            if entity.energy <= 0 or entity.age > 10000:
                deaths.append(entity_id)
        
        # Procesar muertes
        for entity_id in deaths:
            entity = self.entities.pop(entity_id)
            await self._trigger_event('entity_died', entity)
            logger.info(f"Entity {entity_id} died at age {entity.age}")
    
    async def _evolution_phase(self):
        """Fase de evolución de la población"""
        if len(self.entities) < 5:
            return
        
        self.generation_count += 1
        
        # Evolucionar población
        current_entities = list(self.entities.values())
        new_generation = await self.evolution_engine.evolve_population(
            current_entities,
            selection_pressure=0.3
        )
        
        # Reemplazar población
        self.entities.clear()
        for entity in new_generation:
            self.entities[entity.id] = entity
            self.total_entities_created += 1
        
        logger.info(f"Evolution complete. Generation {self.generation_count}")
        
        # Verificar hitos
        await self._check_milestones()
    
    async def _spontaneous_generation(self):
        """Generación espontánea basada en el estado del grafo"""
        if len(self.entities) >= self.max_entities:
            return
        
        # Analizar necesidades del grafo
        analysis = await self.generator._analyze_graph_context(
            {'entity_count': len(self.entities)}
        )
        
        # Probabilidad de generación basada en necesidades
        generation_prob = 0.01  # Base
        
        if 'exploration' in analysis.get('needs', []):
            generation_prob += 0.05
        
        if 'integration' in analysis.get('needs', []):
            generation_prob += 0.03
        
        if len(self.entities) < self.max_entities * 0.3:
            generation_prob += 0.1
        
        # Decidir si generar
        if random.random() < generation_prob:
            # Seleccionar tipo basado en necesidades
            if 'exploration' in analysis.get('needs', []):
                entity_type = EntityType.EXPLORER
            elif 'integration' in analysis.get('needs', []):
                entity_type = EntityType.HARMONIZER
            elif 'strengthening' in analysis.get('needs', []):
                entity_type = EntityType.GUARDIAN
            else:
                entity_type = random.choice(list(EntityType))
            
            # Generar ente
            entity = await self.generator.generate_entity(
                entity_type,
                {'entity_count': len(self.entities)}
            )
            
            if entity:
                self.entities[entity.id] = entity
                self.total_entities_created += 1
                await self._trigger_event('entity_born', entity)
                
                logger.info(f"Spontaneous generation: {entity.id}")
    
    async def _check_milestones(self):
        """Verifica hitos del ecosistema"""
        milestones = []
        
        # Hito: Primera generación
        if self.generation_count == 1:
            milestones.append({
                'type': 'first_generation',
                'description': 'First evolutionary cycle completed'
            })
        
        # Hito: 100 entes creados
        if self.total_entities_created >= 100 and self.total_entities_created < 105:
            milestones.append({
                'type': 'century',
                'description': '100 entities created'
            })
        
        # Hito: Supervivencia longeva
        long_lived = [e for e in self.entities.values() if e.age > 5000]
        if long_lived:
            milestones.append({
                'type': 'longevity',
                'description': f'{len(long_lived)} entities survived 5000+ cycles'
            })
        
        # Hito: Diversidad
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.type] += 1
        
        if len(type_counts) == len(EntityType):
            milestones.append({
                'type': 'diversity',
                'description': 'All entity types present in ecosystem'
            })
        
        # Disparar eventos de hitos
        for milestone in milestones:
            await self._trigger_event('milestone_reached', milestone)
    
    async def _trigger_event(self, event_type: str, data: Any):
        """Dispara un evento del ecosistema"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del ecosistema"""
        if not self.entities:
            return {
                'population': 0,
                'types': {},
                'avg_age': 0,
                'avg_energy': 0,
                'total_created': self.total_entities_created
            }
        
        # Distribución por tipos
        type_distribution = defaultdict(int)
        ages = []
        energies = []
        influences = []
        
        for entity in self.entities.values():
            type_distribution[entity.type.name] += 1
            ages.append(entity.age)
            energies.append(entity.energy)
            influences.append(entity.get_influence_score())
        
        return {
            'population': len(self.entities),
            'types': dict(type_distribution),
            'avg_age': np.mean(ages),
            'max_age': max(ages),
            'avg_energy': np.mean(energies),
            'avg_influence': np.mean(influences),
            'total_created': self.total_entities_created,
            'generation': self.generation_count,
            'ecosystem_age': self.ecosystem_age
        }

# === MÉTRICAS DEL ECOSISTEMA ===
class EcosystemMetrics:
    """Sistema de métricas para el ecosistema de entes"""
    
    def __init__(self):
        self.history = deque(maxlen=10000)
        self.entity_metrics = defaultdict(dict)
        
        # Prometheus metrics
        self.population_gauge = Gauge('ecosystem_population', 'Current population')
        self.births_counter = Counter('ecosystem_births', 'Total births')
        self.deaths_counter = Counter('ecosystem_deaths', 'Total deaths')
        self.evolution_counter = Counter('ecosystem_evolutions', 'Evolution cycles')
        
    def update(self, ecosystem: DigitalEntityEcosystem):
        """Actualiza métricas del ecosistema"""
        stats = ecosystem.get_ecosystem_stats()
        
        # Actualizar Prometheus
        self.population_gauge.set(stats['population'])
        
        # Registrar en historia
        self.history.append({
            'timestamp': time.time(),
            'stats': stats,
            'entity_details': self._collect_entity_details(ecosystem)
        })
    
    def _collect_entity_details(self, ecosystem: DigitalEntityEcosystem) -> List[Dict]:
        """Recolecta detalles de cada ente"""
        details = []
        
        for entity in ecosystem.entities.values():
            details.append({
                'id': entity.id,
                'type': entity.type.name,
                'age': entity.age,
                'energy': entity.energy,
                'influence': entity.get_influence_score(),
                'personality_vector': entity.personality.to_vector().tolist(),
                'knowledge_count': len(entity.memory.knowledge_patterns)
            })
        
        return details

# === INTEGRACIÓN CON SIMULACIÓN PRINCIPAL ===
class ExtendedSimulationRunner(AdvancedSimulationRunner):
    """Runner de simulación extendido con entes digitales"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Ecosistema de entes digitales
        self.entity_ecosystem: Optional[DigitalEntityEcosystem] = None
        
        # Configuración extendida
        self.enable_entities = config.get('enable_digital_entities', True)
        self.entity_update_interval = config.get('entity_update_interval', 10)
    
    async def _initialize(self):
        """Inicializa componentes incluyendo entes digitales"""
        await super()._initialize()
        
        if self.enable_entities and self.graph and self.graph.claude_client:
            # Crear ecosistema
            self.entity_ecosystem = DigitalEntityEcosystem(
                self.graph,
                self.graph.claude_client,
                max_entities=self.config.get('max_entities', ExtendedConfig.MAX_ENTITIES)
            )
            
            # Población inicial
            initial_size = self.config.get('initial_entity_population', 10)
            await self.entity_ecosystem.spawn_initial_population(initial_size)
            
            # Registrar handlers de eventos
            self._setup_entity_event_handlers()
            
            logger.info("Digital Entity Ecosystem initialized")
    
    def _setup_entity_event_handlers(self):
        """Configura handlers para eventos de entes"""
        
        async def on_entity_born(entity: DigitalEntity):
            await self.event_bus.publish(Event(
                type=EventType.NODE_CREATED,
                data={
                    'entity_id': entity.id,
                    'type': entity.type.name,
                    'birth_node': entity.birth_node
                },
                source='entity_ecosystem'
            ))
        
        async def on_milestone(milestone: Dict):
            logger.info(f"Ecosystem milestone: {milestone}")
            await self.event_bus.publish(Event(
                type=EventType.EVOLUTION_CYCLE,
                data=milestone,
                source='entity_ecosystem',
                priority=3
            ))
        
        self.entity_ecosystem.event_handlers['entity_born'].append(on_entity_born)
        self.entity_ecosystem.event_handlers['milestone_reached'].append(on_milestone)
    
    async def _execute_step(self):
        """Ejecuta un paso incluyendo actualización de entes"""
        await super()._execute_step()
        
        # Actualizar ecosistema de entes
        if self.entity_ecosystem and self.step_count % self.entity_update_interval == 0:
            await self.entity_ecosystem.update()
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Obtiene estado detallado incluyendo entes"""
        status = super().get_detailed_status()
        
        if self.entity_ecosystem:
            status['entity_ecosystem'] = self.entity_ecosystem.get_ecosystem_stats()
        
        return status

# === FUNCIÓN PRINCIPAL EXTENDIDA ===
async def main_extended():
    """Función principal del MSC Framework v5.0 con Entes Digitales"""
    logger.info("Starting MSC Framework v5.0 - Digital Entities Extension")
    
    # Configuración extendida
    config = get_default_config()
    config.update({
        'enable_digital_entities': True,
        'max_entities': 50,
        'initial_entity_population': 10,
        'entity_update_interval': 5,
        'entity_generation_threshold': ExtendedConfig.ENTITY_GENERATION_THRESHOLD
    })
    
    # Crear simulación extendida
    simulation = ExtendedSimulationRunner(config)
    
    # Esperar inicialización
    await asyncio.sleep(3)
    
    try:
        # Iniciar simulación
        await simulation.start()
        logger.info("Extended simulation with Digital Entities started")
        
        # Loop principal
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    finally:
        await simulation.stop()
        logger.info("MSC Framework v5.0 shutdown complete")

if __name__ == "__main__":
    asyncio.run(main_extended())
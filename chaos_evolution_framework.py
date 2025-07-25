#!/usr/bin/env python3
"""
CHAOS EVOLUTION FRAMEWORK (CEF) v1.0 - Gemelo Caótico del MSC
Marco del Caos Evolutivo de Inteligencias Artificiales
Integración con MSC Framework v4.0

Características principales:
- Matemática pura del caos y sistemas dinámicos
- Semillas evolutivas con comportamiento emergente
- GNNs caóticas con arquitecturas fractales
- Evolución no determinista pero controlada
- Atractores extraños en el espacio de conocimiento
- Mutaciones cuánticas en el grafo
"""

# === IMPORTACIONES CORE ===
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
import asyncio
import random
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import hashlib
import json
import logging
from abc import ABC, abstractmethod

# Importar del MSC Framework
from msc_simulation import (
    AdvancedCollectiveSynthesisGraph,
    AdvancedKnowledgeComponent,
    Event, EventType,
    ImprovedBaseAgent,
    Config
)

# === MATEMÁTICA DEL CAOS ===
class ChaosMathematics:
    """Implementación de matemática pura del caos"""
    
    # Constantes del caos
    FEIGENBAUM_DELTA = 4.669201609102990671853203820466
    FEIGENBAUM_ALPHA = 2.502907875095892822283902873218
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    SILVER_RATIO = 1 + np.sqrt(2)
    
    @staticmethod
    def lorenz_attractor(state: np.ndarray, sigma: float = 10.0, 
                        rho: float = 28.0, beta: float = 8/3) -> np.ndarray:
        """Sistema de Lorenz - Atractor caótico clásico"""
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])
    
    @staticmethod
    def rossler_attractor(state: np.ndarray, a: float = 0.2, 
                         b: float = 0.2, c: float = 5.7) -> np.ndarray:
        """Atractor de Rössler"""
        x, y, z = state
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return np.array([dx, dy, dz])
    
    @staticmethod
    def henon_map(x: float, y: float, a: float = 1.4, b: float = 0.3) -> Tuple[float, float]:
        """Mapa de Hénon - Sistema caótico discreto"""
        x_new = 1 - a * x**2 + y
        y_new = b * x
        return x_new, y_new
    
    @staticmethod
    def logistic_map(x: float, r: float) -> float:
        """Mapa logístico - Ruta al caos"""
        return r * x * (1 - x)
    
    @staticmethod
    def mandelbrot_iteration(c: complex, max_iter: int = 100) -> int:
        """Iteración del conjunto de Mandelbrot"""
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter
    
    @staticmethod
    def julia_set_point(z: complex, c: complex, max_iter: int = 100) -> int:
        """Punto en el conjunto de Julia"""
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter
    
    @staticmethod
    def calculate_lyapunov_exponent(trajectory: np.ndarray) -> float:
        """Calcula el exponente de Lyapunov"""
        n = len(trajectory) - 1
        if n <= 0:
            return 0.0
        
        lyap_sum = 0.0
        for i in range(n):
            if trajectory[i] != 0:
                lyap_sum += np.log(abs(trajectory[i+1] - trajectory[i]) / abs(trajectory[i]))
        
        return lyap_sum / n
    
    @staticmethod
    def strange_attractor_dimension(points: np.ndarray, epsilon: float = 0.01) -> float:
        """Calcula la dimensión fractal de un atractor extraño"""
        n_points = len(points)
        if n_points < 2:
            return 0.0
        
        # Método de conteo de cajas
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Crear grid
        n_boxes = np.ceil((max_coords - min_coords) / epsilon).astype(int)
        occupied_boxes = set()
        
        for point in points:
            box_coords = tuple(((point - min_coords) / epsilon).astype(int))
            occupied_boxes.add(box_coords)
        
        # Dimensión fractal aproximada
        if len(occupied_boxes) > 0:
            return np.log(len(occupied_boxes)) / np.log(1/epsilon)
        return 0.0
    
    @staticmethod
    def quantum_chaos_operator(state: np.ndarray, hbar: float = 1.0) -> np.ndarray:
        """Operador caótico cuántico"""
        # Hamiltoniano caótico cuántico simplificado
        n = len(state)
        H = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, j] = np.sin(2 * np.pi * i / n)
                elif abs(i - j) == 1:
                    H[i, j] = 0.5 * np.exp(1j * np.pi * (i + j) / n)
        
        # Evolución unitaria
        U = np.exp(-1j * H / hbar)
        return U @ state

# === SEMILLAS EVOLUTIVAS ===
@dataclass
class EvolutionSeed:
    """Semilla evolutiva con propiedades caóticas"""
    id: str
    dna: np.ndarray  # Código genético multidimensional
    fitness: float = 0.0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    chaos_factor: float = 0.5
    birth_time: float = field(default_factory=time.time)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def mutate(self, chaos_math: ChaosMathematics) -> 'EvolutionSeed':
        """Mutación caótica de la semilla"""
        # Aplicar mapa logístico para determinar intensidad de mutación
        r = 3.0 + self.chaos_factor  # r entre 3 y 4 para comportamiento caótico
        mutation_intensity = chaos_math.logistic_map(self.mutation_rate, r)
        
        # Mutar DNA
        mutated_dna = self.dna.copy()
        n_mutations = int(len(self.dna) * mutation_intensity)
        
        for _ in range(n_mutations):
            idx = random.randint(0, len(self.dna) - 1)
            # Mutación usando atractor de Lorenz
            lorenz_state = np.array([mutated_dna[idx], 
                                    random.random(), 
                                    random.random()])
            delta = chaos_math.lorenz_attractor(lorenz_state) * 0.01
            mutated_dna[idx] += delta[0]
        
        # Normalizar
        mutated_dna = np.tanh(mutated_dna)
        
        return EvolutionSeed(
            id=f"{self.id}_mut_{int(time.time()*1000000)}",
            dna=mutated_dna,
            mutation_rate=self.mutation_rate * (1 + random.gauss(0, 0.1)),
            crossover_rate=self.crossover_rate,
            chaos_factor=min(1.0, self.chaos_factor * 1.1),
            generation=self.generation + 1,
            parent_ids=[self.id]
        )
    
    def crossover(self, other: 'EvolutionSeed', chaos_math: ChaosMathematics) -> List['EvolutionSeed']:
        """Cruce caótico con otra semilla"""
        # Determinar puntos de cruce usando mapa de Hénon
        n_points = min(len(self.dna), len(other.dna))
        x, y = 0.5, 0.5
        
        crossover_points = []
        for i in range(3):  # 3 puntos de cruce
            x, y = chaos_math.henon_map(x, y)
            point = int(abs(x) * n_points) % n_points
            crossover_points.append(point)
        
        crossover_points.sort()
        
        # Crear descendencia
        offspring1_dna = np.zeros_like(self.dna)
        offspring2_dna = np.zeros_like(self.dna)
        
        current_parent = 0
        for i in range(n_points):
            if i in crossover_points:
                current_parent = 1 - current_parent
            
            if current_parent == 0:
                offspring1_dna[i] = self.dna[i]
                offspring2_dna[i] = other.dna[i] if i < len(other.dna) else 0
            else:
                offspring1_dna[i] = other.dna[i] if i < len(other.dna) else 0
                offspring2_dna[i] = self.dna[i]
        
        # Añadir ruido caótico
        noise_factor = 0.05 * self.chaos_factor
        offspring1_dna += np.random.normal(0, noise_factor, size=offspring1_dna.shape)
        offspring2_dna += np.random.normal(0, noise_factor, size=offspring2_dna.shape)
        
        offspring1 = EvolutionSeed(
            id=f"{self.id}x{other.id}_1_{int(time.time()*1000000)}",
            dna=np.tanh(offspring1_dna),
            mutation_rate=(self.mutation_rate + other.mutation_rate) / 2,
            crossover_rate=(self.crossover_rate + other.crossover_rate) / 2,
            chaos_factor=(self.chaos_factor + other.chaos_factor) / 2,
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.id, other.id]
        )
        
        offspring2 = EvolutionSeed(
            id=f"{self.id}x{other.id}_2_{int(time.time()*1000000)}",
            dna=np.tanh(offspring2_dna),
            mutation_rate=(self.mutation_rate + other.mutation_rate) / 2,
            crossover_rate=(self.crossover_rate + other.crossover_rate) / 2,
            chaos_factor=(self.chaos_factor + other.chaos_factor) / 2,
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.id, other.id]
        )
        
        return [offspring1, offspring2]

# === GNN CAÓTICA FRACTAL ===
class FractalAttention(MessagePassing):
    """Capa de atención fractal para GNN"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8,
                 fractal_depth: int = 3, chaos_factor: float = 0.1):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.fractal_depth = fractal_depth
        self.chaos_factor = chaos_factor
        
        # Transformaciones fractales
        self.transforms = nn.ModuleList()
        current_dim = in_channels
        
        for depth in range(fractal_depth):
            self.transforms.append(nn.Sequential(
                nn.Linear(current_dim, out_channels * heads),
                nn.LayerNorm(out_channels * heads),
                nn.ReLU(),
                nn.Dropout(0.1 + depth * 0.05)
            ))
            current_dim = out_channels * heads
        
        # Proyecciones de atención
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Aplicar transformaciones fractales
        h = x
        fractal_features = []
        
        for transform in self.transforms:
            h = transform(h)
            fractal_features.append(h)
            
            # Añadir caos
            if self.chaos_factor > 0:
                chaos_noise = torch.randn_like(h) * self.chaos_factor
                h = h + chaos_noise
        
        # Combinar características fractales
        h = sum(fractal_features) / len(fractal_features)
        h = h.view(-1, self.heads, self.out_channels)
        
        # Propagación con atención
        return self.propagate(edge_index, x=h, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Atención caótica
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        
        # Calcular scores de atención con componente caótico
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        
        # Añadir perturbación caótica a la atención
        if self.training and self.chaos_factor > 0:
            chaos_perturbation = torch.randn_like(alpha) * self.chaos_factor
            alpha = alpha + chaos_perturbation
        
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Dropout de atención
        alpha = F.dropout(alpha, p=0.1, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

class ChaoticEvolutionGNN(nn.Module):
    """GNN con arquitectura caótica evolutiva"""
    
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 6, fractal_depth: int = 3,
                 chaos_factor: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.chaos_factor = chaos_factor
        
        # Codificador inicial con proyección caótica
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Capas fractales de atención
        self.fractal_layers = nn.ModuleList()
        for i in range(num_layers):
            self.fractal_layers.append(
                FractalAttention(
                    hidden_dim,
                    hidden_dim,
                    heads=8,
                    fractal_depth=fractal_depth,
                    chaos_factor=chaos_factor * (1 + i * 0.1)  # Incrementar caos
                )
            )
        
        # Operador de evolución caótica
        self.evolution_operator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        
        # Decodificador con múltiples cabezas
        self.decoders = nn.ModuleDict({
            'state': nn.Linear(hidden_dim, 1),
            'importance': nn.Linear(hidden_dim, 1),
            'chaos': nn.Linear(hidden_dim, 1),
            'evolution': nn.Linear(hidden_dim, output_dim)
        })
        
        # Memoria caótica
        self.chaos_memory = nn.Parameter(torch.randn(1, hidden_dim))
        
    def forward(self, x, edge_index, batch=None):
        # Codificación inicial
        h = self.encoder(x)
        
        # Añadir memoria caótica
        if self.chaos_factor > 0:
            chaos_influence = self.chaos_memory.expand(h.size(0), -1)
            h = h + chaos_influence * self.chaos_factor
        
        # Propagación fractal
        residuals = []
        for i, layer in enumerate(self.fractal_layers):
            h_prev = h
            h = layer(h, edge_index)
            
            # Conexión residual con gating caótico
            gate = torch.sigmoid(self.evolution_operator(h))
            h = gate * h + (1 - gate) * h_prev
            
            residuals.append(h)
            
            # Normalización adaptativa
            h = F.layer_norm(h, [self.hidden_dim])
        
        # Agregación multi-escala
        h_multi = sum(residuals) / len(residuals)
        
        # Pooling global
        if batch is not None:
            h_graph = global_mean_pool(h_multi, batch)
        else:
            h_graph = h_multi.mean(dim=0, keepdim=True)
        
        # Decodificación múltiple
        outputs = {}
        for key, decoder in self.decoders.items():
            outputs[key] = decoder(h_graph if batch is not None else h_multi)
        
        return outputs, h_multi

# === SISTEMA DE CAOS EVOLUTIVO ===
class ChaosEvolutionSystem:
    """Sistema principal de evolución caótica"""
    
    def __init__(self, base_graph: AdvancedCollectiveSynthesisGraph):
        self.base_graph = base_graph
        self.chaos_math = ChaosMathematics()
        self.evolution_pool: List[EvolutionSeed] = []
        self.chaos_gnn = ChaoticEvolutionGNN(
            num_features=768,
            hidden_dim=256,
            output_dim=64,
            chaos_factor=0.15
        )
        self.attractor_states: deque = deque(maxlen=10000)
        self.phase_space: Dict[str, np.ndarray] = {}
        self.bifurcation_points: List[float] = []
        
        # Parámetros del sistema
        self.system_state = np.random.randn(3)  # Estado inicial en R³
        self.control_parameter = 3.57  # Cerca del inicio del caos
        
        logger = logging.getLogger(__name__)
        logger.info("Chaos Evolution System initialized")
    
    async def inject_chaos_seed(self, node_id: int, chaos_level: float = 0.5):
        """Inyecta una semilla caótica en un nodo"""
        node = self.base_graph.nodes.get(node_id)
        if not node:
            return None
        
        # Crear semilla evolutiva
        dna_size = 128
        dna = np.random.randn(dna_size) * chaos_level
        
        # Aplicar transformación caótica inicial
        for i in range(10):  # 10 iteraciones de pre-caos
            x = dna[i % dna_size]
            dna[i % dna_size] = self.chaos_math.logistic_map(
                (x + 1) / 2,  # Normalizar a [0,1]
                self.control_parameter
            ) * 2 - 1  # Desnormalizar
        
        seed = EvolutionSeed(
            id=f"chaos_seed_{node_id}_{int(time.time()*1000000)}",
            dna=dna,
            chaos_factor=chaos_level
        )
        
        self.evolution_pool.append(seed)
        
        # Modificar el nodo con influencia caótica
        await self._apply_chaos_to_node(node, seed)
        
        return seed
    
    async def _apply_chaos_to_node(self, node: AdvancedKnowledgeComponent, 
                                  seed: EvolutionSeed):
        """Aplica efectos caóticos a un nodo"""
        # Calcular nuevo estado usando atractor de Lorenz
        current_state = np.array([node.state, seed.fitness, seed.chaos_factor])
        delta = self.chaos_math.lorenz_attractor(current_state) * 0.001
        
        new_state = node.state + delta[0]
        new_state = max(0.01, min(1.0, new_state))
        
        await node.update_state(new_state, source="chaos_system", 
                               reason=f"Chaos injection from seed {seed.id}")
        
        # Añadir keywords caóticos
        chaos_keywords = {
            f"chaos_{int(seed.chaos_factor*100)}",
            f"attractor_{seed.generation}",
            f"fractal_{int(abs(delta[1])*1000)}"
        }
        node.keywords.update(chaos_keywords)
        
        # Marcar con propiedades caóticas
        node.metadata.properties['chaos_seed_id'] = seed.id
        node.metadata.properties['lyapunov_exponent'] = self.chaos_math.calculate_lyapunov_exponent(
            np.array([node.state] + list(node.state_history)[-10:])
        )
    
    async def evolve_population(self, fitness_function: callable = None):
        """Evoluciona la población de semillas"""
        if len(self.evolution_pool) < 2:
            return
        
        # Evaluar fitness
        for seed in self.evolution_pool:
            if fitness_function:
                seed.fitness = await fitness_function(seed, self.base_graph)
            else:
                # Fitness por defecto basado en impacto en el grafo
                seed.fitness = await self._default_fitness(seed)
        
        # Ordenar por fitness
        self.evolution_pool.sort(key=lambda s: s.fitness, reverse=True)
        
        # Selección elitista con componente caótico
        elite_size = max(2, int(len(self.evolution_pool) * 0.2))
        elite = self.evolution_pool[:elite_size]
        
        # Nueva generación
        new_generation = elite.copy()
        
        # Reproducción caótica
        while len(new_generation) < 50:  # Límite de población
            # Selección de padres con torneo caótico
            parent1 = self._chaotic_selection(self.evolution_pool)
            parent2 = self._chaotic_selection(self.evolution_pool)
            
            if random.random() < parent1.crossover_rate:
                # Cruce
                offspring = parent1.crossover(parent2, self.chaos_math)
                new_generation.extend(offspring)
            
            # Mutación
            if random.random() < parent1.mutation_rate:
                mutant = parent1.mutate(self.chaos_math)
                new_generation.append(mutant)
        
        # Actualizar población
        self.evolution_pool = new_generation[:50]
        
        # Detectar bifurcaciones
        await self._detect_bifurcations()
    
    def _chaotic_selection(self, population: List[EvolutionSeed]) -> EvolutionSeed:
        """Selección con componente caótico"""
        # Torneo con tamaño variable según mapa logístico
        tournament_size = 2 + int(self.chaos_math.logistic_map(
            random.random(), 
            self.control_parameter
        ) * 5)
        
        tournament = random.sample(population, 
                                 min(tournament_size, len(population)))
        
        # Añadir perturbación caótica al fitness
        best = max(tournament, key=lambda s: s.fitness + 
                  random.gauss(0, 0.1 * s.chaos_factor))
        
        return best
    
    async def _default_fitness(self, seed: EvolutionSeed) -> float:
        """Función de fitness por defecto"""
        # Buscar nodos afectados por esta semilla
        affected_nodes = [
            node for node in self.base_graph.nodes.values()
            if node.metadata.properties.get('chaos_seed_id') == seed.id
        ]
        
        if not affected_nodes:
            return 0.0
        
        # Fitness basado en:
        # 1. Cambio de estado promedio
        state_changes = sum(
            abs(node.state - 0.5) for node in affected_nodes
        ) / len(affected_nodes)
        
        # 2. Conectividad
        connectivity = sum(
            len(node.connections_out) + len(node.connections_in)
            for node in affected_nodes
        ) / len(affected_nodes)
        
        # 3. Dimensión fractal del espacio de estados
        if len(self.attractor_states) > 100:
            fractal_dim = self.chaos_math.strange_attractor_dimension(
                np.array(list(self.attractor_states)[-100:])
            )
        else:
            fractal_dim = 1.0
        
        # Combinar con pesos caóticos
        weights = np.array([
            self.chaos_math.logistic_map(0.3, self.control_parameter),
            self.chaos_math.logistic_map(0.5, self.control_parameter),
            self.chaos_math.logistic_map(0.7, self.control_parameter)
        ])
        weights = weights / weights.sum()
        
        fitness = (
            weights[0] * state_changes +
            weights[1] * connectivity / 10 +
            weights[2] * fractal_dim / 3
        )
        
        return min(1.0, fitness)
    
    async def _detect_bifurcations(self):
        """Detecta puntos de bifurcación en el sistema"""
        # Incrementar parámetro de control
        self.control_parameter += 0.001
        
        # Simular sistema para detectar cambios cualitativos
        trajectory = []
        x = 0.5
        
        # Dejar que el sistema se estabilice
        for _ in range(1000):
            x = self.chaos_math.logistic_map(x, self.control_parameter)
        
        # Registrar trayectoria
        for _ in range(100):
            x = self.chaos_math.logistic_map(x, self.control_parameter)
            trajectory.append(x)
        
        # Detectar período
        period = self._detect_period(trajectory)
        
        # Si hay cambio de período, es una bifurcación
        if hasattr(self, '_last_period') and period != self._last_period:
            self.bifurcation_points.append(self.control_parameter)
            
            # Evento de bifurcación
            await self.base_graph.event_bus.publish(Event(
                type=EventType.METRICS_UPDATE,
                data={
                    'type': 'bifurcation_detected',
                    'control_parameter': self.control_parameter,
                    'old_period': self._last_period,
                    'new_period': period
                },
                source='chaos_system'
            ))
        
        self._last_period = period
    
    def _detect_period(self, trajectory: List[float], tolerance: float = 1e-6) -> int:
        """Detecta el período de una trayectoria"""
        n = len(trajectory)
        
        for period in range(1, n // 2):
            is_periodic = True
            
            for i in range(period, n - period):
                if abs(trajectory[i] - trajectory[i - period]) > tolerance:
                    is_periodic = False
                    break
            
            if is_periodic:
                return period
        
        return -1  # Caótico o período muy largo
    
    async def generate_fractal_knowledge(self, depth: int = 5) -> List[int]:
        """Genera conocimiento con estructura fractal"""
        created_nodes = []
        
        # Punto inicial en el plano complejo
        c = complex(-0.7, 0.27015)  # Cerca del conjunto de Julia
        
        async def create_fractal_branch(z: complex, level: int, parent_id: Optional[int] = None):
            if level >= depth:
                return
            
            # Iterar función de Julia
            z_new = z * z + c
            
            # Crear nodo si está en el conjunto
            if abs(z_new) < 2:
                # Contenido basado en la posición fractal
                content = f"Fractal Knowledge Node at z={z_new:.4f}, level={level}"
                
                # Keywords fractales
                keywords = {
                    f"fractal_level_{level}",
                    f"julia_set",
                    f"complex_{int(z_new.real*1000)}_{int(z_new.imag*1000)}"
                }
                
                # Crear nodo
                node = await self.base_graph.add_node(
                    content=content,
                    initial_state=abs(z_new) / 2,  # Estado basado en magnitud
                    keywords=keywords,
                    created_by="chaos_fractal_generator",
                    properties={
                        'fractal_position': {'real': z_new.real, 'imag': z_new.imag},
                        'fractal_level': level,
                        'mandelbrot_escape': self.chaos_math.mandelbrot_iteration(z_new)
                    }
                )
                
                created_nodes.append(node.id)
                
                # Conectar con padre si existe
                if parent_id is not None:
                    weight = 1.0 / (1.0 + abs(z_new))
                    await self.base_graph.add_edge(parent_id, node.id, weight)
                
                # Crear ramas hijas
                for angle in np.linspace(0, 2*np.pi, 4, endpoint=False):
                    z_child = z_new + 0.1 * np.exp(1j * angle)
                    await create_fractal_branch(z_child, level + 1, node.id)
        
        # Iniciar desde múltiples puntos
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            z0 = 0.5 * np.exp(1j * angle)
            await create_fractal_branch(z0, 0)
        
        return created_nodes
    
    async def induce_quantum_chaos(self, node_ids: List[int]):
        """Induce caos cuántico en un conjunto de nodos"""
        if not node_ids:
            return
        
        # Crear estado cuántico
        n = len(node_ids)
        quantum_state = np.random.randn(n) + 1j * np.random.randn(n)
        quantum_state = quantum_state / np.linalg.norm(quantum_state)
        
        # Evolucionar con operador caótico
        for _ in range(10):
            quantum_state = self.chaos_math.quantum_chaos_operator(quantum_state)
        
        # Aplicar a nodos
        for i, node_id in enumerate(node_ids):
            node = self.base_graph.nodes.get(node_id)
            if node:
                # Nuevo estado basado en amplitud cuántica
                amplitude = abs(quantum_state[i])
                phase = np.angle(quantum_state[i])
                
                new_state = (amplitude + node.state) / 2
                await node.update_state(
                    new_state,
                    source="quantum_chaos",
                    reason=f"Quantum amplitude: {amplitude:.4f}, phase: {phase:.4f}"
                )
                
                # Añadir propiedades cuánticas
                node.metadata.properties['quantum_amplitude'] = float(amplitude)
                node.metadata.properties['quantum_phase'] = float(phase)
                node.metadata.properties['entanglement'] = float(
                    np.abs(quantum_state @ quantum_state.conj())
                )

# === AGENTE CAÓTICO ===
class ChaoticEvolutionAgent(ImprovedBaseAgent):
    """Agente que utiliza principios del caos para evolucionar"""
    
    def __init__(self, agent_id: str, graph: AdvancedCollectiveSynthesisGraph,
                 config: Dict[str, Any], chaos_system: ChaosEvolutionSystem):
        super().__init__(agent_id, graph, config)
        self.chaos_system = chaos_system
        self.internal_state = np.random.randn(3)  # Estado interno caótico
        self.trajectory = deque(maxlen=1000)
        self.current_attractor = 'lorenz'  # lorenz, rossler, henon
        self.mutation_history = deque(maxlen=100)
        
    def _get_available_actions(self) -> List[str]:
        return [
            'inject_chaos', 'evolve_seeds', 'create_fractal',
            'quantum_chaos', 'bifurcate', 'strange_attract'
        ]
    
    async def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta acciones caóticas"""
        try:
            if action == 'inject_chaos':
                return await self._inject_chaos(context)
            elif action == 'evolve_seeds':
                return await self._evolve_seeds(context)
            elif action == 'create_fractal':
                return await self._create_fractal(context)
            elif action == 'quantum_chaos':
                return await self._quantum_chaos(context)
            elif action == 'bifurcate':
                return await self._bifurcate(context)
            elif action == 'strange_attract':
                return await self._strange_attract(context)
            else:
                return {'success': False, 'error': 'Unknown chaos action'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _inject_chaos(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inyecta caos en el sistema"""
        # Seleccionar nodos objetivo
        target_nodes = self._select_chaos_targets()
        
        injected_seeds = []
        for node_id in target_nodes[:5]:  # Limitar a 5 nodos
            # Nivel de caos basado en estado interno
            chaos_level = abs(self.internal_state[0]) / (1 + abs(self.internal_state[0]))
            
            seed = await self.chaos_system.inject_chaos_seed(node_id, chaos_level)
            if seed:
                injected_seeds.append(seed.id)
        
        # Evolucionar estado interno
        self._evolve_internal_state()
        
        return {
            'success': True,
            'action': 'inject_chaos',
            'seeds_injected': len(injected_seeds),
            'chaos_level': chaos_level,
            'impact': len(injected_seeds) * 0.3
        }
    
    async def _evolve_seeds(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evoluciona las semillas caóticas"""
        initial_pool_size = len(self.chaos_system.evolution_pool)
        
        # Definir función de fitness adaptativa
        async def adaptive_fitness(seed: EvolutionSeed, graph) -> float:
            base_fitness = await self.chaos_system._default_fitness(seed)
            
            # Modificar fitness según estado del agente
            chaos_bonus = self.chaos_system.chaos_math.logistic_map(
                abs(self.internal_state[1]) % 1,
                3.7  # En régimen caótico
            )
            
            return base_fitness * (1 + chaos_bonus * 0.5)
        
        # Evolucionar
        await self.chaos_system.evolve_population(adaptive_fitness)
        
        final_pool_size = len(self.chaos_system.evolution_pool)
        
        return {
            'success': True,
            'action': 'evolve_seeds',
            'initial_pool': initial_pool_size,
            'final_pool': final_pool_size,
            'generations_advanced': 1,
            'impact': abs(final_pool_size - initial_pool_size) * 0.2
        }
    
    async def _create_fractal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crea estructura de conocimiento fractal"""
        # Profundidad basada en estado caótico
        depth = 3 + int(abs(self.internal_state[2]) * 2)
        depth = min(depth, 7)  # Limitar profundidad
        
        created_nodes = await self.chaos_system.generate_fractal_knowledge(depth)
        
        return {
            'success': True,
            'action': 'create_fractal',
            'nodes_created': len(created_nodes),
            'fractal_depth': depth,
            'impact': len(created_nodes) * 0.1,
            'novel': True
        }
    
    async def _quantum_chaos(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica caos cuántico"""
        # Seleccionar nodos para entrelazamiento cuántico
        candidates = [
            n for n in self.graph.nodes.values()
            if n.state > 0.5 and 'quantum' not in n.metadata.properties
        ]
        
        if not candidates:
            return {'success': False, 'error': 'No suitable nodes for quantum chaos'}
        
        # Seleccionar subconjunto
        n_nodes = min(len(candidates), 2 ** int(abs(self.internal_state[0]) + 3))
        selected = random.sample(candidates, n_nodes)
        node_ids = [n.id for n in selected]
        
        await self.chaos_system.induce_quantum_chaos(node_ids)
        
        return {
            'success': True,
            'action': 'quantum_chaos',
            'nodes_affected': len(node_ids),
            'quantum_dimension': int(np.log2(len(node_ids))),
            'impact': len(node_ids) * 0.4
        }
    
    async def _bifurcate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Induce bifurcación en el sistema"""
        # Cambiar parámetro de control
        old_param = self.chaos_system.control_parameter
        
        # Salto basado en mapa de Hénon
        x, y = self.internal_state[0], self.internal_state[1]
        x_new, y_new = self.chaos_system.chaos_math.henon_map(x, y)
        
        delta = (x_new - x) * 0.1
        self.chaos_system.control_parameter += delta
        
        # Detectar cambios
        await self.chaos_system._detect_bifurcations()
        
        return {
            'success': True,
            'action': 'bifurcate',
            'old_parameter': old_param,
            'new_parameter': self.chaos_system.control_parameter,
            'delta': delta,
            'bifurcations_detected': len(self.chaos_system.bifurcation_points),
            'impact': abs(delta) * 10
        }
    
    async def _strange_attract(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crea atractor extraño en el grafo"""
        # Seleccionar nodos para formar el atractor
        n_nodes = 10 + int(abs(self.internal_state[0]) * 10)
        nodes = list(self.graph.nodes.values())[:n_nodes]
        
        if len(nodes) < 3:
            return {'success': False, 'error': 'Insufficient nodes for attractor'}
        
        # Crear conexiones siguiendo dinámica de atractor
        connections_created = 0
        
        for i, node in enumerate(nodes):
            # Estado del atractor
            if self.current_attractor == 'lorenz':
                state = np.array([
                    node.state,
                    float(i) / len(nodes),
                    self.internal_state[2]
                ])
                next_state = self.chaos_system.chaos_math.lorenz_attractor(state)
            elif self.current_attractor == 'rossler':
                state = np.array([
                    node.state,
                    float(i) / len(nodes),
                    abs(self.internal_state[1])
                ])
                next_state = self.chaos_system.chaos_math.rossler_attractor(state)
            else:  # henon
                x, y = node.state, float(i) / len(nodes)
                x_new, y_new = self.chaos_system.chaos_math.henon_map(x, y)
                next_state = np.array([x_new, y_new, 0])
            
            # Conectar según dinámica
            for j, other_node in enumerate(nodes):
                if i != j:
                    # Peso basado en distancia en el espacio de fases
                    weight = 1.0 / (1.0 + np.linalg.norm(next_state[:2]))
                    
                    if weight > 0.3:  # Umbral de conexión
                        success = await self.graph.add_edge(node.id, other_node.id, weight)
                        if success:
                            connections_created += 1
        
        # Cambiar atractor para próxima vez
        attractors = ['lorenz', 'rossler', 'henon']
        self.current_attractor = random.choice(attractors)
        
        return {
            'success': True,
            'action': 'strange_attract',
            'attractor_type': self.current_attractor,
            'nodes_involved': len(nodes),
            'connections_created': connections_created,
            'impact': connections_created * 0.15
        }
    
    def _select_chaos_targets(self) -> List[int]:
        """Selecciona nodos objetivo para el caos"""
        # Criterios caóticos de selección
        candidates = []
        
        for node in self.graph.nodes.values():
            # Score caótico
            score = 0
            
            # Preferir nodos estables (para desestabilizar)
            if 0.4 < node.state < 0.6:
                score += 1
            
            # Preferir nodos con pocas conexiones
            if len(node.connections_out) < 3:
                score += 1
            
            # Bonus aleatorio caótico
            score += self.chaos_system.chaos_math.logistic_map(
                random.random(),
                self.chaos_system.control_parameter
            )
            
            candidates.append((node.id, score))
        
        # Ordenar por score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [node_id for node_id, _ in candidates[:10]]
    
    def _evolve_internal_state(self):
        """Evoluciona el estado interno del agente"""
        # Aplicar dinámica caótica
        if len(self.trajectory) % 10 == 0:
            # Cambiar dinámica ocasionalmente
            dynamics = [
                lambda s: self.chaos_system.chaos_math.lorenz_attractor(s) * 0.01,
                lambda s: self.chaos_system.chaos_math.rossler_attractor(s) * 0.01,
                lambda s: np.array([
                    self.chaos_system.chaos_math.logistic_map(s[0], 3.8),
                    self.chaos_system.chaos_math.logistic_map(s[1], 3.9),
                    self.chaos_system.chaos_math.logistic_map(s[2], 3.7)
                ])
            ]
            self._current_dynamics = random.choice(dynamics)
        
        if hasattr(self, '_current_dynamics'):
            delta = self._current_dynamics(self.internal_state)
            self.internal_state += delta
            
            # Mantener acotado
            self.internal_state = np.tanh(self.internal_state)
        
        # Registrar trayectoria
        self.trajectory.append(self.internal_state.copy())
        
        # Calcular Lyapunov ocasionalmente
        if len(self.trajectory) == 1000:
            lyapunov = self.chaos_system.chaos_math.calculate_lyapunov_exponent(
                np.array([s[0] for s in self.trajectory])
            )
            self.mutation_history.append({
                'timestamp': time.time(),
                'lyapunov': lyapunov,
                'state': self.internal_state.copy()
            })

# === INTEGRACIÓN CON MSC ===
async def integrate_chaos_with_msc(msc_graph: AdvancedCollectiveSynthesisGraph,
                                  config: Dict[str, Any]) -> ChaosEvolutionSystem:
    """Integra el sistema de caos con el MSC existente"""
    
    # Crear sistema de caos
    chaos_system = ChaosEvolutionSystem(msc_graph)
    
    # Registrar eventos de caos
    msc_graph.event_bus.add_filter(
        lambda event: event.source != 'chaos_system' or event.priority > 5
    )
    
    # Crear agentes caóticos
    n_chaos_agents = config.get('chaos_agents', 2)
    chaos_agents = []
    
    for i in range(n_chaos_agents):
        agent = ChaoticEvolutionAgent(
            f"ChaosAgent_{i}",
            msc_graph,
            config,
            chaos_system
        )
        chaos_agents.append(agent)
    
    # Inyectar semillas iniciales
    if config.get('auto_inject_chaos', True):
        # Seleccionar nodos semilla
        seed_nodes = random.sample(
            list(msc_graph.nodes.keys()),
            min(5, len(msc_graph.nodes))
        )
        
        for node_id in seed_nodes:
            await chaos_system.inject_chaos_seed(
                node_id,
                chaos_level=config.get('initial_chaos_level', 0.3)
            )
    
    # Iniciar evolución periódica
    async def periodic_evolution():
        while True:
            await asyncio.sleep(config.get('evolution_interval', 30))
            await chaos_system.evolve_population()
    
    asyncio.create_task(periodic_evolution())
    
    logger = logging.getLogger(__name__)
    logger.info(f"Chaos Evolution System integrated with {n_chaos_agents} agents")
    
    return chaos_system, chaos_agents

# === UTILIDADES DE VISUALIZACIÓN ===
def visualize_chaos_attractor(chaos_system: ChaosEvolutionSystem,
                             n_points: int = 10000) -> np.ndarray:
    """Genera datos para visualizar el atractor del sistema"""
    trajectory = []
    state = chaos_system.system_state.copy()
    
    for _ in range(n_points):
        # Evolucionar según el atractor actual
        delta = chaos_system.chaos_math.lorenz_attractor(state)
        state = state + delta * 0.01
        trajectory.append(state.copy())
    
    return np.array(trajectory)

def calculate_fractal_dimension(graph_nodes: Dict[int, Any]) -> float:
    """Calcula la dimensión fractal del grafo"""
    if len(graph_nodes) < 10:
        return 1.0
    
    # Extraer posiciones (usar estados como coordenadas)
    positions = []
    for node in graph_nodes.values():
        # Usar múltiples propiedades como coordenadas
        pos = [
            node.state,
            len(node.connections_out) / 10.0,
            len(node.keywords) / 20.0
        ]
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Calcular dimensión fractal
    chaos_math = ChaosMathematics()
    return chaos_math.strange_attractor_dimension(positions)

# === MÉTRICAS DEL CAOS ===
class ChaosMetrics:
    """Métricas específicas del sistema caótico"""
    
    def __init__(self):
        self.entropy_history = deque(maxlen=1000)
        self.lyapunov_history = deque(maxlen=1000)
        self.bifurcation_history = []
        
    def calculate_entropy(self, states: List[float]) -> float:
        """Calcula entropía de Shannon de los estados"""
        if not states:
            return 0.0
        
        # Discretizar estados
        bins = np.linspace(0, 1, 20)
        hist, _ = np.histogram(states, bins=bins)
        
        # Normalizar
        hist = hist / hist.sum()
        
        # Calcular entropía
        entropy = -sum(p * np.log2(p) for p in hist if p > 0)
        
        self.entropy_history.append({
            'timestamp': time.time(),
            'entropy': entropy
        })
        
        return entropy
    
    def update_chaos_metrics(self, chaos_system: ChaosEvolutionSystem,
                           graph: AdvancedCollectiveSynthesisGraph):
        """Actualiza todas las métricas del caos"""
        # Recopilar estados
        states = [node.state for node in graph.nodes.values()]
        
        # Entropía
        entropy = self.calculate_entropy(states)
        
        # Exponente de Lyapunov promedio
        if len(states) > 10:
            lyapunov = chaos_system.chaos_math.calculate_lyapunov_exponent(
                np.array(states[:100])
            )
            self.lyapunov_history.append({
                'timestamp': time.time(),
                'lyapunov': lyapunov
            })
        
        # Dimensión fractal
        fractal_dim = calculate_fractal_dimension(graph.nodes)
        
        return {
            'entropy': entropy,
            'lyapunov': lyapunov if len(states) > 10 else 0,
            'fractal_dimension': fractal_dim,
            'evolution_pool_size': len(chaos_system.evolution_pool),
            'bifurcations': len(chaos_system.bifurcation_points)
        }

# === PUNTO DE ENTRADA ===
if __name__ == "__main__":
    print("Chaos Evolution Framework v1.0")
    print("Este módulo debe ser importado por el MSC Framework")
    print("Use: from chaos_evolution_framework import integrate_chaos_with_msc")

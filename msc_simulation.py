import random
import math
import time
from collections import Counter

# --- 1. Definiciones del Grafo de Síntesis (Modificado) ---

class KnowledgeComponent:
    """Representa un nodo (V') en el Grafo de Síntesis."""
    def __init__(self, node_id, content="Abstract Concept", initial_state=0.1, keywords=None):
        self.id = node_id
        self.content = content # Descripción abstracta o datos del nodo
        self.state = initial_state # Confianza/Calidad (sj) - Rango [0, 1]
        self.keywords = keywords if keywords else set() # Almacena palabras clave como un set
        # Conexiones salientes: {target_id: utility_uij}
        self.connections_out = {}
        # Conexiones entrantes: {source_id: utility_uji} (Útil para evaluación)
        self.connections_in = {}

    def update_state(self, new_state):
        """Actualiza el estado asegurando que esté en [0, 1]."""
        self.state = max(0.0, min(1.0, new_state))

    def add_connection(self, target_node, utility):
        """Añade una conexión saliente (arista E'). Evita auto-conexiones básicas."""
        if target_node.id != self.id:
             self.connections_out[target_node.id] = utility
             # Guarda la conexión entrante en el nodo destino para facilitar la evaluación
             target_node.connections_in[self.id] = utility

    def __repr__(self):
        """Representación del nodo, incluyendo keywords si existen."""
        kw_str = f", KW={list(self.keywords)}" if self.keywords else ""
        return f"Node({self.id}, S={self.state:.3f}{kw_str})"

class CollectiveSynthesisGraph:
    """Gestiona el Grafo de Síntesis (G')."""
    def __init__(self):
        self.nodes = {} # Diccionario: {node_id: KnowledgeComponent}
        self.next_node_id = 0 # Contador simple para IDs únicos

    def add_node(self, content="Abstract Concept", initial_state=0.1, keywords=None):
        """Añade un nuevo nodo al grafo."""
        node_id = self.next_node_id
        # Asegurarse de que keywords sea un set
        kw = set(keywords) if keywords else set()
        new_node = KnowledgeComponent(node_id, content, initial_state, kw)
        self.nodes[node_id] = new_node
        self.next_node_id += 1
        return new_node

    def add_edge(self, source_id, target_id, utility):
        """Añade una arista dirigida si ambos nodos existen y no son el mismo."""
        if source_id in self.nodes and target_id in self.nodes and source_id != target_id:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            # Evitar añadir conexión si ya existe (o permitir actualizarla si se desea)
            if target_id not in source_node.connections_out:
                 source_node.add_connection(target_node, utility)
                 # print(f"  Edge added: {source_id} -> {target_id} (U={utility:.2f})")
                 return True
        return False

    def get_node(self, node_id):
        """Obtiene un nodo por su ID."""
        return self.nodes.get(node_id)

    def get_nodes_sorted_by_state(self, descending=True):
        """Devuelve una lista de nodos ordenados por estado."""
        if not self.nodes:
            return []
        return sorted(self.nodes.values(), key=lambda node: node.state, reverse=descending)

    def get_random_node_biased(self):
        """Obtiene un nodo aleatorio, con sesgo hacia nodos de mayor estado (Atención Simple)."""
        if not self.nodes:
            return None
        # Usar los cuadrados de los estados como pesos para el sesgo
        # Se añade un pequeño epsilon para que nodos con estado 0 puedan ser elegidos
        nodes = list(self.nodes.values())
        weights = [(node.state**2) + 0.01 for node in nodes]
        # Asegurarse que la suma de pesos no sea cero (puede pasar si todos tienen estado 0)
        if sum(weights) == 0:
             return random.choice(nodes) if nodes else None
        return random.choices(nodes, weights=weights, k=1)[0]


    def print_summary(self):
        """Imprime un resumen del estado del grafo."""
        num_nodes = len(self.nodes)
        print(f"--- Graph Summary (Nodes: {num_nodes}) ---")
        if num_nodes == 0:
            print("  Graph is empty.")
            print("-" * 40)
            return

        # Mostrar los N nodos con mayor estado
        top_n = min(5, num_nodes) # Mostrar hasta 5 o el total si es menor
        sorted_nodes = self.get_nodes_sorted_by_state()
        for i in range(top_n):
            node = sorted_nodes[i]
            outs = len(node.connections_out)
            ins = len(node.connections_in)
            print(f"  Top {i+1}: {node!r} - Content: '{node.content[:30]}...' (Out: {outs}, In: {ins})")

        # Calcular estadísticas generales
        avg_state = sum(n.state for n in self.nodes.values()) / num_nodes
        avg_keywords = sum(len(n.keywords) for n in self.nodes.values()) / num_nodes if num_nodes > 0 else 0
        num_edges = sum(len(n.connections_out) for n in self.nodes.values())
        print(f"  Avg State: {avg_state:.3f}, Avg Keywords: {avg_keywords:.2f}, Total Edges: {num_edges}")
        print("-" * 40)

# --- 2. Definiciones de Sintetizadores (Actualizado y Nuevo) ---

class Synthesizer:
    """Clase base para los agentes que interactúan con el grafo."""
    def __init__(self, agent_id, graph):
        self.id = agent_id
        self.graph = graph

    def act(self):
        """Método abstracto para la acción del agente en un paso."""
        raise NotImplementedError

class ProposerAgent(Synthesizer):
    """Agente que propone nuevos componentes relacionados, ahora con keywords."""
    def act(self):
        # Ahora usa selección sesgada para encontrar un nodo fuente
        source_node = self.graph.get_random_node_biased()
        if source_node is None:
            # Si el grafo está vacío, propone un nodo inicial
            keywords = {"inicio", "semilla"}
            new_node = self.graph.add_node(content="Initial Seed", initial_state=0.2, keywords=keywords)
            print(f"Proposer {self.id}: Proposed initial {new_node!r}")
            return

        # Generar keywords para el nuevo nodo (ej: una de la fuente + una nueva)
        new_kw = f"kw_{self.graph.next_node_id}"
        # Tomar una keyword aleatoria de la fuente si tiene, sino usar 'related'
        source_kw = random.choice(list(source_node.keywords)) if source_node.keywords else "related"
        new_keywords = {source_kw, new_kw}

        new_content = f"Related concept to {source_node.id} about {new_kw}"
        # El estado inicial podría depender del estado de la fuente
        initial_state = max(0.05, source_node.state * random.uniform(0.3, 0.8))
        new_node = self.graph.add_node(content=new_content, initial_state=initial_state, keywords=new_keywords)

        # Utilidad basada parcialmente en el estado de la fuente y aleatoriedad
        utility = (source_node.state * 0.5 + random.uniform(-0.3, 0.7))
        utility = max(-1.0, min(1.0, utility)) # Clamp utility to [-1, 1]

        self.graph.add_edge(source_node.id, new_node.id, utility)
        print(f"Proposer {self.id}: Proposed {new_node!r} linked from {source_node!r} with U={utility:.2f}")


class EvaluatorAgent(Synthesizer):
    """Agente que evalúa y actualiza el estado de los nodos considerando estado de vecinos y keywords."""
    def __init__(self, agent_id, graph, learning_rate=0.1, keyword_boost=0.05):
        super().__init__(agent_id, graph)
        self.learning_rate = learning_rate
        self.keyword_boost = keyword_boost # Pequeño impulso por keywords compartidas

    def act(self):
        # Usa selección sesgada para elegir qué nodo evaluar
        target_node = self.graph.get_random_node_biased()
        if target_node is None: return # No hay nodos para evaluar

        influence_sum = 0.0
        weight_sum = 0.0 # Ponderación para normalizar la influencia
        shared_keywords_score_factor = 0.0 # Factor acumulado de boost por keywords

        if not target_node.connections_in:
            # Si no tiene entradas, su estado decae lentamente o se mantiene estable
            influence_target = target_node.state * 0.98 # Ligero decaimiento si está aislado
        else:
            # Calcular influencia de los nodos entrantes
            for source_id, utility_uji in target_node.connections_in.items():
                source_node = self.graph.get_node(source_id)
                if source_node:
                    # Influencia = EstadoFuente * UtilidadEnlace
                    influence = source_node.state * utility_uji
                    influence_sum += influence
                    # Ponderación basada en la fuerza absoluta de la conexión
                    weight = abs(utility_uji)
                    weight_sum += weight

                    # Comprobar keywords compartidas con nodos fuente relevantes
                    # Relevancia = EstadoFuente Alto (>0.5) Y Utilidad Positiva (>0.1)
                    if source_node.state > 0.5 and utility_uji > 0.1:
                        common_keywords = target_node.keywords.intersection(source_node.keywords)
                        if common_keywords:
                             # El boost es mayor si hay más keywords en común y el vecino es relevante
                             boost = len(common_keywords) * source_node.state * weight * self.keyword_boost
                             shared_keywords_score_factor += boost


            # Calcular un 'valor objetivo' basado en la influencia promedio ponderada
            if weight_sum > 0.01: # Evitar división por cero
                 base_influence_target = influence_sum / weight_sum
                 # Añadir el boost normalizado por keywords compartidas
                 keyword_influence_boost = shared_keywords_score_factor / weight_sum
                 influence_target = base_influence_target + keyword_influence_boost
            else:
                 influence_target = target_node.state # Mantener estado si no hay influencia ponderable

            # Limitar el objetivo a un rango razonable para evitar divergencias
            influence_target = max(-0.5, min(1.5, influence_target))

        # Actualizar estado: Mover el estado actual hacia el objetivo calculado
        current_state = target_node.state
        # El learning_rate controla la velocidad del cambio
        new_state = current_state + self.learning_rate * (influence_target - current_state)
        # Opcional: ligero decaimiento general para evitar estancamiento en 1.0
        # new_state *= 0.999

        target_node.update_state(new_state) # Asegura que el estado quede en [0, 1]

        print(f"Evaluator {self.id}: Evaluated {target_node!r}. State: {current_state:.3f} -> {target_node.state:.3f} (Target: {influence_target:.3f})")


class CombinerAgent(Synthesizer):
    """Agente que intenta crear enlaces entre nodos existentes no conectados."""
    def __init__(self, agent_id, graph, compatibility_threshold=0.6):
        super().__init__(agent_id, graph)
        # Umbral para decidir si dos nodos son suficientemente compatibles para unirlos
        self.compatibility_threshold = compatibility_threshold

    def act(self):
        if len(self.graph.nodes) < 2: return # Necesita al menos dos nodos para intentar combinar

        # Seleccionar dos nodos distintos, preferiblemente de alto estado (usando sesgo)
        node_a = self.graph.get_random_node_biased()
        node_b = self.graph.get_random_node_biased()

        # Asegurarse de que son distintos y no están ya conectados (A->B o B->A para evitar ciclos simples)
        if node_a is None or node_b is None or node_a.id == node_b.id \
           or node_b.id in node_a.connections_out \
           or node_a.id in node_b.connections_out: # Evitar conexión si ya existe en cualquier dirección
            return

        # Criterio de compatibilidad simple:
        # - Pondera el producto de sus estados
        # - Pondera las keywords compartidas
        state_product = node_a.state * node_b.state
        common_keywords = node_a.keywords.intersection(node_b.keywords)
        # Normalizar score de keywords (ej: basado en Jaccard o similar, aquí simplificado)
        max_possible_keywords = len(node_a.keywords.union(node_b.keywords))
        keyword_similarity = len(common_keywords) / max_possible_keywords if max_possible_keywords > 0 else 0

        # Calcular una puntuación de compatibilidad (ej: 60% estado, 40% keywords)
        compatibility_score = (state_product * 0.6) + (keyword_similarity * 0.4)

        # Si superan el umbral, crear un enlace A -> B
        if compatibility_score >= self.compatibility_threshold:
            # La utilidad del nuevo enlace podría basarse en la compatibilidad o los estados
            utility = compatibility_score * ((node_a.state + node_b.state) / 2.0)
            utility = max(-1.0, min(1.0, utility)) # Clamp utility

            if self.graph.add_edge(node_a.id, node_b.id, utility):
                 print(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} (Score: {compatibility_score:.2f}, U={utility:.2f})")
        # else:
            # # Opcional: imprimir si no son compatibles
            # print(f"Combiner {self.id}: Nodes {node_a.id} & {node_b.id} not compatible (Score: {compatibility_score:.2f})")


# --- 3. Simulación (Actualizada) ---

def run_simulation(num_steps=100, num_proposers=3, num_evaluators=6, num_combiners=2, step_delay=0.1):
    """Ejecuta la simulación del MSC."""
    # Inicializar grafo y agentes
    graph = CollectiveSynthesisGraph()
    agents = []
    # Crear agentes Proposer
    for i in range(num_proposers):
        agents.append(ProposerAgent(f"P{i}", graph))
    # Crear agentes Evaluator
    for i in range(num_evaluators):
        agents.append(EvaluatorAgent(f"E{i}", graph))
    # Crear agentes Combiner
    for i in range(num_combiners):
        agents.append(CombinerAgent(f"C{i}", graph))


    print(f"--- Starting MSC Simulation ({num_steps} steps) ---")
    print(f"Agents: {len(agents)} ({num_proposers}P, {num_evaluators}E, {num_combiners}C)")

    # Bucle principal de la simulación
    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        if not agents:
            print("No agents to run.")
            break # Salir si no hay agentes

        # Selección de agente aleatoria simple
        agent = random.choice(agents)
        # Podríamos añadir lógica para que diferentes tipos de agentes actúen
        # con diferentes frecuencias o bajo condiciones específicas

        # Ejecutar la acción del agente seleccionado
        agent.act()

        # Imprimir resumen del grafo periódicamente o al final
        if (step + 1) % 20 == 0 or step == num_steps - 1:
             graph.print_summary()

        # Pausa para poder observar la simulación en tiempo real
        time.sleep(step_delay)

    print("\n--- Simulation Finished ---")
    # Imprimir resumen final
    graph.print_summary()


# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    # Configurar y ejecutar la simulación
    run_simulation(num_steps=80,      # Número total de pasos de simulación
                   num_proposers=2,  # Número de agentes que proponen nodos
                   num_evaluators=5, # Número de agentes que evalúan nodos
                   num_combiners=2,  # Número de agentes que crean enlaces
                   step_delay=0.05)  # Pausa entre pasos (en segundos)
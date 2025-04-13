import random
import math
import time
import argparse # Para argumentos de línea de comandos
import yaml     # Para leer archivos YAML
from collections import Counter

# --- 1. Definiciones del Grafo de Síntesis (Sin cambios respecto a versión anterior con config) ---

class KnowledgeComponent:
    """Representa un nodo (V') en el Grafo de Síntesis."""
    def __init__(self, node_id, content="Abstract Concept", initial_state=0.1, keywords=None):
        self.id = node_id
        self.content = content
        self.state = initial_state # Confianza/Calidad (sj) - Rango [0, 1]
        self.keywords = keywords if keywords else set()
        self.connections_out = {}
        self.connections_in = {}

    def update_state(self, new_state):
        """Actualiza el estado asegurando que esté en [0, 1]."""
        self.state = max(0.0, min(1.0, new_state))

    def add_connection(self, target_node, utility):
        """Añade una conexión saliente (arista E')."""
        if target_node.id != self.id:
             self.connections_out[target_node.id] = utility
             target_node.connections_in[self.id] = utility

    def __repr__(self):
        """Representación del nodo."""
        kw_str = f", KW={list(self.keywords)}" if self.keywords else ""
        return f"Node({self.id}, S={self.state:.3f}{kw_str})"

class CollectiveSynthesisGraph:
    """Gestiona el Grafo de Síntesis (G')."""
    def __init__(self):
        self.nodes = {}
        self.next_node_id = 0

    def add_node(self, content="Abstract Concept", initial_state=0.1, keywords=None):
        """Añade un nuevo nodo al grafo."""
        node_id = self.next_node_id
        kw = set(keywords) if keywords else set()
        new_node = KnowledgeComponent(node_id, content, initial_state, kw)
        self.nodes[node_id] = new_node
        self.next_node_id += 1
        return new_node

    def add_edge(self, source_id, target_id, utility):
        """Añade una arista dirigida."""
        if source_id in self.nodes and target_id in self.nodes and source_id != target_id:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            if target_id not in source_node.connections_out:
                 source_node.add_connection(target_node, utility)
                 return True
        return False

    def get_node(self, node_id):
        """Obtiene un nodo por su ID."""
        return self.nodes.get(node_id)

    def get_nodes_sorted_by_state(self, descending=True):
        """Devuelve nodos ordenados por estado."""
        if not self.nodes:
            return []
        return sorted(self.nodes.values(), key=lambda node: node.state, reverse=descending)

    def get_random_node_biased(self):
        """Obtiene nodo aleatorio, con sesgo hacia mayor estado."""
        if not self.nodes:
            return None
        nodes = list(self.nodes.values())
        weights = [(node.state**2) + 0.01 for node in nodes]
        if sum(weights) == 0:
             return random.choice(nodes) if nodes else None
        return random.choices(nodes, weights=weights, k=1)[0]

    def print_summary(self):
        """Imprime resumen del grafo."""
        num_nodes = len(self.nodes)
        print(f"--- Graph Summary (Nodes: {num_nodes}) ---")
        if num_nodes == 0:
            print("  Graph is empty.")
            print("-" * 40)
            return
        top_n = min(5, num_nodes)
        sorted_nodes = self.get_nodes_sorted_by_state()
        for i in range(top_n):
            node = sorted_nodes[i]
            outs = len(node.connections_out)
            ins = len(node.connections_in)
            print(f"  Top {i+1}: {node!r} - Content: '{node.content[:30]}...' (Out: {outs}, In: {ins})")
        avg_state = sum(n.state for n in self.nodes.values()) / num_nodes
        avg_keywords = sum(len(n.keywords) for n in self.nodes.values()) / num_nodes if num_nodes > 0 else 0
        num_edges = sum(len(n.connections_out) for n in self.nodes.values())
        print(f"  Avg State: {avg_state:.3f}, Avg Keywords: {avg_keywords:.2f}, Total Edges: {num_edges}")
        print("-" * 40)

# --- 2. Definiciones de Sintetizadores (Evaluator modificado, todos aceptan config) ---

class Synthesizer:
    """Clase base para los agentes."""
    def __init__(self, agent_id, graph, config): # Acepta config
        self.id = agent_id
        self.graph = graph
        self.config = config # Guarda config

    def act(self):
        raise NotImplementedError

class ProposerAgent(Synthesizer):
    """Propone nuevos nodos."""
    # __init__ hereda y usa self.config si fuera necesario
    def act(self):
        source_node = self.graph.get_random_node_biased()
        if source_node is None:
            keywords = {"inicio", "semilla"}
            new_node = self.graph.add_node(content="Initial Seed", initial_state=0.2, keywords=keywords)
            print(f"Proposer {self.id}: Proposed initial {new_node!r}")
            return

        new_kw = f"kw_{self.graph.next_node_id}"
        source_kw = random.choice(list(source_node.keywords)) if source_node.keywords else "related"
        new_keywords = {source_kw, new_kw}
        new_content = f"Related concept to {source_node.id} about {new_kw}"
        initial_state = max(0.05, source_node.state * random.uniform(0.3, 0.8))
        new_node = self.graph.add_node(content=new_content, initial_state=initial_state, keywords=new_keywords)

        utility = (source_node.state * 0.5 + random.uniform(-0.3, 0.7))
        utility = max(-1.0, min(1.0, utility))
        self.graph.add_edge(source_node.id, new_node.id, utility)
        print(f"Proposer {self.id}: Proposed {new_node!r} linked from {source_node!r} with U={utility:.2f}")

class EvaluatorAgent(Synthesizer):
    """Evalúa nodos con lógica mejorada (decay, keywords, inconsistencia básica)."""
    def __init__(self, agent_id, graph, config): # Acepta config
        super().__init__(agent_id, graph, config)
        # Usa parámetros de la configuración, con valores por defecto
        self.learning_rate = config.get('evaluator_learning_rate', 0.1)
        self.keyword_boost = config.get('evaluator_keyword_boost', 0.05)
        self.decay_rate = config.get('evaluator_decay_rate', 0.01) # <-- Nuevo parámetro

    def act(self):
        target_node = self.graph.get_random_node_biased()
        if target_node is None: return

        influence_sum = 0.0
        weight_sum = 0.0
        shared_keywords_score_factor = 0.0

        if not target_node.connections_in:
            # Decaimiento para nodos aislados
            influence_target = target_node.state * (1 - self.decay_rate)
        else:
            # Calcular influencia de los vecinos entrantes y boost por keywords
            for source_id, utility_uji in target_node.connections_in.items():
                source_node = self.graph.get_node(source_id)
                if source_node:
                    influence = source_node.state * utility_uji
                    influence_sum += influence
                    weight = abs(utility_uji)
                    weight_sum += weight
                    # Keyword boost si el vecino es relevante (estado alto, utilidad positiva)
                    if source_node.state > 0.5 and utility_uji > 0.1:
                        common_keywords = target_node.keywords.intersection(source_node.keywords)
                        if common_keywords:
                             boost = len(common_keywords) * source_node.state * weight * self.keyword_boost
                             shared_keywords_score_factor += boost

            # Calcular objetivo base y añadir boost normalizado
            if weight_sum > 0.01:
                 base_influence_target = influence_sum / weight_sum
                 keyword_influence_boost = shared_keywords_score_factor / weight_sum
                 influence_target = base_influence_target + keyword_influence_boost
            else:
                 influence_target = target_node.state # Mantener estado si no hay influencia

            # Penalización por inconsistencias lógicas (simplificada)
            # Penaliza si recibe enlace negativo de un nodo fuente de alta confianza
            for source_id, utility_uji in target_node.connections_in.items():
               source_node = self.graph.get_node(source_id)
               if source_node and utility_uji < 0 and source_node.state > 0.7:
                   influence_target *= 0.9 # Factor de penalización (ej. 10%)

        # Limitar objetivo y actualizar estado
        influence_target = max(-0.5, min(1.5, influence_target)) # Limita el objetivo
        current_state = target_node.state
        new_state = current_state + self.learning_rate * (influence_target - current_state)
        # Aplicar decaimiento general (opcional, además del de aislamiento)
        # new_state *= (1 - self.decay_rate * 0.1) # Decaimiento general más lento

        target_node.update_state(new_state) # Asegura que el estado quede en [0, 1]
        print(f"Evaluator {self.id}: Evaluated {target_node!r}. State: {current_state:.3f} -> {target_node.state:.3f} (Target: {influence_target:.3f})")


class CombinerAgent(Synthesizer):
    """Combina nodos existentes."""
    def __init__(self, agent_id, graph, config): # Acepta config
        super().__init__(agent_id, graph, config)
        # Usa parámetro de la configuración
        self.compatibility_threshold = config.get('combiner_compatibility_threshold', 0.6)

    def act(self):
        if len(self.graph.nodes) < 2: return

        node_a = self.graph.get_random_node_biased()
        node_b = self.graph.get_random_node_biased()

        if node_a is None or node_b is None or node_a.id == node_b.id \
           or node_b.id in node_a.connections_out \
           or node_a.id in node_b.connections_out:
            return

        state_product = node_a.state * node_b.state
        common_keywords = node_a.keywords.intersection(node_b.keywords)
        max_possible_keywords = len(node_a.keywords.union(node_b.keywords))
        keyword_similarity = len(common_keywords) / max_possible_keywords if max_possible_keywords > 0 else 0
        compatibility_score = (state_product * 0.6) + (keyword_similarity * 0.4)

        if compatibility_score >= self.compatibility_threshold:
            utility = compatibility_score * ((node_a.state + node_b.state) / 2.0)
            utility = max(-1.0, min(1.0, utility))
            if self.graph.add_edge(node_a.id, node_b.id, utility):
                 print(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} (Score: {compatibility_score:.2f}, U={utility:.2f})")

# --- 3. Simulación (Usa config) ---

def run_simulation(config): # Acepta diccionario de configuración
    """Ejecuta la simulación del MSC."""
    num_steps = config.get('simulation_steps', 100)
    num_proposers = config.get('num_proposers', 3)
    num_evaluators = config.get('num_evaluators', 6)
    num_combiners = config.get('num_combiners', 2)
    step_delay = config.get('step_delay', 0.1)

    graph = CollectiveSynthesisGraph()
    agents = []
    # Pasa config a los constructores de agentes
    for i in range(num_proposers):
        agents.append(ProposerAgent(f"P{i}", graph, config))
    for i in range(num_evaluators):
        agents.append(EvaluatorAgent(f"E{i}", graph, config))
    for i in range(num_combiners):
        agents.append(CombinerAgent(f"C{i}", graph, config))

    print(f"--- Starting MSC Simulation ({num_steps} steps) ---")
    print(f"Configuration: {config}") # Imprime config usada
    print(f"Agents: {len(agents)} ({num_proposers}P, {num_evaluators}E, {num_combiners}C)")

    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        if not agents: break
        agent = random.choice(agents)
        agent.act()
        if (step + 1) % 20 == 0 or step == num_steps - 1:
             graph.print_summary()
        time.sleep(step_delay)

    print("\n--- Simulation Finished ---")
    graph.print_summary()

# --- 4. Carga de Configuración y Punto de Entrada (Actualizado) ---

def load_config(args):
    """Carga la configuración desde defaults, archivo YAML y argumentos CLI."""
    # Valores por defecto (incluyendo el nuevo parámetro)
    config = {
        'simulation_steps': 100,
        'num_proposers': 3,
        'num_evaluators': 6,
        'num_combiners': 2,
        'step_delay': 0.1,
        'evaluator_learning_rate': 0.1,
        'evaluator_keyword_boost': 0.05,
        'evaluator_decay_rate': 0.01, # <-- Default para decay rate
        'combiner_compatibility_threshold': 0.6
    }

    # 1. Cargar desde YAML si se especifica
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config.update(yaml_config)
                    print(f"Loaded configuration from: {args.config}")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {args.config}. Using defaults/args.")
        except Exception as e:
            print(f"Error loading config file {args.config}: {e}. Using defaults/args.")

    # 2. Sobrescribir con argumentos CLI si se especifican
    cli_args = {key: value for key, value in vars(args).items() if value is not None and key != 'config'}
    config.update(cli_args)

    return config

if __name__ == "__main__":
    # Configurar parser de argumentos (añadir el nuevo)
    parser = argparse.ArgumentParser(description="Run the Collective Synthesis Framework (MSC) Simulation.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('--simulation_steps', type=int, help='Number of simulation steps.')
    parser.add_argument('--num_proposers', type=int, help='Number of proposer agents.')
    parser.add_argument('--num_evaluators', type=int, help='Number of evaluator agents.')
    parser.add_argument('--num_combiners', type=int, help='Number of combiner agents.')
    parser.add_argument('--step_delay', type=float, help='Delay between simulation steps (in seconds).')
    parser.add_argument('--evaluator_learning_rate', type=float, help='Learning rate for evaluator agents.')
    parser.add_argument('--evaluator_keyword_boost', type=float, help='Keyword boost factor for evaluator agents.')
    parser.add_argument('--evaluator_decay_rate', type=float, help='State decay rate for evaluator agents.') # <-- Argumento añadido
    parser.add_argument('--combiner_compatibility_threshold', type=float, help='Threshold for combiner agents.')

    args = parser.parse_args()
    final_config = load_config(args)
    run_simulation(final_config)
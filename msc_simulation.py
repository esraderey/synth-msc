import random
import math
import time
import argparse
import yaml
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
from sentence_transformers import SentenceTransformer  # <-- NUEVA IMPORTACIÓN

# --- Cargar Modelo de Embeddings de Texto (Globalmente) ---
print("Loading Sentence Transformer model...")
embedding_model_name = 'all-MiniLM-L6-v2'
text_embedding_model = SentenceTransformer(embedding_model_name)
TEXT_EMBEDDING_DIM = text_embedding_model.get_sentence_embedding_dimension()
print(f"Sentence Transformer model '{embedding_model_name}' loaded (Dim: {TEXT_EMBEDDING_DIM}).")
# --- Fin Carga Modelo ---

# --- Definición del Modelo GNN Básico ---
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, embedding_dim):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- 1. Definiciones del Grafo de Síntesis ---
class KnowledgeComponent:
    def __init__(self, node_id, content="Abstract Concept", initial_state=0.1, keywords=None):
        self.id = node_id
        self.content = content
        self.state = initial_state
        self.keywords = keywords if keywords else set()
        self.connections_out = {}
        self.connections_in = {}
    def update_state(self, new_state):
        self.state = max(0.0, min(1.0, new_state))
    def add_connection(self, target_node, utility):
        if target_node.id != self.id:
            self.connections_out[target_node.id] = utility
            target_node.connections_in[self.id] = utility
    def __repr__(self):
        kw_str = f", KW={list(self.keywords)}" if self.keywords else ""
        return f"Node({self.id}, S={self.state:.3f}{kw_str})"

class CollectiveSynthesisGraph:
    """Gestiona el Grafo de Síntesis (G') e integra la GNN."""
    def __init__(self, config):
        self.nodes = {}
        self.next_node_id = 0
        self.config = config

        # --- Modificación GNN: Dimensión de entrada ahora es dinámica ---
        # 4 características base + dimensión del embedding de texto
        num_base_features = 4
        num_node_features = num_base_features + TEXT_EMBEDDING_DIM
        config['gnn_input_dim'] = num_node_features  # Actualizar config con la dimensión real

        hidden_channels = config.get('gnn_hidden_dim', 16)
        embedding_dim = config.get('gnn_embedding_dim', 8)

        self.gnn_model = GNNModel(num_node_features, hidden_channels, embedding_dim)
        self.node_embeddings = {}
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        print(f"GNN Initialized: Input={num_node_features} (Base:{num_base_features}+Text:{TEXT_EMBEDDING_DIM}), Hidden={hidden_channels}, Embedding={embedding_dim}")

    # --- MÉTODO _prepare_pyg_data ACTUALIZADO CON EMBEDDINGS DE TEXTO ---
    def _prepare_pyg_data(self):
        """Prepara los datos del grafo actual para PyTorch Geometric, incluyendo embeddings de texto."""
        if not self.nodes:
            return None, None, None

        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        num_nodes = len(self.nodes)
        num_base_features = 4
        num_node_features = num_base_features + TEXT_EMBEDDING_DIM

        print(f"GNN Prep: Preparing features for {num_nodes} nodes (Dim: {num_node_features})...")
        all_node_content = [self.nodes[self.idx_to_node_id[i]].content for i in range(num_nodes)]

        try:
            with torch.no_grad():
                text_embeddings_np = text_embedding_model.encode(all_node_content, convert_to_numpy=True, show_progress_bar=False)
            text_embeddings = torch.tensor(text_embeddings_np, dtype=torch.float)
        except Exception as e:
            print(f"Error generating text embeddings: {e}. Using zero vectors.")
            text_embeddings = torch.zeros((num_nodes, TEXT_EMBEDDING_DIM), dtype=torch.float)

        node_features = torch.zeros((num_nodes, num_node_features), dtype=torch.float)
        for i in range(num_nodes):
            node_id = self.idx_to_node_id[i]
            node = self.nodes[node_id]
            node_features[i, 0] = node.state
            node_features[i, 1] = float(len(node.connections_in))
            node_features[i, 2] = float(len(node.connections_out))
            node_features[i, 3] = float(len(node.keywords))
            if i < text_embeddings.shape[0]:
                node_features[i, num_base_features:] = text_embeddings[i]

        source_indices = []
        target_indices = []
        for source_node_id, node in self.nodes.items():
            if source_node_id not in self.node_id_to_idx:
                continue
            source_idx = self.node_id_to_idx[source_node_id]
            for target_node_id in node.connections_out.keys():
                if target_node_id in self.node_id_to_idx:
                    target_idx = self.node_id_to_idx[target_node_id]
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

        print(f"GNN Prep: Features shape: {node_features.shape}, Edge index shape: {edge_index.shape}")
        return node_features, edge_index

    def update_embeddings(self):
        if not self.nodes:
            self.node_embeddings = {}
            return
        print("GNN Prep: Calling prepare data...")
        node_features, edge_index = self._prepare_pyg_data()
        if node_features is None or edge_index is None:
            print("GNN Prep: Data is None.")
            return
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(1) if edge_index.shape[0] == 2 else edge_index.unsqueeze(0)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t()
        if edge_index.shape[1] == 0:
            print("GNN Warning: No edges found. Skipping embedding update.")
            self.node_embeddings = {nid: torch.zeros(self.config.get('gnn_embedding_dim', 8)) for nid in self.nodes}
            return
        print("GNN: Running forward pass...")
        self.gnn_model.eval()
        with torch.no_grad():
            try:
                all_embeddings_tensor = self.gnn_model(node_features, edge_index)
                self.node_embeddings = {self.idx_to_node_id[i]: embedding for i, embedding in enumerate(all_embeddings_tensor) if i in self.idx_to_node_id}
            except IndexError as ie:
                print(f"GNN IndexError during forward pass: {ie}. Indices might be out of bounds. Num nodes: {node_features.shape[0]}, Edge index max: {edge_index.max() if edge_index.numel() > 0 else 'N/A'}")
            except Exception as e:
                print(f"GNN Error during forward pass: {e}")

    def add_node(self, content="Abstract Concept", initial_state=0.1, keywords=None):
        node_id = self.next_node_id
        kw = set(keywords) if keywords else set()
        new_node = KnowledgeComponent(node_id, content, initial_state, kw)
        self.nodes[node_id] = new_node
        self.next_node_id += 1
        return new_node

    def add_edge(self, source_id, target_id, utility):
        if source_id in self.nodes and target_id in self.nodes and source_id != target_id:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            if target_id not in source_node.connections_out:
                source_node.add_connection(target_node, utility)
                return True
        return False

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_embedding(self, node_id):
        return self.node_embeddings.get(node_id, None)

    def get_nodes_sorted_by_state(self, descending=True):
        if not self.nodes:
            return []
        return sorted(self.nodes.values(), key=lambda node: node.state, reverse=descending)

    def get_random_node_biased(self):
        if not self.nodes:
            return None
        nodes = list(self.nodes.values())
        weights = [(node.state**2) + 0.01 for node in nodes]
        if sum(weights) <= 0.0:
            return random.choice(nodes) if nodes else None
        return random.choices(nodes, weights=weights, k=1)[0]

    def print_summary(self):
        num_nodes = len(self.nodes)
        print(f"--- Graph Summary (Nodes: {num_nodes}, Embeddings: {len(self.node_embeddings)}) ---")
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
        if num_nodes > 0:
            avg_state = sum(n.state for n in self.nodes.values()) / num_nodes
            avg_keywords = sum(len(n.keywords) for n in self.nodes.values()) / num_nodes
            num_edges = sum(len(n.connections_out) for n in self.nodes.values())
            print(f"  Avg State: {avg_state:.3f}, Avg Keywords: {avg_keywords:.2f}, Total Edges: {num_edges}")
        print("-" * 40)

    def visualize_graph(self, config):
        if not self.nodes:
            print("Cannot visualize empty graph.")
            return
        print("Generating graph visualization...")
        G = nx.DiGraph()
        node_labels = {}
        node_sizes = []
        node_colors = []
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
            node_labels[node_id] = f"{node_id}\nS={node.state:.2f}"
            node_sizes.append(100 + node.state * 1500)
            node_colors.append(node.state)
        edge_list = []
        edge_weights = []
        edge_colors = []
        for node_id, node in self.nodes.items():
            for target_id, utility in node.connections_out.items():
                if target_id in G:
                    edge_list.append((node_id, target_id))
                    edge_weights.append(1 + abs(utility) * 4)
                    edge_colors.append(utility)
        if not G.nodes:
            print("No nodes to visualize.")
            return
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_title("MSC Graph Visualization")
        ax.axis('off')
        try:
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        except Exception as e:
            print(f"Error calculating layout: {e}. Using random layout.")
            pos = nx.random_layout(G)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap=plt.cm.viridis, node_size=node_sizes, alpha=0.8)
        edges = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm, width=edge_weights, alpha=0.6, arrows=True, arrowstyle='->', arrowsize=15)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=8)
        if node_colors:
            norm_nodes = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
            sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_nodes)
            sm_nodes.set_array([])
            cbar_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.5)
            cbar_nodes.set_label('Node State (sj)')
        if edge_colors:
            norm_edges = plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
            sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm_edges)
            sm_edges.set_array([])
            cbar_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.5)
            cbar_edges.set_label('Edge Utility (uij)')
        plt.show()

    def save_state(self, base_file_path):
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        print(f"Attempting to save state to base path: {base_file_path}")
        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            keywords_str = ",".join(sorted(list(node.keywords)))
            G.add_node(node_id, state=node.state, content=node.content, keywords=keywords_str)
        for node_id, node in self.nodes.items():
            for target_id, utility in node.connections_out.items():
                if node_id in G and target_id in G:
                    G.add_edge(node_id, target_id, utility=utility)
        try:
            nx.write_graphml(G, graphml_path)
            print(f"Graph structure saved to {graphml_path}")
        except Exception as e:
            print(f"Error saving GraphML to {graphml_path}: {e}")
        try:
            torch.save(self.gnn_model.state_dict(), gnn_model_path)
            print(f"GNN model state saved to {gnn_model_path}")
        except Exception as e:
            print(f"Error saving GNN model state to {gnn_model_path}: {e}")
        try:
            detached_embeddings = {node_id: emb.detach().cpu() for node_id, emb in self.node_embeddings.items()}
            torch.save(detached_embeddings, embeddings_path)
            print(f"Node embeddings saved to {embeddings_path}")
        except Exception as e:
            print(f"Error saving node embeddings to {embeddings_path}: {e}")

    # --- MÉTODO load_state (REVISADO PARA CORREGIR SYNTAXERROR) ---
    def load_state(self, base_file_path):
        """Carga el estado completo: grafo (GraphML), modelo GNN (.pth), y embeddings (.pth)."""
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        print(f"Attempting to load state from base path: {base_file_path}")

        # 1. Cargar Grafo (GraphML)
        try:  # Bloque try principal para carga de GraphML
            if not os.path.exists(graphml_path):
                 print(f"Error: GraphML file not found at {graphml_path}. Cannot load state.")
                 return False  # Indicar fallo

            G = nx.read_graphml(graphml_path)
            self.nodes = {}  # Limpiar grafo actual
            self.next_node_id = 0  # Reiniciar contador
            max_id = -1

            # Cargar Nodos
            print("Loading nodes...")
            for node_id_str, data in G.nodes(data=True):
                 try:  # Try interno para conversión de ID de nodo
                      node_id = int(node_id_str)
                 except ValueError:
                      print(f"Warning: Skipping node with non-integer ID '{node_id_str}' from GraphML.")
                      continue  # Saltar al siguiente nodo

                 # Convertir string de keywords de nuevo a set
                 keywords_str = data.get('keywords', '')
                 keywords = set(k for k in keywords_str.split(',') if k)

                 # Crear el componente de conocimiento
                 new_node = KnowledgeComponent(
                     node_id,
                     content=data.get('content', ""),
                     initial_state=float(data.get('state', 0.1)),
                     keywords=keywords
                 )
                 self.nodes[node_id] = new_node
                 max_id = max(max_id, node_id)
            # Fin del bucle de nodos

            self.next_node_id = max_id + 1  # Ajustar next_node_id al máximo cargado + 1
            print(f"Loaded {len(self.nodes)} nodes. Next node ID: {self.next_node_id}")

            # Cargar Aristas
            print("Loading edges...")
            edge_count = 0
            for source_id_str, target_id_str, data in G.edges(data=True):
                  try:  # Try interno para conversión de IDs de arista
                       source_id = int(source_id_str)
                       target_id = int(target_id_str)
                       # Asegurarse que ambos nodos existen antes de añadir la arista
                       if source_id in self.nodes and target_id in self.nodes:
                            self.add_edge(source_id, target_id, float(data.get('utility', 0.0)))
                            edge_count += 1
                  except ValueError:  # Captura error si IDs no son enteros
                       print(f"Warning: Skipping edge with non-integer source/target ID ('{source_id_str}' -> '{target_id_str}') from GraphML.")
                       continue  # Saltar a la siguiente arista
            # Fin del bucle de aristas
            print(f"Loaded {edge_count} edges.")
            print(f"Graph structure loaded from {graphml_path}")
        except Exception as e:
             print(f"CRITICAL Error loading GraphML from {graphml_path}: {e}")
             return False  # Indicar fallo

        # 2. Cargar Estado del Modelo GNN
        try:  # Try para cargar modelo GNN
             if not os.path.exists(gnn_model_path):
                  print(f"Warning: GNN model state file not found at {gnn_model_path}. Using initialized GNN model.")
             else:
                  print(f"Loading GNN model state from {gnn_model_path}...")
                  state_dict = torch.load(gnn_model_path)
                  self.gnn_model.load_state_dict(state_dict)
                  self.gnn_model.eval()  # Poner en modo evaluación
                  print(f"GNN model state loaded successfully.")
        except Exception as e:
             print(f"Error loading GNN model state from {gnn_model_path}: {e}. Using initialized GNN model.")

        # 3. Cargar Embeddings Calculados
        try:  # Try para cargar embeddings
             if not os.path.exists(embeddings_path):
                  print(f"Warning: Node embeddings file not found at {embeddings_path}. Embeddings dictionary initialized empty.")
                  self.node_embeddings = {}
             else:
                  print(f"Loading node embeddings from {embeddings_path}...")
                  self.node_embeddings = torch.load(embeddings_path)
                  print(f"Node embeddings loaded successfully for {len(self.node_embeddings)} nodes.")
        except Exception as e:
             print(f"Error loading node embeddings from {embeddings_path}: {e}. Embeddings dictionary initialized empty.")
             self.node_embeddings = {}

        # 4. Recalcular mapeos ID <-> Índice después de cargar
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        print("Node ID mappings recalculated.")

        return True  # Indicar éxito en la carga general

# --- 2. Definiciones de Sintetizadores ---
class Synthesizer:
    def __init__(self, agent_id, graph, config):
        self.id = agent_id
        self.graph = graph
        self.config = config
    def act(self):
        raise NotImplementedError

class ProposerAgent(Synthesizer):
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
    def __init__(self, agent_id, graph, config):
        super().__init__(agent_id, graph, config)
        self.learning_rate = config.get('evaluator_learning_rate', 0.1)
        self.similarity_boost_factor = config.get('evaluator_similarity_boost', 0.05)
        self.decay_rate = config.get('evaluator_decay_rate', 0.01)
    def calculate_cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return (sim + 1) / 2
    def act(self):
        target_node = self.graph.get_random_node_biased()
        if target_node is None:
            return
        target_embedding = self.graph.get_embedding(target_node.id)
        influence_sum = 0.0
        weight_sum = 0.0
        accumulated_similarity_boost = 0.0
        penalty_factor = 1.0
        if not target_node.connections_in:
            influence_target = target_node.state * (1 - self.decay_rate)
        else:
            for source_id, utility_uji in target_node.connections_in.items():
                source_node = self.graph.get_node(source_id)
                if source_node:
                    influence = source_node.state * utility_uji
                    influence_sum += influence
                    weight = abs(utility_uji)
                    weight_sum += weight
                    source_embedding = self.graph.get_embedding(source_node.id)
                    if source_node and source_node.state > 0.5 and utility_uji > 0.1 and target_embedding is not None and source_embedding is not None:
                        similarity = self.calculate_cosine_similarity(target_embedding, source_embedding)
                        boost_from_source = similarity * source_node.state * weight * self.similarity_boost_factor
                        accumulated_similarity_boost += boost_from_source
                    if source_node and utility_uji < 0 and source_node.state > 0.7:
                        penalty_factor *= 0.9
            if weight_sum > 0.01:
                base_influence_target = influence_sum / weight_sum
                normalized_similarity_boost = accumulated_similarity_boost / weight_sum
                influence_target = base_influence_target + normalized_similarity_boost
            else:
                influence_target = target_node.state
            influence_target *= penalty_factor
        influence_target = max(-0.5, min(1.5, influence_target))
        current_state = target_node.state
        new_state = current_state + self.learning_rate * (influence_target - current_state)
        target_node.update_state(new_state)
        print(f"Evaluator {self.id}: Evaluated {target_node!r}. State: {current_state:.3f} -> {target_node.state:.3f} (Target: {influence_target:.3f})")

# --- CLASE CombinerAgent (REVISADA INDENTACIÓN) ---
class CombinerAgent(Synthesizer):
    """Combina nodos existentes basándose en la similitud de sus embeddings o fallback."""
    def __init__(self, agent_id, graph, config):
        super().__init__(agent_id, graph, config)
        self.similarity_threshold = config.get('combiner_similarity_threshold', 0.7)
        self.compatibility_threshold = config.get('combiner_compatibility_threshold', 0.6)

    def calculate_cosine_similarity(self, emb1, emb2):
        """Calcula la similitud coseno [-1, 1]. Devuelve 0.0 si falta un embedding."""
        if emb1 is None or emb2 is None:
            return 0.0
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def act(self):
        if len(self.graph.nodes) < 2:
            return

        node_a = self.graph.get_random_node_biased()
        node_b = self.graph.get_random_node_biased()

        if node_a is None or node_b is None or node_a.id == node_b.id \
           or node_b.id in node_a.connections_out \
           or node_a.id in node_b.connections_out:
            return

        emb_a = self.graph.get_embedding(node_a.id)
        emb_b = self.graph.get_embedding(node_b.id)
        edge_added = False  # Flag para saber si ya se añadió una arista

        # --- Lógica Principal con Embeddings ---
        if emb_a is not None and emb_b is not None:
            cosine_sim = self.calculate_cosine_similarity(emb_a, emb_b)
            normalized_sim = (cosine_sim + 1) / 2  # Similitud en [0, 1] para el umbral

            print(f"Combiner {self.id}: Checking nodes {node_a.id} & {node_b.id}. Embedding Similarity: {cosine_sim:.3f} (Norm: {normalized_sim:.3f})")

            if normalized_sim >= self.similarity_threshold:
                utility = max(-1.0, min(1.0, cosine_sim))
                if self.graph.add_edge(node_a.id, node_b.id, utility):
                    print(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} based on embedding similarity (Score: {normalized_sim:.2f}, U={utility:.2f})")
                    edge_added = True

        # --- Lógica de Fallback (si no se añadió con embeddings) ---
        if not edge_added:
            state_product = node_a.state * node_b.state
            common_keywords = node_a.keywords.intersection(node_b.keywords)
            max_possible_keywords = len(node_a.keywords.union(node_b.keywords))
            keyword_similarity = len(common_keywords) / max_possible_keywords if max_possible_keywords > 0 else 0
            compatibility_score = (state_product * 0.6) + (keyword_similarity * 0.4)

            if compatibility_score >= self.compatibility_threshold:
                utility = compatibility_score * ((node_a.state + node_b.state) / 2.0)
                utility = max(-1.0, min(1.0, utility))
                if self.graph.add_edge(node_a.id, node_b.id, utility):
                    print(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} using FALLBACK logic (Score: {compatibility_score:.2f}, U={utility:.2f})")

# --- 3. Simulación ---
def run_simulation(config):
    graph = CollectiveSynthesisGraph(config)
    load_path = config.get('load_state', None)
    if load_path:
        print(f"\n--- Loading state from: {load_path} ---")
        if not graph.load_state(load_path):
            print("Failed to load state. Starting fresh simulation.")
        else:
            print("GNN: Updating embeddings after load...")
            graph.update_embeddings()
            if graph.node_embeddings:
                print(f"GNN: Embeddings calculated for {len(graph.node_embeddings)} nodes.")
    num_steps = config.get('simulation_steps', 100)
    step_delay = config.get('step_delay', 0.1)
    visualize_at_end = config.get('visualize_graph', False)
    gnn_update_frequency = config.get('gnn_update_frequency', 10)
    agents = []
    num_proposers = config.get('num_proposers', 3)
    num_evaluators = config.get('num_evaluators', 6)
    num_combiners = config.get('num_combiners', 2)
    for i in range(num_proposers):
        agents.append(ProposerAgent(f"P{i}", graph, config))
    for i in range(num_evaluators):
        agents.append(EvaluatorAgent(f"E{i}", graph, config))
    for i in range(num_combiners):
        agents.append(CombinerAgent(f"C{i}", graph, config))
    print(f"--- Starting MSC Simulation ({num_steps} steps) ---")
    print(f"Configuration: {config}")
    print(f"Agents: {len(agents)} ({num_proposers}P, {num_evaluators}E, {num_combiners}C)")
    for step in range(num_steps):
        print(f"\n--- Step {step + 1}/{num_steps} ---")
        if not agents:
            break
        if step > 0 and step % gnn_update_frequency == 0:
            print(f"GNN: Updating embeddings at step {step+1}...")
            graph.update_embeddings()
            if graph.node_embeddings:
                print(f"GNN: Embeddings available for {len(graph.node_embeddings)} nodes.")
        agent = random.choice(agents)
        agent.act()
        num_nodes_current = len(graph.nodes)
        if not visualize_at_end and ((step + 1) % 20 == 0 or step == num_steps - 1):
            graph.print_summary()
        elif visualize_at_end and step == num_steps - 1:
            graph.print_summary()
        time.sleep(step_delay)
    print("\n--- Simulation Finished ---")
    print("GNN: Final embedding update...")
    graph.update_embeddings()
    if graph.node_embeddings:
        print(f"GNN: Final embeddings calculated for {len(graph.node_embeddings)} nodes.")
    save_path = config.get('save_state', None)
    if save_path:
        print(f"\n--- Saving state to: {save_path} ---")
        graph.save_state(save_path)
    if visualize_at_end:
        if graph.nodes:
            graph.visualize_graph(config)
        else:
            print("Graph is empty, skipping visualization.")
            graph.print_summary()
    elif graph.nodes:
        graph.print_summary()

# --- 4. Carga de Configuración y Punto de Entrada ---
def load_config(args):
    """Carga la configuración."""
    config = {
        'simulation_steps': 100,
        'num_proposers': 3,
        'num_evaluators': 6,
        'num_combiners': 2,
        'step_delay': 0.1,
        'evaluator_learning_rate': 0.1,
        'evaluator_similarity_boost': 0.05,
        'evaluator_decay_rate': 0.01,
        'combiner_compatibility_threshold': 0.6,
        'combiner_similarity_threshold': 0.7,
        'visualize_graph': False,
        'gnn_input_dim': 4,
        'gnn_hidden_dim': 16,
        'gnn_embedding_dim': 8,
        'gnn_update_frequency': 10,
        'save_state': None,
        'load_state': None
    }
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config.update(yaml_config)
                    print(f"Loaded config from: {args.config}")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {args.config}.")
        except Exception as e:
            print(f"Error loading config file {args.config}: {e}.")
    cli_args = vars(args).copy()
    config_file_path = cli_args.pop('config', None)
    if 'visualize_graph' in cli_args:
        if cli_args['visualize_graph']:
            config['visualize_graph'] = True
        del cli_args['visualize_graph']
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Collective Synthesis Framework (MSC) Simulation with GNN.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('--simulation_steps', type=int, help='Number of simulation steps.')
    parser.add_argument('--num_proposers', type=int, help='Number of proposer agents.')
    parser.add_argument('--num_evaluators', type=int, help='Number of evaluator agents.')
    parser.add_argument('--num_combiners', type=int, help='Number of combiner agents.')
    parser.add_argument('--step_delay', type=float, help='Delay between simulation steps.')
    parser.add_argument('--evaluator_learning_rate', type=float, help='Learning rate for evaluators.')
    parser.add_argument('--evaluator_similarity_boost', type=float, help='Embedding similarity boost factor.')
    parser.add_argument('--evaluator_decay_rate', type=float, help='State decay rate.')
    parser.add_argument('--combiner_compatibility_threshold', type=float, help='Fallback threshold for combiners.')
    parser.add_argument('--combiner_similarity_threshold', type=float, help='Embedding similarity threshold [0,1].')
    parser.add_argument('--gnn_input_dim', type=int, help='Input feature dimension for GNN nodes (default should match feature prep).')
    parser.add_argument('--gnn_hidden_dim', type=int, help='Hidden dimension for GNN layers.')
    parser.add_argument('--gnn_embedding_dim', type=int, help='Output embedding dimension for GNN.')
    parser.add_argument('--gnn_update_frequency', type=int, help='Frequency (in steps) to update GNN embeddings.')
    parser.add_argument('--visualize_graph', action='store_true', help='Visualize graph at the end.')
    parser.add_argument('--save_state', type=str, help='Base path to save simulation state.')
    parser.add_argument('--load_state', type=str, help='Base path to load simulation state from.')
    
    args = parser.parse_args()
    final_config = load_config(args)
    run_simulation(final_config)




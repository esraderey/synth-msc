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

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: 'sentence-transformers' not found. Text embeddings disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

import logging
import threading
from flask import Flask, jsonify

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cargar Modelo de Embeddings ---
text_embedding_model = None
TEXT_EMBEDDING_DIM = 0
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        logging.info("Loading Sentence Transformer model ('all-MiniLM-L6-v2')...")
        embedding_model_name = 'all-MiniLM-L6-v2'
        text_embedding_model = SentenceTransformer(embedding_model_name)
        TEXT_EMBEDDING_DIM = text_embedding_model.get_sentence_embedding_dimension()
        logging.info(f"ST model '{embedding_model_name}' loaded (Dim: {TEXT_EMBEDDING_DIM}).")
    except Exception as e:
        logging.error(f"ERROR loading Sentence Transformer model: {e}")
        logging.warning("Text embeddings disabled.")
        text_embedding_model = None
        TEXT_EMBEDDING_DIM = 0

# --- Modelo GNN ---
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, embedding_dim):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- Grafo y Nodos ---
class KnowledgeComponent:
    def __init__(self, node_id, content="Abstract Concept", initial_state=0.1, keywords=None):
        self.id = node_id
        self.content = content
        self.state = initial_state
        self.keywords = keywords if keywords else set()
        self.connections_out = {}
        self.connections_in = {}

    def update_state(self, new_state):
        MIN_STATE = 0.01
        self.state = max(MIN_STATE, min(1.0, new_state))

    def add_connection(self, target_node, utility):
        if target_node.id != self.id:
            self.connections_out[target_node.id] = utility
            target_node.connections_in[self.id] = utility

    def __repr__(self):
        kw_str = f", KW={list(self.keywords)}" if self.keywords else ""
        return f"Node({self.id}, S={self.state:.3f}{kw_str})"

class CollectiveSynthesisGraph:
    def __init__(self, config):
        self.nodes = {}
        self.next_node_id = 0
        self.config = config
        self.num_base_features = 4
        self.num_node_features = self.num_base_features + (TEXT_EMBEDDING_DIM if text_embedding_model else 0)
        config['gnn_input_dim'] = self.num_node_features
        hidden_channels = config.get('gnn_hidden_dim', 16)
        embedding_dim = config.get('gnn_embedding_dim', 8)
        self.gnn_model = GNNModel(self.num_node_features, hidden_channels, embedding_dim)
        self.node_embeddings = {}
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        logging.info(f"GNN Initialized: Input={self.num_node_features}, Hidden={hidden_channels}, Embedding={embedding_dim}")

    def _prepare_pyg_data(self):
        if not self.nodes:
            return None, None
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        num_nodes = len(self.nodes)
        num_node_features = self.num_node_features
        node_features = torch.zeros((num_nodes, num_node_features), dtype=torch.float)
        text_embeddings = None
        if text_embedding_model is not None and TEXT_EMBEDDING_DIM > 0:
            all_node_text = []
            for i in range(num_nodes):
                if i not in self.idx_to_node_id:
                    continue
                node = self.nodes[self.idx_to_node_id[i]]
                combined_text = node.content
                if node.keywords:
                    combined_text += " " + " ".join(sorted(node.keywords))
                all_node_text.append(combined_text)
            if not all_node_text:
                text_embeddings = torch.zeros((num_nodes, TEXT_EMBEDDING_DIM), dtype=torch.float)
            else:
                try:
                    with torch.no_grad():
                        text_embeddings_np = text_embedding_model.encode(all_node_text, convert_to_numpy=True, show_progress_bar=False)
                        text_embeddings = torch.tensor(text_embeddings_np, dtype=torch.float)
                except Exception as e:
                    logging.warning(f"GNN Prep Warning: Error generating text embeddings: {e}")
                    text_embeddings = torch.zeros((num_nodes, TEXT_EMBEDDING_DIM), dtype=torch.float)
        for i in range(num_nodes):
            if i not in self.idx_to_node_id:
                continue
            node_id = self.idx_to_node_id[i]
            node = self.nodes[node_id]
            node_features[i, 0] = node.state
            node_features[i, 1] = float(len(node.connections_in))
            node_features[i, 2] = float(len(node.connections_out))
            node_features[i, 3] = float(len(node.keywords))
            if text_embeddings is not None and i < text_embeddings.shape[0]:
                node_features[i, self.num_base_features:] = text_embeddings[i]
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
        return node_features, edge_index

    def update_embeddings(self):
        if not self.nodes:
            self.node_embeddings = {}
            return
        node_features, edge_index = self._prepare_pyg_data()
        if node_features is None or edge_index is None:
            return
        if edge_index.numel() == 0:
            logging.warning("GNN Warning: No edges found. Cannot compute GNN embeddings.")
            self.node_embeddings = {nid: torch.zeros(self.config.get('gnn_embedding_dim', 8)) for nid in self.nodes}
            return
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(1) if edge_index.shape[0] == 2 else edge_index.unsqueeze(0)
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t()
        if edge_index.shape[0] != 2:
            logging.error(f"GNN Error: edge_index has wrong shape {edge_index.shape}. Expected [2, num_edges].")
            return
        self.gnn_model.eval()
        with torch.no_grad():
            try:
                all_embeddings_tensor = self.gnn_model(node_features, edge_index)
                self.node_embeddings = {self.idx_to_node_id[i]: embedding
                                        for i, embedding in enumerate(all_embeddings_tensor)
                                        if i in self.idx_to_node_id}
            except IndexError as ie:
                logging.error(f"GNN IndexError during forward pass: {ie}. Indices: {self.idx_to_node_id.keys()}, EdgeIndexMax: {edge_index.max() if edge_index.numel() > 0 else 'N/A'}")
            except Exception as e:
                logging.error(f"GNN Error during forward pass: {e}")

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

    def get_random_node_biased(self):
        if not self.nodes:
            return None
        nodes = list(self.nodes.values())
        weights = [(node.state ** 2) + 0.01 for node in nodes]
        if sum(weights) <= 0.0:
            return random.choice(nodes)
        return random.choices(nodes, weights=weights, k=1)[0]

    def print_summary(self, log_level=logging.INFO):
        num_nodes = len(self.nodes)
        num_edges = sum(len(n.connections_out) for n in self.nodes.values())
        avg_state = (sum(n.state for n in self.nodes.values()) / num_nodes) if num_nodes > 0 else 0
        logging.log(log_level, f"Graph Summary - Nodes: {num_nodes}, Edges: {num_edges}, Average State: {avg_state:.3f}")

    def save_state(self, base_file_path):
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        logging.info(f"Attempting to save state to base path: {base_file_path}")
        G = nx.DiGraph()
        G.graph['next_node_id'] = self.next_node_id
        for node_id, node in self.nodes.items():
            keywords_str = ",".join(sorted(list(node.keywords)))
            G.add_node(str(node_id), state=node.state, content=node.content, keywords=keywords_str)
        for node_id, node in self.nodes.items():
            for target_id, utility in node.connections_out.items():
                if str(node_id) in G and str(target_id) in G:
                    G.add_edge(str(node_id), str(target_id), utility=utility)
        try:
            nx.write_graphml(G, graphml_path)
            logging.info(f"Graph structure saved to {graphml_path}")
        except Exception as e:
            logging.error(f"Error saving GraphML to {graphml_path}: {e}")
        try:
            torch.save(self.gnn_model.state_dict(), gnn_model_path)
            logging.info(f"GNN model state saved to {gnn_model_path}")
        except Exception as e:
            logging.error(f"Error saving GNN model state to {gnn_model_path}: {e}")
        try:
            detached_embeddings = {node_id: emb.detach().cpu() for node_id, emb in self.node_embeddings.items()}
            torch.save(detached_embeddings, embeddings_path)
            logging.info(f"Node embeddings saved to {embeddings_path}")
        except Exception as e:
            logging.error(f"Error saving node embeddings to {embeddings_path}: {e}")

    def load_state(self, base_file_path):
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        logging.info(f"Attempting to load state from base path: {base_file_path}")
        try:
            if not os.path.exists(graphml_path):
                logging.error(f"Error: GraphML file not found at {graphml_path}. Cannot load state.")
                return False
            G = nx.read_graphml(graphml_path)
            self.nodes = {}
            self.next_node_id = 0
            max_id = -1
            logging.info("Loading nodes...")
            for node_id_str, data in G.nodes(data=True):
                try:
                    node_id = int(node_id_str)
                except ValueError:
                    logging.warning(f"Skipping node with non-integer ID '{node_id_str}'.")
                    continue
                keywords_str = data.get('keywords', '')
                keywords = set(k for k in keywords_str.split(',') if k)
                new_node = KnowledgeComponent(node_id,
                                              content=data.get('content', ""),
                                              initial_state=float(data.get('state', 0.1)),
                                              keywords=keywords)
                self.nodes[node_id] = new_node
                max_id = max(max_id, node_id)
            self.next_node_id = int(G.graph.get('next_node_id', max_id + 1))
            logging.info(f"Loaded {len(self.nodes)} nodes. Next ID: {self.next_node_id}")
            logging.info("Loading edges...")
            edge_count = 0
            for source_id_str, target_id_str, data in G.edges(data=True):
                try:
                    source_id = int(source_id_str)
                    target_id = int(target_id_str)
                    if source_id in self.nodes and target_id in self.nodes:
                        self.add_edge(source_id, target_id, float(data.get('utility', 0.0)))
                        edge_count += 1
                except ValueError:
                    logging.warning(f"Skipping edge with non-integer IDs ('{source_id_str}' -> '{target_id_str}').")
                    continue
            logging.info(f"Loaded {edge_count} edges.")
            logging.info(f"Graph structure loaded from {graphml_path}")
        except Exception as e:
            logging.critical(f"CRITICAL Error loading GraphML: {e}")
            return False

        try:
            if not os.path.exists(gnn_model_path):
                logging.warning(f"GNN model file not found at {gnn_model_path}.")
            else:
                logging.info(f"Loading GNN model state from {gnn_model_path}...")
                state_dict = torch.load(gnn_model_path)
                self.gnn_model.load_state_dict(state_dict)
                self.gnn_model.eval()
                logging.info("GNN model state loaded.")
        except Exception as e:
            logging.error(f"Error loading GNN model state: {e}. Using initialized model.")
        try:
            if not os.path.exists(embeddings_path):
                logging.warning(f"Embeddings file not found at {embeddings_path}.")
                self.node_embeddings = {}
            else:
                logging.info(f"Loading node embeddings from {embeddings_path}...")
                self.node_embeddings = torch.load(embeddings_path)
                logging.info(f"Node embeddings loaded for {len(self.node_embeddings)} nodes.")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}.")
            self.node_embeddings = {}
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        logging.info("Node ID mappings recalculated.")
        return True

    def visualize_graph(self, config):
        if not self.nodes:
            logging.warning("Cannot visualize empty graph.")
            return
        logging.info("Generating graph visualization...")
        G = nx.DiGraph()
        node_labels = {}
        node_sizes = []
        node_colors = []
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            G.add_node(node_id_str)
            node_labels[node_id_str] = f"{node_id}\nS={node.state:.2f}"
            node_sizes.append(100 + node.state * 1500)
            node_colors.append(node.state)
        edge_list = []
        edge_weights = []
        edge_colors = []
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            for target_id, utility in node.connections_out.items():
                target_id_str = str(target_id)
                if node_id_str in G and target_id_str in G:
                    edge_list.append((node_id_str, target_id_str))
                    edge_weights.append(1 + abs(utility) * 4)
                    edge_colors.append(utility)
        if not G.nodes:
            logging.warning("No nodes to visualize.")
            return
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_title("MSC Graph Visualization")
        ax.axis('off')
        try:
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        except Exception as e:
            logging.error(f"Error calculating layout: {e}. Using random layout.")
            pos = nx.random_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap=plt.cm.viridis,
                               node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list, edge_color=edge_colors,
                               edge_cmap=plt.cm.coolwarm, width=edge_weights, alpha=0.6,
                               arrows=True, arrowstyle='->', arrowsize=15)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=8)
        if node_colors:
            norm_nodes = plt.Normalize(vmin=min(node_colors or [0]), vmax=max(node_colors or [1]))
            sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm_nodes)
            sm_nodes.set_array([])
            cbar_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.5)
            cbar_nodes.set_label('Node State (sj)')
        if edge_colors:
            norm_edges = plt.Normalize(vmin=min(edge_colors or [-1]), vmax=max(edge_colors or [1]))
            sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm_edges)
            sm_edges.set_array([])
            cbar_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.5)
            cbar_edges.set_label('Edge Utility (uij)')
        plt.show()

# --- Definiciones de Sintetizadores ---
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
            logging.info(f"Proposer {self.id}: Proposed initial {new_node!r}")
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
        logging.info(f"Proposer {self.id}: Proposed {new_node!r} linked from {source_node!r} with U={utility:.2f}")

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
        influence_sum = 0.0
        weight_sum = 0.0
        accumulated_similarity_boost = 0.0
        penalty_factor = 1.0
        target_embedding = self.graph.get_embedding(target_node.id)
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
                    if source_node.state > 0.5 and utility_uji > 0.1 and target_embedding is not None and source_embedding is not None:
                        similarity = self.calculate_cosine_similarity(target_embedding, source_embedding)
                        boost_from_source = similarity * source_node.state * weight * self.similarity_boost_factor
                        accumulated_similarity_boost += boost_from_source
                    if source_node.state > 0.7 and utility_uji < 0:
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
        logging.info(f"Evaluator {self.id}: Evaluated {target_node!r}. State: {current_state:.3f} -> {target_node.state:.3f} (Target: {influence_target:.3f})")

class CombinerAgent(Synthesizer):
    def __init__(self, agent_id, graph, config):
        super().__init__(agent_id, graph, config)
        self.similarity_threshold = config.get('combiner_similarity_threshold', 0.7)
        self.compatibility_threshold = config.get('combiner_compatibility_threshold', 0.6)

    def calculate_cosine_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def act(self):
        if len(self.graph.nodes) < 2:
            return
        node_a = self.graph.get_random_node_biased()
        node_b = self.graph.get_random_node_biased()
        if node_a is None or node_b is None or node_a.id == node_b.id or \
           node_b.id in node_a.connections_out or node_a.id in node_b.connections_out:
            return
        emb_a = self.graph.get_embedding(node_a.id)
        emb_b = self.graph.get_embedding(node_b.id)
        edge_added = False
        if emb_a is not None and emb_b is not None:
            cosine_sim = self.calculate_cosine_similarity(emb_a, emb_b)
            normalized_sim = (cosine_sim + 1) / 2
            if normalized_sim >= self.similarity_threshold:
                utility = max(-1.0, min(1.0, cosine_sim))
                if self.graph.add_edge(node_a.id, node_b.id, utility):
                    logging.info(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} based on embedding similarity (Score: {normalized_sim:.2f}, U={utility:.2f})")
                    edge_added = True
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
                    logging.info(f"Combiner {self.id}: Combined {node_a!r} -> {node_b!r} using FALLBACK logic (Score: {compatibility_score:.2f}, U={utility:.2f})")

# --- Clase SimulationRunner ---
class SimulationRunner:
    """Encapsula y ejecuta la simulación en un hilo separado."""
    def __init__(self, config):
        self.config = config
        self.graph = CollectiveSynthesisGraph(config)
        self.agents = []
        self.is_running = False
        self.simulation_thread = None
        self.step_count = 0
        self.lock = threading.Lock()

        load_path = self.config.get('load_state', None)
        if load_path:
            logging.info(f"--- Loading initial state from: {load_path} ---")
            if not self.graph.load_state(load_path):
                logging.error("Failed to load state. Starting fresh simulation.")
            else:
                logging.info("GNN: Initializing embeddings after load...")
                self.graph.update_embeddings()
                if self.graph.node_embeddings:
                    logging.info(f"GNN: Initial embeddings available for {len(self.graph.node_embeddings)} nodes.")

        num_proposers = config.get('num_proposers', 3)
        num_evaluators = config.get('num_evaluators', 6)
        num_combiners = config.get('num_combiners', 2)
        for i in range(num_proposers):
            self.agents.append(ProposerAgent(f"P{i}", self.graph, config))
        for i in range(num_evaluators):
            self.agents.append(EvaluatorAgent(f"E{i}", self.graph, config))
        for i in range(num_combiners):
            self.agents.append(CombinerAgent(f"C{i}", self.graph, config))

    def _simulation_loop(self):
        step_delay = self.config.get('step_delay', 0.1)
        gnn_update_frequency = self.config.get('gnn_update_frequency', 10)
        summary_frequency = self.config.get('summary_frequency', 50)
        max_steps = self.config.get('simulation_steps', None)
        is_api_mode = self.config.get('run_api', False)
        run_continuously = is_api_mode or (max_steps is None)

        while self.is_running:
            self.step_count += 1
            log_level = logging.DEBUG if self.step_count % summary_frequency != 0 else logging.INFO
            logging.log(log_level, f"--- Step {self.step_count} ---")

            if not run_continuously and max_steps is not None and self.step_count > max_steps:
                logging.info(f"Reached max steps ({max_steps}). Stopping simulation loop.")
                self.is_running = False
                break

            if not self.agents:
                logging.warning("No agents to run.")
                self.is_running = False
                break

            with self.lock:
                if self.step_count % gnn_update_frequency == 0:
                    self.graph.update_embeddings()
                agent = random.choice(self.agents)
                agent.act()
                if self.step_count % summary_frequency == 0:
                    self.graph.print_summary(logging.INFO)
            time.sleep(step_delay)
        logging.info("--- Simulation loop finished ---")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
            logging.info("--- Simulation thread started ---")
        else:
            logging.warning("Simulation thread already running.")

    def stop(self):
        if self.is_running:
            logging.info("--- Stopping simulation thread ---")
            self.is_running = False
        if self.simulation_thread is not None and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5.0)
            if self.simulation_thread.is_alive():
                logging.warning("Simulation thread did not stop gracefully.")
        logging.info("--- Performing final actions ---")
        with self.lock:
            logging.info("GNN: Final embedding update...")
            self.graph.update_embeddings()
            if self.graph.node_embeddings:
                logging.info(f"GNN: Final embeddings calculated for {len(self.graph.node_embeddings)} nodes.")
            save_path = self.config.get('save_state', None)
            if save_path:
                logging.info(f"--- Saving final state to: {save_path} ---")
                self.graph.save_state(save_path)
            logging.info("--- Final Graph State ---")
            self.graph.print_summary(logging.INFO)
            is_api_mode = self.config.get('run_api', False)
            should_visualize = (not is_api_mode) and self.config.get('visualize_graph', False)
            if should_visualize:
                if self.graph.nodes:
                    logging.info("Attempting to display graph visualization...")
                    try:
                        self.graph.visualize_graph(self.config)
                    except Exception as e:
                        logging.error(f"Error during final visualization: {e}")
                else:
                    logging.warning("Graph is empty, skipping visualization.")
        logging.info("--- Simulation Runner Stopped ---")

    def get_status(self):
        with self.lock:
            num_nodes = len(self.graph.nodes)
            num_edges = sum(len(n.connections_out) for n in self.graph.nodes.values())
            avg_state = (sum(n.state for n in self.graph.nodes.values()) / num_nodes) if num_nodes > 0 else 0
            status = {
                "is_running": self.is_running,
                "current_step": self.step_count,
                "node_count": num_nodes,
                "edge_count": num_edges,
                "average_state": round(avg_state, 3),
                "embeddings_count": len(self.graph.node_embeddings)
            }
        return status

# --- Carga de Configuración ---
def load_config(args):
    config = {
        'simulation_steps': None,
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
        'gnn_hidden_dim': 16,
        'gnn_embedding_dim': 8,
        'gnn_update_frequency': 10,
        'save_state': None,
        'load_state': None,
        'summary_frequency': 50,
        'run_api': False
    }
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
            logging.info(f"Loaded config from: {args.config}")
        except FileNotFoundError:
            logging.warning(f"Warning: Config file not found at {args.config}.")
        except Exception as e:
            logging.error(f"Error loading config file {args.config}: {e}.")
    cli_args = vars(args).copy()
    cli_args.pop('config', None)
    cli_args.pop('gnn_input_dim', None)
    if 'visualize_graph' in cli_args:
        config['visualize_graph'] = cli_args.pop('visualize_graph')
    if 'run_api' in cli_args:
        config['run_api'] = cli_args.pop('run_api')
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value
    config['gnn_input_dim'] = 4 + (TEXT_EMBEDDING_DIM if text_embedding_model else 0)
    return config

# --- API Flask y Punto de Entrada ---
simulation_runner = None
app = Flask(__name__)

@app.route('/status')
def get_simulation_status():
    if simulation_runner:
        return jsonify(simulation_runner.get_status())
    else:
        return jsonify({"error": "Simulation not initialized"}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MSC Simulation with GNN and optional API.")
    parser.add_argument('--config', type=str, help='Path to YAML config file.')
    parser.add_argument('--simulation_steps', type=int, default=None,
                        help='Run for a fixed number of steps (default: continuous if --run_api is off).')
    parser.add_argument('--num_proposers', type=int)
    parser.add_argument('--num_evaluators', type=int)
    parser.add_argument('--num_combiners', type=int)
    parser.add_argument('--step_delay', type=float)
    parser.add_argument('--evaluator_learning_rate', type=float)
    parser.add_argument('--evaluator_similarity_boost', type=float)
    parser.add_argument('--evaluator_decay_rate', type=float)
    parser.add_argument('--combiner_compatibility_threshold', type=float)
    parser.add_argument('--combiner_similarity_threshold', type=float)
    parser.add_argument('--gnn_hidden_dim', type=int)
    parser.add_argument('--gnn_embedding_dim', type=int)
    parser.add_argument('--gnn_update_frequency', type=int)
    parser.add_argument('--visualize_graph', action='store_true', help='Show graph plot at the end (only in fixed-step mode without --run_api).')
    parser.add_argument('--save_state', type=str, help='Base path to save simulation state on stop.')
    parser.add_argument('--load_state', type=str, help='Base path to load simulation state at start.')
    parser.add_argument('--run_api', action='store_true', help='Run Flask API server (runs continuously).')
    parser.add_argument('--summary_frequency', type=int, help='Frequency (in steps) to log graph summary.')

    args = parser.parse_args()
    final_config = load_config(args)
    final_config['run_api'] = args.run_api

    simulation_runner = SimulationRunner(final_config)
    simulation_runner.start()

    if final_config.get('run_api', False):
        logging.info("Starting Flask API server on http://127.0.0.1:5000")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        try:
            app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
        except KeyboardInterrupt:
            logging.info("Ctrl+C detected. Stopping Flask API and simulation...")
        finally:
            simulation_runner.stop()
    else:
        logging.info("Simulation running in background thread.")
        logging.info("Press Ctrl+C to stop if running indefinitely.")
        try:
            while simulation_runner.is_running and simulation_runner.simulation_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt detected. Stopping simulation...")
            simulation_runner.stop()


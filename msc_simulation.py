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
from torch_geometric.utils import negative_sampling
import os
from dotenv import load_dotenv
import statistics         # Métricas de estado
import csv                # Para escribir archivos CSV
import requests
from abc import ABC, abstractmethod

load_dotenv()  # Carga variables desde .env al entorno

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: 'sentence-transformers' not found. Text embeddings disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
    wikipedia.set_lang("es")  # O usa "en" si prefieres inglés
except ImportError:
    print("WARNING: 'wikipedia' library not found. KnowledgeFetcherAgent disabled.")
    WIKIPEDIA_AVAILABLE = False
    wikipedia = None

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
else:
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

    def decode(self, z, edge_label_index):
        emb_src = z[edge_label_index[0]]
        emb_dst = z[edge_label_index[1]]
        return (emb_src * emb_dst).sum(dim=-1)

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
        self.num_base_features = 5  # Se añade una feature extra para PageRank
        self.num_node_features = self.num_base_features + (TEXT_EMBEDDING_DIM if text_embedding_model else 0)
        config['gnn_input_dim'] = self.num_node_features
        hidden_channels = config.get('gnn_hidden_dim', 16)
        embedding_dim = config.get('gnn_embedding_dim', 8)
        self.gnn_model = GNNModel(self.num_node_features, hidden_channels, embedding_dim)
        self.node_embeddings = {}
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        logging.info(f"GNN Initialized: Input={self.num_node_features}, Hidden={hidden_channels}, Embedding={embedding_dim}")

        self.gnn_optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=config.get('gnn_learning_rate', 0.01))
        self.gnn_loss_fn = torch.nn.BCEWithLogitsLoss()

    def _prepare_pyg_data(self):
        if not self.nodes:
            return None, None
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        num_nodes = len(self.nodes)
        node_features = torch.zeros((num_nodes, self.num_node_features), dtype=torch.float)

        temp_G = nx.DiGraph()
        for node_id in self.nodes.keys():
            temp_G.add_node(node_id)
        for node in self.nodes.values():
            for target_id in node.connections_out.keys():
                if target_id in self.nodes:
                    temp_G.add_edge(node.id, target_id)
        pagerank = nx.pagerank(temp_G) if temp_G.number_of_nodes() > 0 else {nid: 0.0 for nid in self.nodes.keys()}

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
                    logging.warning(f"GNN Prep Warning: {e}")
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
            node_features[i, 4] = pagerank.get(node_id, 0.0)
            if text_embeddings is not None and i < text_embeddings.shape[0]:
                node_features[i, self.num_base_features:] = text_embeddings[i]
        source_indices = []
        target_indices = []
        for source_node_id, node in self.nodes.items():
            if source_node_id not in self.node_id_to_idx:
                continue
            source_idx = self.node_id_to_idx[source_node_id]
            for target_id in node.connections_out.keys():
                if target_id in self.node_id_to_idx:
                    target_idx = self.node_id_to_idx[target_id]
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
            logging.warning("GNN Warning: No edges found.")
            self.node_embeddings = {nid: torch.zeros(self.config.get('gnn_embedding_dim', 8)) for nid in self.nodes}
            return
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(1) if edge_index.shape[0] == 2 else edge_index.unsqueeze(0)
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t()
        if edge_index.shape[0] != 2:
            logging.error(f"GNN Error: edge_index shape {edge_index.shape} is incorrect.")
            return
        self.gnn_model.eval()
        with torch.no_grad():
            try:
                all_embeddings_tensor = self.gnn_model(node_features, edge_index)
                self.node_embeddings = {self.idx_to_node_id[i]: embedding
                                        for i, embedding in enumerate(all_embeddings_tensor)
                                        if i in self.idx_to_node_id}
            except Exception as e:
                logging.error(f"GNN forward pass error: {e}")

    def train_gnn(self, num_epochs=10):
        if not self.nodes or len(self.nodes) < 2:
            logging.warning("GNN Training: Not enough nodes to train.")
            return
        node_features, edge_index = self._prepare_pyg_data()
        if node_features is None or edge_index is None or edge_index.numel() == 0:
            logging.warning("GNN Training: Insufficient data for training.")
            return
        num_nodes = node_features.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(device)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        self.gnn_model.train()
        total_loss = 0.0
        try:
            for epoch in range(num_epochs):
                self.gnn_optimizer.zero_grad()
                z = self.gnn_model(node_features, edge_index)
                pos_edge_label_index = edge_index
                neg_edge_label_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=pos_edge_label_index.size(1),
                    method='sparse'
                )
                if neg_edge_label_index.numel() == 0:
                    logging.warning(f"Epoch {epoch+1}: No negative samples generated.")
                    continue
                edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=-1)
                pos_labels = torch.ones(pos_edge_label_index.size(1), device=device)
                neg_labels = torch.zeros(neg_edge_label_index.size(1), device=device)
                edge_labels = torch.cat([pos_labels, neg_labels], dim=0)
                out_scores = self.gnn_model.decode(z, edge_label_index)
                if out_scores.size(0) != edge_labels.size(0):
                    logging.error(f"Epoch {epoch+1}: Output scores size {out_scores.size(0)} does not match labels size {edge_labels.size(0)}.")
                    continue
                loss = self.gnn_loss_fn(out_scores, edge_labels)
                loss.backward()
                self.gnn_optimizer.step()
                total_loss += loss.item()
                logging.debug(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}")
        except Exception as e:
            logging.error(f"GNN Training error: {e}")
        finally:
            self.gnn_model.eval()
            avg_loss = total_loss / num_epochs if num_epochs > 0 else float('inf')
            logging.info(f"GNN Training finished: {num_epochs} epochs, Avg Loss = {avg_loss:.4f}")

    def add_node(self, content="Abstract Concept", initial_state=0.1, keywords=None):
        node_id = self.next_node_id
        new_node = KnowledgeComponent(node_id, content, initial_state, set(keywords) if keywords else set())
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
        logging.log(log_level, f"Graph Summary - Nodes: {num_nodes}, Edges: {num_edges}, Avg State: {avg_state:.3f}")

    def log_global_metrics(self, log_level=logging.INFO, current_step="N/A"):
        logging.log(log_level, f"--- Global Metrics (Step: {current_step}) ---")
        num_nodes = len(self.nodes)
        if num_nodes < 2:
            logging.log(log_level, "--- Global Metrics: Skipped (Graph too small) ---")
            return
        G = nx.DiGraph()
        node_states = []
        for node_id, node in self.nodes.items():
            G.add_node(str(node_id))
            node_states.append(node.state)
        num_edges = 0
        for node_id, node in self.nodes.items():
            for target_id, utility in node.connections_out.items():
                if str(node_id) in G and str(target_id) in G:
                    G.add_edge(str(node_id), str(target_id))
                    num_edges += 1
        try:
            density = nx.density(G)
            logging.log(log_level, f"  Density: {density:.4f}")
        except Exception as e:
            logging.warning(f"  Density calculation error: {e}")
        try:
            avg_clustering = nx.average_clustering(G.to_undirected())
            logging.log(log_level, f"  Avg Clustering Coefficient: {avg_clustering:.4f}")
        except Exception as e:
            logging.warning(f"  Clustering calculation error: {e}")
        try:
            num_components = nx.number_weakly_connected_components(G)
            logging.log(log_level, f"  Weakly Connected Components: {num_components}")
        except Exception as e:
            logging.warning(f"  Component calculation error: {e}")
        if node_states:
            mean_state = statistics.mean(node_states)
            median_state = statistics.median(node_states)
            stdev_state = statistics.stdev(node_states) if len(node_states) > 1 else 0.0
            min_state = min(node_states)
            max_state = max(node_states)
            logging.log(log_level, f"  Node State Stats: Mean={mean_state:.4f}, Median={median_state:.4f}, StdDev={stdev_state:.4f}, Min={min_state:.4f}, Max={max_state:.4f}")
        else:
            logging.log(log_level, "  Node State Stats: N/A")
    def get_global_metrics(self):
        metrics = {field: None for field in ["Nodes", "Edges", "Density", "AvgClustering",
                                               "Components", "MeanState", "MedianState", "StdDevState", "MinState", "MaxState"]}
        num_nodes = len(self.nodes)
        metrics["Nodes"] = num_nodes
        if num_nodes == 0:
            return metrics
        node_states = [n.state for n in self.nodes.values()]
        if node_states:
            metrics["MeanState"] = statistics.mean(node_states)
            metrics["MedianState"] = statistics.median(node_states)
            metrics["MinState"] = min(node_states)
            metrics["MaxState"] = max(node_states)
            metrics["StdDevState"] = statistics.stdev(node_states) if num_nodes > 1 else 0.0
        G = nx.DiGraph()
        for node_id in self.nodes.keys():
            G.add_node(str(node_id))
        num_edges = 0
        for node_id, node in self.nodes.items():
            for target_id in node.connections_out.keys():
                if str(node_id) in G and str(target_id) in G:
                    G.add_edge(str(node_id), str(target_id))
                    num_edges += 1
        metrics["Edges"] = num_edges
        if num_nodes >= 2:
            try:
                metrics["Density"] = nx.density(G)
            except:
                pass
        if num_nodes >= 1:
            try:
                metrics["Components"] = nx.number_weakly_connected_components(G)
            except:
                pass
            try:
                metrics["AvgClustering"] = nx.average_clustering(G) if num_edges > 0 and num_nodes > 1 else 0.0
            except Exception as e:
                logging.warning(f"Avg Clustering error: {e}")
                metrics["AvgClustering"] = None
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics[key] = round(value, 4)
        return metrics
    def save_state(self, base_file_path):
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        logging.info(f"Saving state to: {base_file_path}")
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
            logging.info(f"Graph saved to {graphml_path}")
        except Exception as e:
            logging.error(f"GraphML save error: {e}")
        try:
            torch.save(self.gnn_model.state_dict(), gnn_model_path)
            logging.info(f"GNN model saved to {gnn_model_path}")
        except Exception as e:
            logging.error(f"GNN model save error: {e}")
        try:
            detached_embeddings = {node_id: emb.detach().cpu() for node_id, emb in self.node_embeddings.items()}
            torch.save(detached_embeddings, embeddings_path)
            logging.info(f"Embeddings saved to {embeddings_path}")
        except Exception as e:
            logging.error(f"Embeddings save error: {e}")
    def load_state(self, base_file_path):
        graphml_path = base_file_path + ".graphml"
        gnn_model_path = base_file_path + ".gnn.pth"
        embeddings_path = base_file_path + ".embeddings.pth"
        logging.info(f"Loading state from: {base_file_path}")
        try:
            if not os.path.exists(graphml_path):
                logging.error(f"GraphML file not found at {graphml_path}.")
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
                    logging.warning(f"Skipping non-integer node ID: {node_id_str}")
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
            logging.info(f"Loaded {len(self.nodes)} nodes. Next node ID: {self.next_node_id}")
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
                    logging.warning(f"Skipping edge with non-integer IDs: {source_id_str} -> {target_id_str}")
                    continue
            logging.info(f"Loaded {edge_count} edges.")
        except Exception as e:
            logging.critical(f"Critical error loading state: {e}")
            return False
        try:
            if not os.path.exists(gnn_model_path):
                logging.warning(f"GNN model file not found at {gnn_model_path}.")
            else:
                logging.info(f"Loading GNN model from {gnn_model_path}...")
                state_dict = torch.load(gnn_model_path)
                self.gnn_model.load_state_dict(state_dict)
                self.gnn_model.eval()
                logging.info("GNN model loaded.")
        except Exception as e:
            logging.error(f"GNN model load error: {e}")
        try:
            if not os.path.exists(embeddings_path):
                logging.warning(f"Embeddings file not found at {embeddings_path}.")
                self.node_embeddings = {}
            else:
                logging.info(f"Loading embeddings from {embeddings_path}...")
                self.node_embeddings = torch.load(embeddings_path)
                logging.info(f"Loaded embeddings for {len(self.node_embeddings)} nodes.")
        except Exception as e:
            logging.error(f"Embeddings load error: {e}")
            self.node_embeddings = {}
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        logging.info("Node ID mappings recalculated.")
        return True

    def get_graph_elements_for_cytoscape(self):
        elements = []
        with self.lock:
            try:
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=0, vmax=1)
                for node_id, node in self.nodes.items():
                    try:
                        node_color = cmap(norm(node.state))
                        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
                    except Exception as ex:
                        app.logger.error("Error processing node %s: %s", node_id, ex)
                        hex_color = "#cccccc"
                    elements.append({
                        'data': {
                            'id': str(node_id),
                            'label': f'Node {node_id}\nS={node.state:.2f}',
                            'state': node.state,
                            'keywords': ", ".join(sorted(list(node.keywords))) if node.keywords else ""
                        },
                        'style': {
                            'background-color': hex_color,
                            'width': f"{20 + node.state * 40}px",
                            'height': f"{20 + node.state * 40}px"
                        }
                    })
                for source_id, node in self.graph.nodes.items():
                    for target_id, utility in node.connections_out.items():
                        if source_id in self.graph.nodes and target_id in self.graph.nodes:
                            elements.append({
                                'data': {
                                    'source': str(source_id),
                                    'target': str(target_id),
                                    'utility': utility
                                },
                                'style': {
                                    'width': 1 + abs(utility) * 2,
                                    'line-color': 'red' if utility < 0 else ('blue' if utility > 0 else 'grey')
                                }
                            })
            except Exception as e:
                app.logger.error("Unexpected error in get_graph_elements_for_cytoscape: %s", e)
                raise e
        return elements

def generate_code(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("FATAL ERROR: GEMINI_API_KEY not found.")
        return None
    try:
        try:
            import genai
        except ImportError:
            logging.error("Module 'genai' is not installed.")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        else:
            logging.warning(f"Gemini API returned no text. Response: {response}")
            return ""
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return ""

# --- Clase Base para Agentes ---
class Synthesizer:
    def __init__(self, agent_id, graph, config):
        self.id = agent_id
        self.graph = graph
        self.config = config
        self.omega = config.get('initial_omega', 100.0)
        self.reputation = config.get('initial_reputation', 1.0)
    def act(self):
        raise NotImplementedError

# --- Clase Base para Agentes Institucionales ---
class InstitutionAgent(Synthesizer, ABC):
    def act(self):
        self.institution_action()
    @abstractmethod
    def institution_action(self):
        pass
    def log_institution(self, message):
        logging.info(f"[{self.__class__.__name__} {self.id}] {message}")

# --- JUZGADO MSC ---
class InspectorAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Checking system integrity...")
        anomalies = [node for node in self.graph.nodes.values() if node.state < 0.05]
        if anomalies:
            self.log_institution(f"Detected {len(anomalies)} anomalies.")
        else:
            self.log_institution("All systems normal.")

class PoliceAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Monitoring suspicious activity...")
        offenders = [node for node in self.graph.nodes.values() if any(u < -0.8 for u in node.connections_out.values())]
        if offenders:
            for offender in offenders:
                self.log_institution(f"Sanctioning node {offender.id}.")
                offender.update_state(offender.state * 0.9)
        else:
            self.log_institution("No abuse detected.")

class CoordinatorAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Resolving conflicts among nodes...")
        conflicts = []
        for node in self.graph.nodes.values():
            for target_id, utility in node.connections_out.items():
                target = self.graph.get_node(target_id)
                if target and abs(node.state - target.state) > 0.5:
                    conflicts.append((node, target))
        if conflicts:
            self.log_institution(f"{len(conflicts)} conflicts detected; mediating...")
            for node, target in conflicts:
                avg_state = (node.state + target.state) / 2
                node.update_state(avg_state)
                target.update_state(avg_state)
        else:
            self.log_institution("No conflicts found.")

class RepairAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Repairing nodes with low reputation despite high state...")
        for node in self.graph.nodes.values():
            if node.state > 0.7 and hasattr(node, 'reputation') and node.reputation < 0.5:
                self.log_institution(f"Rehabilitating node {node.id}.")
                node.update_state(min(1.0, node.state + 0.1))

# --- UNIVERSIDAD MSC ---
class MasterAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Organizing learning paths for nodes...")
        weak_nodes = [node for node in self.graph.nodes.values() if node.state < 0.3]
        advanced_nodes = [node for node in self.graph.nodes.values() if node.state > 0.8]
        if weak_nodes and advanced_nodes:
            for node in weak_nodes:
                mentor = random.choice(advanced_nodes)
                if self.graph.add_edge(mentor.id, node.id, 0.8):
                    self.log_institution(f"Connected node {node.id} with mentor {mentor.id}.")

class StudentAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Seeking learning opportunities...")
        candidate = self.graph.get_random_node_biased()
        if candidate:
            increment = 0.05
            candidate.update_state(candidate.state + increment)
            self.log_institution(f"Increased state of node {candidate.id} by {increment:.2f}.")

class ScientistAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Generating and publishing new hypotheses...")
        hypothesis = f"Hypothesis generated by {self.id}."
        new_node = self.graph.add_node(content=hypothesis, initial_state=0.6, keywords={"theory", "hypothesis"})
        self.graph.add_edge(random.choice(list(self.graph.nodes.keys())), new_node.id, 0.5)
        self.log_institution(f"Created theoretical node {new_node.id}.")

class StorageAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Archiving key nodes...")
        archived = sum(1 for node in self.graph.nodes.values() if "archive" in node.keywords or node.state > 0.9)
        self.log_institution(f"Archived {archived} nodes.")

# --- INSTITUTO FINANCIERO MSC ---
class BankAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Reviewing internal economic balance...")
        for agent in self.graph.agents if hasattr(self.graph, 'agents') else []:
            if agent.omega < 50:
                bonus = 10
                agent.omega += bonus
                self.log_institution(f"Granted bonus of {bonus} Ω to agent {agent.id}.")

class MerchantAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Facilitating resource exchanges...")
        resourceful = [agent for agent in self.graph.agents if agent.omega > 150] if hasattr(self.graph, 'agents') else []
        if resourceful:
            donor = random.choice(resourceful)
            recipient = random.choice([agent for agent in self.graph.agents if agent.omega < 80])
            transfer = 20
            donor.omega -= transfer
            recipient.omega += transfer
            self.log_institution(f"Transferred {transfer} Ω from {donor.id} to {recipient.id}.")

class MinerAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Exploring for unclaimed resources...")
        target = self.graph.get_random_node_biased()
        if target:
            bonus = 15
            target.update_state(target.state + 0.05)
            self.log_institution(f"Added {bonus} Ω bonus and increased state of node {target.id}.")

# --- INSTITUTO DE DENSIDAD Y POBLACIÓN ---

class PopulationRegulatorAgent(InstitutionAgent):
    """
    Reduce nodos redundantes y controla la explosión poblacional en el grafo.
    """
    def institution_action(self):
        self.log_institution("Regulating population: analyzing redundancy and density...")
        # Ejemplo: identificar nodos con estados muy similares en grupos muy densos
        redundant_nodes = []
        for node in self.graph.nodes.values():
            # Criterio: si un nodo tiene muchas conexiones y muy poca variación con sus vecinos
            if len(node.connections_out) > 5:
                neighbors = [self.graph.get_node(nid) for nid in node.connections_out.keys()]
                if all(abs(node.state - n.state) < 0.05 for n in neighbors if n):
                    redundant_nodes.append(node)
        if redundant_nodes:
            self.log_institution(f"Identified {len(redundant_nodes)} redundant nodes for regulation.")
            # Ejemplo: reducir estado o marcar nodos para eventual eliminación
            for rn in redundant_nodes:
                rn.update_state(rn.state * 0.95)
        else:
            self.log_institution("No redundant nodes detected.")

class SeederAgent(InstitutionAgent):
    """
    Siembra nuevos nodos en regiones del grafo con baja densidad o actividad.
    """
    def institution_action(self):
        self.log_institution("Seeding new nodes in sparse areas...")
        # Ejemplo: agregar un nodo nuevo si se detecta baja conectividad
        if len(self.graph.nodes) < 10:
            new_node = self.graph.add_node(content="Seeded Node", initial_state=0.2)
            self.log_institution(f"Seeded new node {new_node.id} due to low overall density.")
        else:
            # Buscar regiones con nodos poco conectados
            sparse_nodes = [n for n in self.graph.nodes.values() if len(n.connections_in) + len(n.connections_out) < 2]
            if sparse_nodes:
                new_node = self.graph.add_node(content="Additional Seed", initial_state=0.3)
                target = random.choice(sparse_nodes)
                self.graph.add_edge(target.id, new_node.id, utility=0.5)
                self.log_institution(f"Seeded new node {new_node.id} attached to sparse node {target.id}.")
            else:
                self.log_institution("No significantly sparse regions detected.")

class ClusterBalancerAgent(InstitutionAgent):
    """
    Redistribuye conexiones entre grupos de nodos para evitar desequilibrios topológicos.
    """
    def institution_action(self):
        self.log_institution("Balancing clusters in the graph...")
        # Ejemplo: ajustar conexiones de nodos aislados o sobreconectados para elevar la cohesión
        for node in self.graph.nodes.values():
            if len(node.connections_out) > 8:
                # Reducir conexiones excesivas: por ejemplo, eliminar la de menor utilidad
                min_conn = min(node.connections_out.items(), key=lambda item: item[1])
                node.connections_out.pop(min_conn[0])
                self.log_institution(f"Removed low-utility connection from node {node.id}.")
            elif len(node.connections_out) < 1:
                # Fomentar nuevas conexiones conectándolo con un nodo aleatorio
                candidate = self.graph.get_random_node_biased()
                if candidate and candidate.id != node.id:
                    self.graph.add_edge(node.id, candidate.id, utility=0.7)
                    self.log_institution(f"Added connection from node {node.id} to {candidate.id} for better balance.")

class MediatorAgent(InstitutionAgent):
    """
    Agente de mediación dinámica para resolver conflictos o coordinar acciones entre clusters.
    """
    def institution_action(self):
        self.log_institution("Mediating between clusters...")
        # Ejemplo: buscar pares de nodos con estados muy discrepantes conectados y ajustar su estado
        conflicts = []
        for node in self.graph.nodes.values():
            for target_id, utility in node.connections_out.items():
                target = self.graph.get_node(target_id)
                if target and abs(node.state - target.state) > 0.3:
                    conflicts.append((node, target))
        if conflicts:
            self.log_institution(f"Mediating {len(conflicts)} conflict pairs.")
            for n1, n2 in conflicts:
                average_state = (n1.state + n2.state) / 2
                n1.update_state(average_state)
                n2.update_state(average_state)
        else:
            self.log_institution("No conflicts require mediation.")

class MigrationAgent(InstitutionAgent):
    """
    Evalúa los clústeres del grafo y migra nodos desde áreas saturadas hacia regiones menos pobladas
    para fomentar la colaboración y reequilibrar la conectividad.
    """
    def institution_action(self):
        self.log_institution("Evaluating clusters for adaptive migration...")
        # Identificar nodos saturados y nodos en zonas poco conectadas (escasas conexiones)
        saturated_nodes = [node for node in self.graph.nodes.values() if len(node.connections_out) > 8]
        sparse_nodes = [node for node in self.graph.nodes.values() if (len(node.connections_in) + len(node.connections_out)) < 2]
        
        if saturated_nodes and sparse_nodes:
            for node in saturated_nodes:
                target = random.choice(sparse_nodes)
                # Si el nodo saturado tiene conexiones, quitar la de menor utilidad
                if node.connections_out:
                    min_conn = min(node.connections_out.items(), key=lambda item: item[1])
                    removed_util = node.connections_out.pop(min_conn[0])
                    self.log_institution(f"Removed low-utility connection from node {node.id} to {min_conn[0]} (Utility: {removed_util:.2f}).")
                    # Agregar nueva conexión que favorezca migración hacia cluster menos saturado
                    if self.graph.add_edge(node.id, target.id, utility=0.6):
                        self.log_institution(f"Migrated node {node.id} by connecting to sparse node {target.id}.")
        else:
            self.log_institution("No migration required at this step.")

# --- Agentes Operativos ---
class ProposerAgent(Synthesizer):
    def act(self):
        logging.info(f"ProposerAgent {self.id} acting.")

class EvaluatorAgent(Synthesizer):
    def act(self):
        logging.info(f"EvaluatorAgent {self.id} acting.")

class CombinerAgent(Synthesizer):
    def act(self):
        logging.info(f"CombinerAgent {self.id} acting.")

class AdvancedCoderAgent(Synthesizer):
    def act(self):
        logging.info(f"AdvancedCoderAgent {self.id} acting.")

class BridgingAgent(Synthesizer):
    def act(self):
        logging.info(f"BridgingAgent {self.id} acting.")

class KnowledgeFetcherAgent(Synthesizer):
    def act(self):
        logging.info(f"KnowledgeFetcherAgent {self.id} acting.")

class HorizonScannerAgent(Synthesizer):
    def act(self):
        logging.info(f"HorizonScannerAgent {self.id} acting.")

class EpistemicValidatorAgent(Synthesizer):
    def act(self):
        logging.info(f"EpistemicValidatorAgent {self.id} acting.")

class TechnogenesisAgent(Synthesizer):
    def act(self):
        logging.info(f"TechnogenesisAgent {self.id} acting.")

# --- Simulation Runner ---
class SimulationRunner:
    """Encapsulates and runs the simulation in a separate thread."""
    def __init__(self, config):
        self.config = config
        self.graph = CollectiveSynthesisGraph(config)
        self.agents = []
        self.is_running = False
        self.simulation_thread = None
        self.step_count = 0
        self.lock = threading.Lock()

        # --- Inicialización del Log de Métricas CSV ---
        self.metrics_file = None
        self.metrics_writer = None
        self.metrics_fieldnames = ["Step", "Nodes", "Edges", "Density", "AvgClustering", "Components", "MeanState", "MedianState", "StdDevState", "MinState", "MaxState"]
        metrics_path = self.config.get('metrics_log_path', None)
        if metrics_path:
            try:
                is_new_file = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
                self.metrics_file = open(metrics_path, 'a', newline='', encoding='utf-8')
                self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.metrics_fieldnames)
                if is_new_file:
                    self.metrics_writer.writeheader()
                logging.info(f"Logging global metrics to CSV: {metrics_path}")
            except Exception as e:
                logging.error(f"Error opening metrics log file: {e}")
                self.metrics_file = None
                self.metrics_writer = None

        load_path = self.config.get('load_state', None)
        if load_path:
            logging.info(f"Loading initial state from: {load_path}")
            if not self.graph.load_state(load_path):
                logging.error("State load failed. Starting fresh simulation.")
            else:
                logging.info("Initializing embeddings after state load...")
                self.graph.update_embeddings()
                if self.graph.node_embeddings:
                    logging.info(f"Initial embeddings for {len(self.graph.node_embeddings)} nodes available.")

        # Instanciación de agentes operativos
        num_proposers = config.get('num_proposers', 3)
        num_evaluators = config.get('num_evaluators', 6)
        num_combiners = config.get('num_combiners', 2)
        num_coders = config.get('num_coders', 1)
        num_bridging_agents = config.get('num_bridging_agents', 1)
        num_knowledge_fetchers = config.get('num_knowledge_fetchers', 1)
        num_horizon_scanners = config.get('num_horizon_scanners', 1)
        num_epistemic_validators = config.get('num_epistemic_validators', 1)
        num_technogenesis_agents = config.get('num_technogenesis_agents', 1)
        for i in range(num_proposers):
            self.agents.append(ProposerAgent(f"P{i}", self.graph, config))
        for i in range(num_evaluators):
            self.agents.append(EvaluatorAgent(f"E{i}", self.graph, config))
        for i in range(num_combiners):
            self.agents.append(CombinerAgent(f"C{i}", self.graph, config))
        for i in range(num_coders):
            self.agents.append(AdvancedCoderAgent(f"CD{i}", self.graph, config))
        for i in range(num_bridging_agents):
            self.agents.append(BridgingAgent(f"B{i}", self.graph, config))
        for i in range(num_knowledge_fetchers):
            self.agents.append(KnowledgeFetcherAgent(f"KF{i}", self.graph, config))
        for i in range(num_horizon_scanners):
            self.agents.append(HorizonScannerAgent(f"HS{i}", self.graph, config))
        for i in range(num_epistemic_validators):
            self.agents.append(EpistemicValidatorAgent(f"EV{i}", self.graph, config))
        for i in range(num_technogenesis_agents):
            self.agents.append(TechnogenesisAgent(f"TG{i}", self.graph, config))

        # Instanciación de agentes institucionales (Juzgado MSC, Universidad MSC, Instituto Financiero MSC)
        num_inspectors = self.config.get('num_inspectors', 1)
        for i in range(num_inspectors):
            self.agents.append(InspectorAgent(f"INSP{i}", self.graph, self.config))
        num_police = self.config.get('num_police', 1)
        for i in range(num_police):
            self.agents.append(PoliceAgent(f"POL{i}", self.graph, self.config))
        num_coordinators = self.config.get('num_coordinators', 1)
        for i in range(num_coordinators):
            self.agents.append(CoordinatorAgent(f"COORD{i}", self.graph, self.config))
        num_repair = self.config.get('num_repair', 1)
        for i in range(num_repair):
            self.agents.append(RepairAgent(f"REP{i}", self.graph, self.config))
        num_master = self.config.get('num_master', 1)
        for i in range(num_master):
            self.agents.append(MasterAgent(f"MAS{i}", self.graph, self.config))
        num_students = self.config.get('num_students', 1)
        for i in range(num_students):
            self.agents.append(StudentAgent(f"STU{i}", self.graph, self.config))
        num_scientists = self.config.get('num_scientists', 1)
        for i in range(num_scientists):
            self.agents.append(ScientistAgent(f"SCI{i}", self.graph, self.config))
        num_storage = self.config.get('num_storage', 1)
        for i in range(num_storage):
            self.agents.append(StorageAgent(f"STR{i}", self.graph, self.config))
        num_bank = self.config.get('num_bank', 1)
        for i in range(num_bank):
            self.agents.append(BankAgent(f"BANK{i}", self.graph, self.config))
        num_merchant = self.config.get('num_merchant', 1)
        for i in range(num_merchant):
            self.agents.append(MerchantAgent(f"MER{i}", self.graph, self.config))
        num_miner = self.config.get('num_miner', 1)
        for i in range(num_miner):
            self.agents.append(MinerAgent(f"MIN{i}", self.graph, self.config))
        num_population_regulators = self.config.get('num_population_regulators', 1)
        for i in range(num_population_regulators):
            self.agents.append(PopulationRegulatorAgent(f"POP{i}", self.graph, self.config))
        num_seeders = self.config.get('num_seeders', 1)
        for i in range(num_seeders):
            self.agents.append(SeederAgent(f"SEED{i}", self.graph, self.config))
        num_cluster_balancers = self.config.get('num_cluster_balancers', 1)
        for i in range(num_cluster_balancers):
            self.agents.append(ClusterBalancerAgent(f"CLB{i}", self.graph, self.config))
        num_mediators = self.config.get('num_mediators', 1)
        for i in range(num_mediators):
            self.agents.append(MediatorAgent(f"MED{i}", self.graph, self.config))
        num_migration_agents = self.config.get('num_migration_agents', 1)
        for i in range(num_migration_agents):
            self.agents.append(MigrationAgent(f"MIG{i}", self.graph, self.config))

        logging.info(f"Created agents: "
                     f"Proposers={config.get('num_proposers',0)}, Evaluators={config.get('num_evaluators',0)}, "
                     f"Combiners={config.get('num_combiners',0)}, Bridging={num_bridging_agents}, "
                     f"KnowledgeFetchers={num_knowledge_fetchers}, HorizonScanners={num_horizon_scanners}, "
                     f"EpistemicValidators={num_epistemic_validators}, TechnogenesisAgents={num_technogenesis_agents}, "
                     f"Inspectors={num_inspectors}, Police={num_police}, Coordinators={num_coordinators}, "
                     f"RepairAgents={num_repair}, Masters={num_master}, Students={num_students}, Scientists={num_scientists}, "
                     f"StorageAgents={num_storage}, BankAgents={num_bank}, MerchantAgents={num_merchant}, MinerAgents={num_miner}, "
                     f"PopulationRegulators={num_population_regulators}, Seeders={num_seeders}, ClusterBalancers={num_cluster_balancers}, Mediators={num_mediators}, MigrationAgents={num_migration_agents}")

    def _simulation_loop(self):
        step_delay = self.config.get('step_delay', 0.1)
        gnn_update_frequency = self.config.get('gnn_update_frequency', 10)
        summary_frequency = self.config.get('summary_frequency', 50)
        max_steps = self.config.get('simulation_steps', None)
        is_api_mode = self.config.get('run_api', False)
        run_continuously = is_api_mode or (max_steps is None)
        gnn_training_frequency = self.config.get('gnn_training_frequency', 50)
        gnn_training_epochs = self.config.get('gnn_training_epochs', 5)
        while self.is_running:
            self.step_count += 1
            log_level = logging.DEBUG if self.step_count % summary_frequency != 0 else logging.INFO
            logging.log(log_level, f"--- Step {self.step_count} ---")
            if not run_continuously and max_steps is not None and self.step_count > max_steps:
                logging.info(f"Reached max steps ({max_steps}). Stopping simulation.")
                self.is_running = False
                break
            if not self.agents:
                logging.warning("No agents available to act.")
                self.is_running = False
                break
            with self.lock:
                if self.step_count > 0 and self.step_count % gnn_update_frequency == 0:
                    self.graph.update_embeddings()
                if self.step_count > 0 and self.step_count % gnn_training_frequency == 0:
                    logging.info(f"GNN training at step {self.step_count}...")
                    self.graph.train_gnn(num_epochs=gnn_training_epochs)
                runnable_agents = []
                agent_costs = {
                    "ProposerAgent": self.config.get('proposer_cost', 1.0),
                    "EvaluatorAgent": self.config.get('evaluator_cost', 0.5),
                    "CombinerAgent": self.config.get('combiner_cost', 1.5),
                    "BridgingAgent": self.config.get('bridging_agent_cost', 2.0),
                    "KnowledgeFetcherAgent": self.config.get('knowledge_fetcher_cost', 2.5),
                    "HorizonScannerAgent": self.config.get('horizonscanner_cost', 3.0),
                    "EpistemicValidatorAgent": self.config.get('epistemic_validator_cost', 2.0),
                    "TechnogenesisAgent": self.config.get('technogenesis_cost', 2.5)
                }
                for agent in self.agents:
                    cost = agent_costs.get(type(agent).__name__, 1.0)
                    if agent.omega >= cost:
                        runnable_agents.append({'agent': agent, 'cost': cost})
                if runnable_agents:
                    chosen = random.choice(runnable_agents)
                    agent = chosen['agent']
                    cost = chosen['cost']
                    omega_before = agent.omega
                    agent.omega -= cost
                    agent.act()
                    logging.debug(f"Agent {agent.id} acted (Cost: {cost:.2f}, Omega: {omega_before:.2f} -> {agent.omega:.2f})")
                else:
                    logging.warning("No agents have sufficient Omega to act.")
                regen_rate = self.config.get('omega_regeneration_rate', 0.05)
                if regen_rate > 0:
                    for a in self.agents:
                        a.omega += regen_rate * a.reputation
                if self.step_count % summary_frequency == 0:
                    self.graph.print_summary(logging.INFO)
                    if self.metrics_writer:
                        metrics = self.graph.get_global_metrics()
                        if metrics:
                            metrics["Step"] = self.step_count
                        try:
                            self.metrics_writer.writerow(metrics)
                            self.metrics_file.flush()
                        except Exception as e:
                            logging.error(f"CSV write error at step {self.step_count}: {e}")
            time.sleep(step_delay)
        logging.info("--- Simulation loop finished ---")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
            logging.info("--- Simulation thread started ---")
        else:
            logging.warning("Simulation already running.")

    def stop(self):
        if self.is_running:
            logging.info("--- Stopping simulation ---")
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
                logging.info(f"GNN: Final embeddings computed for {len(self.graph.node_embeddings)} nodes.")
            save_path = self.config.get('save_state', None)
            if save_path:
                logging.info(f"Saving final state to: {save_path}")
                self.graph.save_state(save_path)
            logging.info("--- Final Graph State ---")
            self.graph.print_summary(logging.INFO)
            self.graph.log_global_metrics(logging.INFO, current_step=self.step_count)
            if self.metrics_writer:
                logging.info("Writing final metrics to CSV...")
                metrics = self.graph.get_global_metrics()
                if metrics:
                    metrics["Step"] = self.step_count
                    try:
                        self.metrics_writer.writerow(metrics)
                    except Exception as e:
                        logging.error(f"Final CSV write error: {e}")
            if self.metrics_file:
                try:
                    self.metrics_file.close()
                    logging.info("Metrics log file closed.")
                except Exception as e:
                    logging.error(f"Error closing metrics file: {e}")
            save_vis_path = self.config.get('visualization_output_path', None)
            if save_vis_path:
                if self.graph.nodes:
                    logging.info(f"Saving final graph visualization to {save_vis_path}...")
                    try:
                        self.graph.visualize_graph(self.config)
                    except Exception as e:
                        logging.error(f"Error saving visualization: {e}")
                else:
                    logging.warning("Graph empty, skipping visualization.")
        logging.info("--- Simulation Runner Stopped ---")

    def get_status(self):
        with self.lock:
            num_nodes = len(self.graph.nodes)
            num_edges = sum(len(n.connections_out) for n in self.graph.nodes.values())
            avg_state = (sum(n.state for n in self.graph.nodes.values()) / num_nodes) if num_nodes > 0 else 0
            num_agents = len(self.agents)
            avg_reputation = (sum(a.reputation for a in self.agents) / num_agents) if num_agents > 0 else 0.0
            avg_omega = (sum(a.omega for a in self.agents) / num_agents) if num_agents > 0 else 0.0
            return {
                "is_running": self.is_running,
                "current_step": self.step_count,
                "node_count": num_nodes,
                "edge_count": num_edges,
                "average_state": round(avg_state, 3),
                "embeddings_count": len(self.graph.node_embeddings),
                "average_reputation": round(avg_reputation, 3),
                "average_omega": round(avg_omega, 2)
            }

    def get_graph_elements_for_cytoscape(self):
        elements = []
        with self.lock:
            try:
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=0, vmax=1)
                for node_id, node in self.graph.nodes.items():
                    try:
                        node_color = cmap(norm(node.state))
                        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
                    except Exception as ex:
                        app.logger.error("Error processing node %s: %s", node_id, ex)
                        hex_color = "#cccccc"
                    elements.append({
                        'data': {
                            'id': str(node_id),
                            'label': f'Node {node_id}\nS={node.state:.2f}',
                            'state': node.state,
                            'keywords': ", ".join(sorted(list(node.keywords))) if node.keywords else ""
                        },
                        'style': {
                            'background-color': hex_color,
                            'width': f"{20 + node.state * 40}px",
                            'height': f"{20 + node.state * 40}px"
                        }
                    })
                for source_id, node in self.graph.nodes.items():
                    for target_id, utility in node.connections_out.items():
                        if source_id in self.graph.nodes and target_id in self.graph.nodes:
                            elements.append({
                                'data': {
                                    'source': str(source_id),
                                    'target': str(target_id),
                                    'utility': utility
                                },
                                'style': {
                                    'width': 1 + abs(utility) * 2,
                                    'line-color': 'red' if utility < 0 else ('blue' if utility > 0 else 'grey')
                                }
                            })
            except Exception as e:
                app.logger.error("Unexpected error in get_graph_elements_for_cytoscape: %s", e)
                raise e
        return elements

def load_config(args):
    config = {
        'simulation_steps': None,
        'num_proposers': 3,
        'num_evaluators': 6,
        'num_combiners': 2,
        'step_delay': 0.1,
        'evaluator_learning_rate': 0.1,
        'evaluator_decay_rate': 0.01,
        'combiner_compatibility_threshold': 0.6,
        'combiner_similarity_threshold': 0.7,
        'gnn_hidden_dim': 16,
        'gnn_embedding_dim': 8,
        'gnn_update_frequency': 10,
        'save_state': None,
        'load_state': None,
        'summary_frequency': 50,
        'run_api': False,
        'gnn_training_frequency': 50,
        'gnn_training_epochs': 5,
        'gnn_learning_rate': 0.01,
        'initial_omega': 100.0,
        'proposer_cost': 1.0,
        'evaluator_cost': 0.5,
        'combiner_cost': 1.5,
        'omega_regeneration_rate': 0.1,
        'evaluator_reward_factor': 2.0,
        'evaluator_reward_threshold': 0.05,
        'proposer_reward_factor': 0.5,
        'combiner_reward_factor': 1.0,
        'visualization_output_path': None,
        'metrics_log_path': None,
        'num_bridging_agents': 1,
        'bridging_agent_cost': 2.0,
        'bridging_similarity_threshold': 0.75,
        'bridging_adjusted_threshold': 0.65,
        'knowledge_fetcher_cost': 2.5,
        'num_knowledge_fetchers': 1,
        'horizonscanner_cost': 3.0,
        'horizonscanner_psi_reward': 0.05,
        'num_horizon_scanners': 1,
        'epistemic_validator_cost': 2.0,
        'epistemic_validator_omega_reward': 0.05,
        'epistemic_validator_psi_reward': 0.01,
        'num_epistemic_validators': 1,
        'technogenesis_cost': 2.5,
        'technogenesis_omega_reward': 0.05,
        'technogenesis_psi_reward': 0.01,
        'num_technogenesis_agents': 1,
        'num_inspectors': 1,
        'num_police': 1,
        'num_coordinators': 1,
        'num_repair': 1,
        'num_master': 1,
        'num_students': 1,
        'num_scientists': 1,
        'num_storage': 1,
        'num_bank': 1,
        'num_merchant': 1,
        'num_miner': 1,
        'num_population_regulators': 1,
        'num_seeders': 1,
        'num_cluster_balancers': 1,
        'num_mediators': 1,
        'num_migration_agents': 1,
    }
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
            logging.info(f"Loaded config from: {args.config}")
        except FileNotFoundError:
            logging.warning(f"Config file not found at {args.config}.")
        except Exception as e:
            logging.error(f"Error loading config from {args.config}: {e}.")
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

simulation_runner = None
app = Flask(__name__)

@app.route('/status')
def get_simulation_status():
    if simulation_runner:
        return jsonify(simulation_runner.get_status())
    else:
        return jsonify({"error": "Simulation not initialized"}), 500

@app.route('/graph_data')
def get_graph_data():
    try:
        if simulation_runner:
            elements = simulation_runner.get_graph_elements_for_cytoscape()
            return jsonify(elements)
        else:
            return jsonify({"error": "Simulation not initialized"}), 500
    except Exception as e:
        app.logger.error("Error in /graph_data: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/control/stop', methods=['POST'])
def stop_simulation():
    if simulation_runner:
        simulation_runner.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Simulation not initialized"}), 500

@app.route('/control/start', methods=['POST'])
def start_simulation():
    if simulation_runner and not simulation_runner.is_running:
        simulation_runner.start()
        return jsonify({"status": "started"})
    return jsonify({"error": "Simulation already running or not initialized"}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MSC Simulation with GNN and optional API.")
    parser.add_argument('--config', type=str, help='Path to YAML config file.')
    parser.add_argument('--simulation_steps', type=int, default=None, help='Fixed number of simulation steps.')
    parser.add_argument('--num_proposers', type=int)
    parser.add_argument('--num_evaluators', type=int)
    parser.add_argument('--num_combiners', type=int)
    parser.add_argument('--step_delay', type=float)
    parser.add_argument('--evaluator_learning_rate', type=float)
    parser.add_argument('--evaluator_decay_rate', type=float)
    parser.add_argument('--combiner_compatibility_threshold', type=float)
    parser.add_argument('--combiner_similarity_threshold', type=float)
    parser.add_argument('--gnn_hidden_dim', type=int)
    parser.add_argument('--gnn_embedding_dim', type=int)
    parser.add_argument('--gnn_update_frequency', type=int)
    parser.add_argument('--visualize_graph', action='store_true', help='Show graph visualization at end (only in fixed-step mode).')
    parser.add_argument('--save_state', type=str, help='Base path to save simulation state upon stop.')
    parser.add_argument('--load_state', type=str, help='Base path to load simulation state from.')
    parser.add_argument('--visualization_output_path', type=str, default=None, help='Path to save final graph visualization image.')
    parser.add_argument('--metrics_log_path', type=str, default=None, help='Path to save global metrics CSV log.')
    parser.add_argument('--run_api', action='store_true', help='Run the Flask API server.')
    parser.add_argument('--summary_frequency', type=int, help='Frequency (steps) for summary logging.')
    parser.add_argument('--initial_omega', type=float, help='Initial Omega resource for agents.')
    parser.add_argument('--proposer_cost', type=float, help='Omega cost for Proposer action.')
    parser.add_argument('--evaluator_cost', type=float, help='Omega cost for Evaluator action.')
    parser.add_argument('--combiner_cost', type=float, help='Omega cost for Combiner action.')
    parser.add_argument('--omega_regeneration_rate', type=float, help='Omega regenerated per agent per step.')
    parser.add_argument('--evaluator_reward_factor', type=float, help='Omega reward multiplier for evaluation.')
    parser.add_argument('--evaluator_reward_threshold', type=float, help='Minimum state increase for evaluator reward.')
    parser.add_argument('--proposer_reward_factor', type=float, help='Omega reward multiplier for proposer.')
    parser.add_argument('--combiner_reward_factor', type=float, help='Omega reward multiplier for combiner.')
    parser.add_argument('--gnn_training_frequency', type=int, help='Frequency (steps) to train the GNN.')
    parser.add_argument('--gnn_training_epochs', type=int, help='Epochs per GNN training iteration.')
    parser.add_argument('--gnn_learning_rate', type=float, help='Learning rate for the GNN optimizer.')
    parser.add_argument('--num_bridging_agents', type=int, help='Number of BridgingAgents.')
    parser.add_argument('--bridging_agent_cost', type=float, help='Omega cost for BridgingAgent action.')
    parser.add_argument('--bridging_similarity_threshold', type=float, help='Similarity threshold for bridging (0-1).')
    parser.add_argument('--bridging_adjusted_threshold', type=float, help='Adjusted similarity threshold for bridging.')
    parser.add_argument('--knowledge_fetcher_cost', type=float, help='Omega cost for KnowledgeFetcher action.')
    parser.add_argument('--num_knowledge_fetchers', type=int, default=1, help='Number of KnowledgeFetcher agents.')
    parser.add_argument('--horizonscanner_cost', type=float, help='Omega cost for HorizonScanner action.')
    parser.add_argument('--horizonscanner_psi_reward', type=float, help='Psi reward for HorizonScanner action.')
    parser.add_argument('--num_horizon_scanners', type=int, default=1, help='Number of HorizonScanner agents.')
    parser.add_argument('--epistemic_validator_cost', type=float, help='Omega cost for EpistemicValidator action.')
    parser.add_argument('--epistemic_validator_omega_reward', type=float, help='Omega reward for EpistemicValidator action.')
    parser.add_argument('--epistemic_validator_psi_reward', type=float, help='Psi reward for EpistemicValidator action.')
    parser.add_argument('--num_epistemic_validators', type=int, default=1, help='Number of EpistemicValidator agents.')
    parser.add_argument('--technogenesis_cost', type=float, help='Omega cost for TechnogenesisAgent action.')
    parser.add_argument('--technogenesis_omega_reward', type=float, help='Omega reward for TechnogenesisAgent action.')
    parser.add_argument('--technogenesis_psi_reward', type=float, help='Psi reward for TechnogenesisAgent action.')
    parser.add_argument('--num_technogenesis_agents', type=int, default=1, help='Number of TechnogenesisAgent agents.')
    parser.add_argument('--num_inspectors', type=int, help='Number of InspectorAgents.')
    parser.add_argument('--num_police', type=int, help='Number of PoliceAgents.')
    parser.add_argument('--num_coordinators', type=int, help='Number of CoordinatorAgents.')
    parser.add_argument('--num_repair', type=int, help='Number of RepairAgents.')
    parser.add_argument('--num_master', type=int, help='Number of MasterAgents.')
    parser.add_argument('--num_students', type=int, help='Number of StudentAgents.')
    parser.add_argument('--num_scientists', type=int, help='Number of ScientistAgents.')
    parser.add_argument('--num_storage', type=int, help='Number of StorageAgents.')
    parser.add_argument('--num_bank', type=int, help='Number of BankAgents.')
    parser.add_argument('--num_merchant', type=int, help='Number of MerchantAgents.')
    parser.add_argument('--num_miner', type=int, help='Number of MinerAgents.')
    parser.add_argument('--num_population_regulators', type=int, help='Number of PopulationRegulatorAgents.')
    parser.add_argument('--num_seeders', type=int, help='Number of SeederAgents.')
    parser.add_argument('--num_cluster_balancers', type=int, help='Number of ClusterBalancerAgents.')
    parser.add_argument('--num_mediators', type=int, help='Number of MediatorAgents.')
    parser.add_argument('--num_migration_agents', type=int, help='Number of MigrationAgents.')
    args = parser.parse_args()
    final_config = load_config(args)
    final_config['run_api'] = args.run_api
    simulation_runner = SimulationRunner(final_config)
    simulation_runner.start()
    time.sleep(2)
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

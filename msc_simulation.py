# Modificación en la parte de importaciones - Agregar nuestra clase del nuevo visualizador
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
import concurrent.futures
from flask_socketio import SocketIO
from tqdm import tqdm
import json
import re
import importlib.util
import sys  # Para la carga dinámica de módulos
import threading
import logging
from flask import Flask, jsonify

# Importar el nuevo visualizador
try:
    from msc_viewer import MSCViewerServer, SimulationAdapter
    MSC_VIEWER_AVAILABLE = True
    logging.info("MSC Viewer imported successfully")
except ImportError:
    MSC_VIEWER_AVAILABLE = False
    logging.warning("MSC Viewer module not found. Visualization disabled.")

load_dotenv()  # Carga variables desde .env al entorno

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: 'sentence-transformers' not found. Text embeddings disabled.")
    SentenceTransformer = None

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
    wikipedia.set_lang("es")  # O usa "en" si prefieres inglés
except ImportError:
    print("WARNING: 'wikipedia' library not found. KnowledgeFetcherAgent disabled.")
    WIKIPEDIA_AVAILABLE = False
    wikipedia = None

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
        self.lock = threading.Lock()  # Añadir esta línea
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
        self._last_modified = time.time()

    # Performance optimization for GNN processing
    def _prepare_pyg_data(self):
        # Cache results for unchanged graph structure
        if hasattr(self, '_cached_node_features') and hasattr(self, '_cached_edge_index') and \
           len(self.nodes) == self._cached_graph_size and self._cached_last_modified == self._last_modified:
            return self._cached_node_features, self._cached_edge_index

        # Original implementation continues...
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

        # Cache the results
        self._cached_node_features = node_features
        self._cached_edge_index = edge_index
        self._cached_graph_size = len(self.nodes)
        self._cached_last_modified = self._last_modified
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
        self._last_modified = time.time()
        return new_node

    def add_edge(self, source_id, target_id, utility):
        if source_id in self.nodes and target_id in self.nodes and source_id != target_id:
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            if target_id not in source_node.connections_out:
                source_node.add_connection(target_node, utility)
                self._last_modified = time.time()
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
        try:
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=0, vmax=1)
            for node_id, node in self.nodes.items():
                try:
                    node_color = cmap(norm(node.state))
                    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
                except Exception as ex:
                    logging.error(f"Error processing node {node_id}: {ex}")
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
            for source_id, node in self.nodes.items():
                for target_id, utility in node.connections_out.items():
                    if target_id in self.nodes:
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
            logging.error(f"Unexpected error in get_graph_elements_for_cytoscape: {e}")
        return elements

    def trim_stale_nodes(self, threshold=0.05, max_retain=1000):
        """Remove nodes with very low state to manage memory if graph grows too large"""
        if len(self.nodes) <= max_retain:
            return 0
            
        # Find stale nodes (low state, few connections)
        candidates = sorted(
            [(node_id, node) for node_id, node in self.nodes.items()],
            key=lambda x: (x[1].state, len(x[1].connections_in) + len(x[1].connections_out))
        )
        
        # Keep nodes until we're under the limit or hit important nodes
        removed = 0
        for node_id, node in candidates:
            if len(self.nodes) <= max_retain:
                break
            if node.state <= threshold and "important" not in node.keywords:
                # Remove all connections to this node
                for other_id, other_node in self.nodes.items():
                    if node_id in other_node.connections_out:
                        other_node.connections_out.pop(node_id)
                    if node_id in other_node.connections_in:
                        other_node.connections_in.pop(node_id)
                # Remove the node
                self.nodes.pop(node_id)
                removed += 1
                
        if removed > 0:
            logging.info(f"Memory management: Removed {removed} stale nodes")
            # Invalidate caches after structure change
            if hasattr(self, '_cached_node_features'):
                delattr(self, '_cached_node_features')
            if hasattr(self, '_cached_edge_index'):
                delattr(self, '_cached_edge_index')
            self._last_modified = time.time()
            
        return removed

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

# Agregar después de la función generate_code
def execute_generated_code(code_string, context_vars=None):
    """
    Ejecuta código generado dinámicamente en un entorno controlado.
    
    Args:
        code_string (str): El código Python a ejecutar
        context_vars (dict): Variables de contexto a pasar al código
        
    Returns:
        tuple: (result, error) - Resultado de la ejecución y cualquier error
    """
    if not code_string:
        return None, "Código vacío"
    
    # Validación básica de seguridad
    forbidden_keywords = ['import os', 'subprocess', 'system', '__import__', 'eval(', 'exec(', 
                         'open(', 'file(', 'delete', 'remove(', 'rmdir', 'shutil']
    
    for keyword in forbidden_keywords:
        if keyword in code_string:
            return None, f"Código inseguro detectado: {keyword}"
    
    # Preparar entorno de ejecución limitado
    safe_globals = {
        'math': math,
        'random': random,
        'statistics': statistics,
        'Counter': Counter,
        'logging': logging,
        'torch': torch,
        'F': F,
        'nx': nx,
        'np': torch.tensor, # Simulando NumPy limitado
    }
    
    local_vars = context_vars or {}
    
    try:
        # Extraer nombre de la función del código
        import re
        func_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', code_string)
        if not func_match:
            return None, "No se encontró definición de función"
        
        func_name = func_match.group(1)
        
        # Ejecutar el código en el entorno limitado
        exec(code_string, safe_globals, local_vars)
        
        # Verificar que la función existe
        if func_name not in local_vars:
            return None, f"La función {func_name} no se definió correctamente"
        
        # Devolver la función
        return local_vars[func_name], None
        
    except Exception as e:
        return None, f"Error ejecutando código: {str(e)}"

# Clase para persistir y gestionar el código autogenerado
class CodeRepository:
    def __init__(self, save_dir="generated_code"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.functions = {}
        self.load_saved_functions()
        
    def save_function(self, name, code, metadata=None):
        """Guarda una función generada con metadatos"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.save_dir}/{name}_{timestamp}.py"
        
        metadata = metadata or {}
        metadata["timestamp"] = timestamp
        
        with open(filename, 'w') as f:
            f.write(f"# Auto-generated by TAEC\n")
            f.write(f"# {json.dumps(metadata)}\n\n")
            f.write(code)
        
        self.functions[name] = {
            'code': code,
            'metadata': metadata,
            'path': filename
        }
        return filename
    
    def load_function(self, name):
        """Carga una función por nombre"""
        if name in self.functions:
            return self.functions[name]['code']
        return None
    
    def load_saved_functions(self):
        """Carga todas las funciones guardadas"""
        if not os.path.exists(self.save_dir):
            return
            
        for file in os.listdir(self.save_dir):
            if file.endswith(".py"):
                try:
                    path = os.path.join(self.save_dir, file)
                    name = file.split('_')[0]
                    with open(path, 'r') as f:
                        code = f.read()
                    
                    # Extraer metadatos
                    import re
                    metadata_match = re.search(r'# (\{.*\})', code)
                    metadata = json.loads(metadata_match.group(1)) if metadata_match else {}
                    
                    self.functions[name] = {
                        'code': code,
                        'metadata': metadata,
                        'path': path
                    }
                except Exception as e:
                    logging.error(f"Error loading function from {file}: {e}")

# --- Clase Base para Agentes ---
class Synthesizer:
    def __init__(self, agent_id, graph, config):
        self.id = agent_id
        self.graph = graph
        self.config = config
        self.omega = config.get('initial_omega', 100.0)
        self.reputation = 1.0

    def act(self):
        logging.info(f"{type(self).__name__} {self.id} acting (base implementation).")

# --- Clase Base para Agentes Institucionales ---
class InstitutionAgent(ABC):
    def __init__(self, agent_id, graph, config):
        self.id = agent_id
        self.graph = graph
        self.config = config
        self.omega = config.get('initial_omega', 100.0)
        self.reputation = 1.0
        
    def act(self):
        """Implementa la acción institucional principal"""
        logging.info(f"InstitutionAgent {self.id} acting...")
        self.institution_action()
        
    @abstractmethod
    def institution_action(self):
        """Método abstracto a implementar por cada agente institucional"""
        pass
        
    def log_institution(self, message):
        """Registra mensajes de acción institucional"""
        logging.info(f"[Institution {self.id}] {message}")

class RepairAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Repairing nodes with low reputation despite high state...")
        for node in self.graph.nodes.values():
            if node.state > 0.7 and hasattr(node, 'reputation') and node.reputation < 0.5:
                self.log_institution(f"Rehabilitating node {node.id}.")
                node.update_state(min(1.0, node.state + 0.1))

class InspectorAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Inspecting node states and connections...")
        for node in self.graph.nodes.values():
            if node.state < 0.1:
                node.update_state(node.state + 0.05)
                self.log_institution(f"Improved low-state node {node.id}")

class PoliceAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Enforcing rules and pruning invalid connections...")
        for node in self.graph.nodes.values():
            if len(node.connections_out) > 15:  # Demasiadas conexiones
                # Quitar algunas conexiones
                self.log_institution(f"Node {node.id} has too many connections. Pruning.")
                connections = list(node.connections_out.keys())
                for i in range(5):  # Quitar 5 conexiones
                    if connections:
                        del node.connections_out[connections.pop()]

class CoordinatorAgent(InstitutionAgent):
    def institution_action(self):
        self.log_institution("Coordinating agent activities...")
        # Ejemplo: redistribuir omega entre agentes
        if hasattr(self.graph, 'agents'):
            agents_low_omega = [a for a in self.graph.agents if a.omega < 50]
            if agents_low_omega:
                for agent in agents_low_omega:
                    agent.omega += 5
                    self.log_institution(f"Boosted omega for agent {agent.id}")

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
        if len(self.graph.nodes) < 10:
            new_node = self.graph.add_node(content="Seeded Node", initial_state=0.2)
            self.log_institution(f"Seeded new node {new_node.id} due to low overall density.")
        else:
            sparse_nodes = [n for n in self.graph.nodes.values() if (len(n.connections_in) + len(n.connections_out)) < 2]
            if sparse_nodes:
                # Inyecta contenido estratégico para vincular clusters aislados
                new_content = "Strategic Seed: Bridge Node"
                new_node = self.graph.add_node(content=new_content, initial_state=0.35, keywords={"bridge", "seed"})
                target = random.choice(sparse_nodes)
                if self.graph.add_edge(target.id, new_node.id, utility=0.5):
                    self.log_institution(f"Seeded strategic node {new_node.id} attached to sparse node {target.id}.")
            else:
                self.log_institution("No significantly sparse regions detected for seeding.")

class ClusterBalancerAgent(InstitutionAgent):
    """
    Redistribuye conexiones entre grupos de nodos para evitar desequilibrios topológicos.
    """
    def institution_action(self):
        self.log_institution("Balancing clusters in the graph...")
        for node in self.graph.nodes.values():
            if len(node.connections_out) > 10:
                min_conn = min(node.connections_out.items(), key=lambda item: item[1])
                node.connections_out.pop(min_conn[0])
                self.log_institution(f"Removed low-utility connection from node {node.id} (over-connected).")
            elif len(node.connections_out) < 2:
                # Seleccionar candidato preferentemente de un cluster diferente (mínima intersección de keywords)
                candidates = [candidate for candidate in self.graph.nodes.values()
                              if candidate.id != node.id and len(node.keywords.intersection(candidate.keywords)) < 1]
                candidate = random.choice(candidates) if candidates else self.graph.get_random_node_biased()
                if candidate and candidate.id != node.id:
                    if self.graph.add_edge(node.id, candidate.id, utility=0.5):
                        self.log_institution(f"Added strategic connection from node {node.id} to candidate {candidate.id}.")

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
        self.log_institution("Evaluating clusters for adaptive migration with aggressive strategy...")
        saturated_nodes = [node for node in self.graph.nodes.values() if len(node.connections_out) > 8]
        sparse_nodes = [node for node in self.graph.nodes.values() if (len(node.connections_in) + len(node.connections_out)) < 2]
        if saturated_nodes and sparse_nodes:
            for node in saturated_nodes:
                migration_attempts = 0
                while len(node.connections_out) > 8 and migration_attempts < 3:
                    # Seleccionar target de sparse_nodes que comparta pocas keywords con el nodo actual
                    filtered_targets = [t for t in sparse_nodes if len(node.keywords.intersection(t.keywords)) < 1]
                    target = random.choice(filtered_targets) if filtered_targets else random.choice(sparse_nodes)
                    if node.connections_out:
                        min_conn = min(node.connections_out.items(), key=lambda item: item[1])
                        removed_util = node.connections_out.pop(min_conn[0])
                        self.log_institution(f"Removed low-utility connection from node {node.id} to {min_conn[0]} (Utility: {removed_util:.2f}).")
                        if self.graph.add_edge(node.id, target.id, utility=0.6):
                            self.log_institution(f"Migrated node {node.id} by connecting to sparse node {target.id}.")
                    migration_attempts += 1
        else:
            self.log_institution("No migration required at this step.")

# Agregar después del MigrationAgent
class TAECEvolutionAgent(InstitutionAgent):
    """
    Implementa la Teoría de Auto-Evolución Cognitiva (TAEC) para evolucionar 
    el grafo y su propio código de manera autónoma, adaptándose a patrones emergentes.
    """
    def institution_action(self):
        self.log_institution("Iniciando ciclo de auto-evolución TAEC...")
        
        # 1. Analizar el estado actual del grafo
        metrics = self.graph.get_global_metrics()
        mean_state = metrics.get('MeanState', 0)
        node_count = len(self.graph.nodes)
        
        # 2. Identificar áreas para evolución cognitiva
        high_potential_nodes = [node for node in self.graph.nodes.values() 
                               if node.state > 0.7 and len(node.connections_out) > 3]
        
        low_connectivity_nodes = [node for node in self.graph.nodes.values() 
                                 if len(node.connections_out) < 2 and node.state > 0.4]
                                 
        # 3. Auto-modificación de código utilizando generates de prompts dinámicos
        if hasattr(self, 'step_count') and self.step_count % 100 == 0:
            self.log_institution("TAEC: Iniciando auto-modificación de código...")
            
            # Verificar que tengamos acceso al repositorio de código
            if not hasattr(self.graph, 'code_repository'):
                # Inicializar repositorio si no existe
                self.graph.code_repository = CodeRepository(
                    save_dir=os.path.join(os.path.dirname(self.config.get('save_state', 'generated_code')), 'taec_code')
                )
            
            # Generar prompt basado en el estado actual del grafo
            prompt = f"""
            Genera una función Python que mejore el algoritmo de evolución del grafo MSC.
            La función debe analizar nodos con estado medio de {mean_state:.3f} 
            y crear conexiones estratégicas para optimizar la fluidez cognitiva.
            
            Los nodos tienen estas propiedades:
            - state: Valor entre 0 y 1 que representa su estado de activación
            - keywords: Conjunto de palabras clave (set)
            - connections_out: Diccionario de conexiones salientes (key=node_id, value=utility)
            
            Usa este formato exactamente:
            
            def enhance_graph_evolution(graph, target_nodes):
                '''Función generada por TAEC para mejorar la evolución del grafo'''
                # código aquí
                return {'modified_nodes': count, 'average_improvement': value}
            """
            
            # Esta parte genera nuevo código funcional
            if 'generate_code' in globals():
                new_code = generate_code(prompt)
                if new_code:
                    self.log_institution(f"TAEC: Código auto-generado: {len(new_code)} caracteres")
                    
                    # Evaluar e integrar el código
                    function, error = execute_generated_code(new_code)
                    
                    if error:
                        self.log_institution(f"TAEC: Error en código generado: {error}")
                    else:
                        # Guardar la función en el repositorio
                        function_name = "enhance_graph_evolution"
                        metadata = {
                            'step': self.step_count,
                            'mean_state': mean_state,
                            'node_count': len(self.graph.nodes)
                        }
                        self.graph.code_repository.save_function(function_name, new_code, metadata)
                        
                        # Ejecutar la función con el grafo actual
                        try:
                            # Seleccionar nodos para evolucionar
                            suitable_nodes = [n for n in self.graph.nodes.values() 
                                             if n.state > 0.4 and len(n.connections_out) > 2]
                            
                            if suitable_nodes:
                                target_nodes = random.sample(suitable_nodes, 
                                                            min(10, len(suitable_nodes)))
                                
                                # Ejecutar la función generada
                                results = function(self.graph, target_nodes)
                                self.log_institution(f"TAEC: Función ejecutada con resultados: {results}")
                                
                                # Marcar nodos que han sido modificados
                                for node in target_nodes:
                                    node.keywords.add("taec_auto_evolved")
                        
                        except Exception as execution_error:
                            self.log_institution(f"TAEC: Error ejecutando función: {execution_error}")
                            
                    # Añadir a la lista de palabras clave que este agente ha generado código
                    if not hasattr(self, 'keywords'):
                        self.keywords = set()
                    self.keywords.add("taec_code_evolution")
        
        # 4. Evolución mediante síntesis avanzada
        if high_potential_nodes:
            # Seleccionar múltiples nodos fuente para síntesis emergente
            sources = random.sample(high_potential_nodes, min(3, len(high_potential_nodes)))
            keywords = set()
            for source in sources:
                keywords.update(source.keywords)
            
            keywords.add("taec_evolved")
            content = f"TAEC Evolution: Meta-synthesis from {[s.id for s in sources]}"
            
            # Crear nodo con estado emergente aumentado
            new_state = min(1.0, sum(s.state for s in sources)/len(sources) + 0.15)
            new_node = self.graph.add_node(
                content=content,
                initial_state=new_state,
                keywords=keywords
            )
            
            # Conectar bidireccionalmente con fuentes de inspiración
            for source in sources:
                self.graph.add_edge(source.id, new_node.id, utility=0.8)
                self.graph.add_edge(new_node.id, source.id, utility=0.6)
                
            self.log_institution(f"TAEC: Creado nodo evolucionado {new_node.id} a partir de {len(sources)} nodos fuente")
            
            # Buscar conexiones emergentes adicionales
            for _ in range(3):
                target = self.graph.get_random_node_biased()
                if target and target.id != new_node.id and target not in sources:
                    utility = random.uniform(0.5, 0.9)
                    self.graph.add_edge(new_node.id, target.id, utility=utility)
        
        # 5. Recuperación auto-adaptativa de nodos aislados
        if low_connectivity_nodes:
            for i in range(min(3, len(low_connectivity_nodes))):
                target = low_connectivity_nodes[i]
                target.keywords.add("taec_enhanced")
                
                # Incremento adaptativo basado en el promedio global
                enhancement = 0.1 + (mean_state - target.state) * 0.2
                target.update_state(target.state + enhancement)
                
                # Agregar conexiones estratégicas basadas en patrones emergentes
                potential_connections = [n for n in self.graph.nodes.values() 
                                       if n.id != target.id and 
                                       (n.state > mean_state or "taec_evolved" in n.keywords)]
                                       
                if potential_connections:
                    # Conectar en ambos sentidos para crear flujo bidireccional
                    new_connection = random.choice(potential_connections)
                    if self.graph.add_edge(target.id, new_connection.id, utility=0.7):
                        self.graph.add_edge(new_connection.id, target.id, utility=0.5)
                        self.log_institution(f"TAEC: Nodo {target.id} mejorado con conexión bidireccional a {new_connection.id}")
        
        # 6. Auto-optimización de grafo
        if node_count > 50 and random.random() < 0.3:
            self.log_institution("TAEC: Iniciando auto-optimización topológica...")
            
            # Identificar nodos de alta centralidad (cuellos de botella)
            G = nx.DiGraph()
            for node_id in self.graph.nodes:
                G.add_node(node_id)
            for node in self.graph.nodes.values():
                for target_id in node.connections_out:
                    G.add_edge(node.id, target_id)
            
            try:
                # Calcular betweenness centrality para identificar nodos clave
                if len(G) > 1:
                    centrality = nx.betweenness_centrality(G)
                    high_centrality_nodes = sorted(centrality.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)[:3]
                    
                    for node_id, score in high_centrality_nodes:
                        if score > 0.2 and node_id in self.graph.nodes:
                            # Fortalecer nodos clave
                            node = self.graph.get_node(node_id)
                            if node:
                                node.update_state(min(1.0, node.state + 0.05))
                                node.keywords.add("taec_optimized")
                                self.log_institution(f"TAEC: Optimizado nodo central {node_id} (score: {score:.3f})")
            except Exception as e:
                self.log_institution(f"TAEC: Error en optimización: {e}")
        
        # 7. Crear nuevos agentes si es necesario (simulación de auto-replicación)
        if hasattr(self.graph, 'agents') and len(self.graph.agents) > 0:
            existing_taec = sum(1 for a in self.graph.agents if isinstance(a, TAECEvolutionAgent))
            if existing_taec < 3 and random.random() < 0.1:
                # Nota: en implementación real, aquí se crearían agentes dinámicamente
                self.log_institution(f"TAEC: Auto-replicación sugerida (agentes TAEC actuales: {existing_taec})")
        
        # 8. Meta-análisis TAEC
        taec_evolved_count = len([n for n in self.graph.nodes.values() if 
                                "taec_evolved" in n.keywords or 
                                "taec_enhanced" in n.keywords or
                                "taec_optimized" in n.keywords])
        
        self.log_institution(f"TAEC Meta-análisis: {taec_evolved_count} nodos evolucionados por TAEC ({(taec_evolved_count/node_count*100):.1f}%)")
        self.log_institution(f"TAEC Info: {len(high_potential_nodes)} nodos de alto potencial, {len(low_connectivity_nodes)} nodos para recuperación")

# --- MINISTERIO DE DESARROLLO SOSTENIBLE COGNITIVO ---

# Instituto de Ecología de Red
class NodeBalancerAgent(InstitutionAgent):
    """
    Ajusta la población, densidad y conexiones para mantener el equilibrio de la red MSC.
    """
    def institution_action(self):
        self.log_institution("Balancing node population and connections...")
        # Si un nodo está sobredimensionado (muchas conexiones), se elimina la de menor utilidad.
        for node in self.graph.nodes.values():
            if len(node.connections_out) > 10:
                min_conn = min(node.connections_out.items(), key=lambda item: item[1])
                node.connections_out.pop(min_conn[0])
                self.log_institution(f"Removed low-utility connection from node {node.id} (over-connected).")
            # Si un nodo tiene pocas conexiones, se le añade una nueva conexión.
            elif len(node.connections_out) < 2:
                candidate = self.graph.get_random_node_biased()
                if candidate and candidate.id != node.id:
                    if self.graph.add_edge(node.id, candidate.id, utility=0.5):
                        self.log_institution(f"Added connection from node {node.id} to {candidate.id} (under-connected).")

class ClusterFormationAgent(InstitutionAgent):
    """
    Fomenta la creación de comunidades naturales conectando nodos con similitudes.
    """
    def institution_action(self):
        self.log_institution("Encouraging natural cluster formation...")
        for node in self.graph.nodes.values():
            # Encuentra nodos con keywords comunes.
            similar_nodes = [other for other in self.graph.nodes.values()
                             if other.id != node.id and node.keywords.intersection(other.keywords)]
            if similar_nodes:
                target = random.choice(similar_nodes)
                # Se agrega la conexión si aún no existe.
                if target.id not in node.connections_out:
                    if self.graph.add_edge(node.id, target.id, utility=0.6):
                        self.log_institution(f"Formed cluster link between node {node.id} and {target.id}.")

# Instituto de Diversidad Epistémica
class PerspectiveAgent(InstitutionAgent):
    """
    Introduce puntos de vista divergentes en la red para enriquecer el conocimiento.
    """
    def institution_action(self):
        self.log_institution("Injecting divergent perspectives...")
        node = self.graph.get_random_node_biased()
        if node:
            perspective = f"perspective_{random.randint(1,100)}"
            node.keywords.add(perspective)
            # Ajuste opcional: modificar ligeramente el estado para reflejar diversidad.
            node.update_state(max(0, node.state - 0.05))
            self.log_institution(f"Node {node.id} enriched with perspective: {perspective}.")

class CulturalMirrorAgent(InstitutionAgent):
    """
    Representa voces de distintas regiones o idiomas en la red MSC.
    """
    def institution_action(self):
        self.log_institution("Reflecting cultural diversity in the network...")
        culture = random.choice(["es", "en", "fr", "de", "zh"])
        content = f"Cultural input from region {culture} provided by {self.id}"
        new_node = self.graph.add_node(content=content, initial_state=0.5, keywords={culture, "culture"})
        self.log_institution(f"Created cultural mirror node {new_node.id} with culture {culture}.")

# --- MINISTERIO DE SÍNTESIS E INFERENCIA - Ministerio de Síntesis e Inferencia ---

# Instituto de Síntesis Predictiva
class SynthesizerAgent(InstitutionAgent):
    """
    Fusiona nodos para crear hipótesis emergentes a partir del conocimiento existente.
    """
    def institution_action(self):
        self.log_institution("Synthesizing new hypotheses by merging nodes (bridging synthesis)...")
        # Seleccionar node1 y buscar node2 que comparta pocas keywords
        node1 = self.graph.get_random_node_biased()
        candidates = [n for n in self.graph.nodes.values() if n.id != node1.id and len(node1.keywords.intersection(n.keywords)) < 1]
        node2 = random.choice(candidates) if candidates else self.graph.get_random_node_biased()
        if node1 and node2 and node1.id != node2.id:
            new_content = f"Synthesized Bridge: {node1.content} + {node2.content}"
            new_keywords = node1.keywords.union(node2.keywords).union({"synthesized", "bridge"})
            base_state = (node1.state + node2.state) / 2
            new_state = min(1.0, base_state + 0.05)
            new_node = self.graph.add_node(content=new_content, initial_state=new_state, keywords=new_keywords)
            self.graph.add_edge(new_node.id, node1.id, utility=0.7)
            self.graph.add_edge(new_node.id, node2.id, utility=0.7)
            self.log_institution(f"Created synthesized bridge node {new_node.id} merging nodes {node1.id} and {node2.id}. New state: {new_state:.2f}")
        else:
            self.log_institution("Not enough suitable nodes available for bridging synthesis.")

class PatternMinerAgent(InstitutionAgent):
    """
    Encuentra patrones y correlaciones inesperadas dentro del grafo.
    """
    def institution_action(self):
        self.log_institution("Mining patterns and unexpected correlations...")
        patterns_found = 0
        for node in self.graph.nodes.values():
            if "pattern" in node.keywords:
                continue
            similar_nodes = [n for n in self.graph.nodes.values() if n != node and node.keywords.intersection(n.keywords)]
            if len(similar_nodes) >= 2:
                avg_state = (node.state + sum(n.state for n in similar_nodes)) / (len(similar_nodes) + 1)
                node.keywords.add("pattern")
                for n in similar_nodes:
                    n.keywords.add("pattern")
                self.log_institution(f"Pattern detected in node {node.id} and {len(similar_nodes)} similar nodes. Avg state: {avg_state:.2f}")
                patterns_found += 1
        if patterns_found == 0:
            self.log_institution("No significant patterns detected this cycle.")

# Instituto de Tendencias y Prospección
class TrendDetectorAgent(InstitutionAgent):
    """
    Detecta temas crecientes o tendencias globales a partir de la evolución del grafo.
    """
    def institution_action(self):
        self.log_institution("Detecting emerging trends across the graph...")
        trend_count = {}
        for node in self.graph.nodes.values():
            for keyword in node.keywords:
                trend_count[keyword] = trend_count.get(keyword, 0) + node.state
        if trend_count:
            trending_keyword = max(trend_count, key=trend_count.get)
            self.log_institution(f"Detected trending topic: {trending_keyword} with score {trend_count[trending_keyword]:.2f}")
        else:
            self.log_institution("No trends detected.")

class ChronoAgent(InstitutionAgent):
    """
    Analiza la evolución de conceptos en el tiempo, creando líneas temporales de nodos relacionados.
    """
    def institution_action(self):
        self.log_institution("Analyzing temporal evolution of concepts...")
        if self.graph.nodes:
            sorted_nodes = sorted(self.graph.nodes.values(), key=lambda n: n.id)
            timeline = [(node.id, node.state) for node in sorted_nodes]
            self.log_institution(f"Chronological data collected: {timeline}")
        else:
            self.log_institution("Graph empty, no temporal data available.")

# --- MINISTERIO DE EVALUACIÓN Y APRENDIZAJE ---

# Instituto de Retroalimentación
class FeedbackLoopAgent(InstitutionAgent):
    """
    Compara las predicciones del sistema MSC con eventos reales
    y ajusta estrategias basándose en la diferencia.
    """
    def institution_action(self):
        self.log_institution("Running feedback loop: comparing predictions with real outcomes...")
        predicted = self.graph.get_random_node_biased()
        actual = self.graph.get_random_node_biased()
        if predicted and actual:
            diff = abs(predicted.state - actual.state)
            if diff > 0.2:
                self.log_institution(f"Significant deviation detected (diff = {diff:.2f}) between node {predicted.id} and {actual.id}.")
            else:
                self.log_institution(f"Feedback: predictions align (diff = {diff:.2f}).")
        else:
            self.log_institution("Insufficient nodes for feedback comparison.")

class MemoryAdjusterAgent(InstitutionAgent):
    """
    Reajusta estados y conexiones en el grafo en base a la retroalimentación evaluativa.
    """
    def institution_action(self):
        self.log_institution("Adjusting memory based on evaluative feedback...")
        for node in self.graph.nodes.values():
            if node.state > 0.8 and "verified" not in node.keywords:
                old_state = node.state
                node.update_state(node.state * 0.95)
                self.log_institution(f"Node {node.id} state adjusted from {old_state:.2f} to {node.state:.2f} (lacking verification).")
            elif node.state < 0.3:
                old_state = node.state
                node.update_state(node.state + 0.05)
                self.log_institution(f"Node {node.id} state increased from {old_state:.2f} to {node.state:.2f} for recovery.")

# Instituto de Medición Global
class GlobalMetricsAgent(InstitutionAgent):
    """
    Evalúa métricas agregadas del sistema (estado, densidad, clusters)
    y reporta los hallazgos globales.
    """
    def institution_action(self):
        self.log_institution("Evaluating global graph metrics...")
        metrics = self.graph.get_global_metrics()
        self.log_institution(f"Global Metrics: {metrics}")

class LogEvaluatorAgent(InstitutionAgent):
    """
    Analiza logs previos y eventos del sistema para generar reportes
    de evaluación longitudinal y aprendizaje a largo plazo.
    """
    def institution_action(self):
        self.log_institution("Analyzing system logs to generate longitudinal evaluation reports...")
        # Simulación: generar un reporte a partir del conteo de pasos (o cualquier otro indicador)
        report = f"Report generated at simulation step {self.graph.config.get('current_step', 'N/A')}."
        self.log_institution(f"Longitudinal Evaluation Report: {report}")

# --- GOBIERNO COGNITIVO DEL MSC - Ministerio de Conectividad Global ---

# Instituto de Infraestructura Informativa
class WebCrawlerAgent(InstitutionAgent):
    """
    Rastrea páginas web y repositorios para incorporar nuevos datos al grafo.
    """
    def institution_action(self):
        self.log_institution("Crawling web pages for new data...")
        # Implementación de ejemplo: simula la creación de un nodo con contenido extraído web.
        new_content = f"Web data fetched by {self.id}"
        new_node = self.graph.add_node(content=new_content, initial_state=0.5, keywords={"web", "data"})
        self.log_institution(f"Created node {new_node.id} with web-fetched content.")

class RSSListenerAgent(InstitutionAgent):
    """
    Escucha fuentes RSS para incorporar noticias o actualizaciones al grafo.
    """
    def institution_action(self):
        self.log_institution("Listening to RSS feeds for news updates...")
        # Simulación: si se detecta una 'noticia', se agrega un nodo.
        news = f"News item captured by {self.id}"
        new_node = self.graph.add_node(content=news, initial_state=0.4, keywords={"news"})
        self.log_institution(f"Added news node {new_node.id} from RSS feed.")

class APICollectorAgent(InstitutionAgent):
    """
    Accede a datos vivos de APIs reales y los integra en el grafo.
    """
    def institution_action(self):
        self.log_institution("Collecting live data from external APIs...")
        # Simulación: se crea un nodo representativo de datos externos.
        api_data = f"API data collected by {self.id}"
        new_node = self.graph.add_node(content=api_data, initial_state=0.6, keywords={"api", "external"})
        self.log_institution(f"Inserted API data node {new_node.id}.")

# Instituto de Enlace Humano-Máquina
class InterfaceAgent(InstitutionAgent):
    """
    Recoge preguntas e inputs humanos y los inserta en el grafo para ser procesados.
    """
    def institution_action(self):
        self.log_institution("Gathering human interface inputs...")
        # Simulación: se añade un nodo que representa una consulta o input humano.
        human_input = f"Human query received by {self.id}"
        new_node = self.graph.add_node(content=human_input, initial_state=0.4, keywords={"human", "query"})
        self.log_institution(f"Created interface node {new_node.id} with human input.")

class TranslatorAgent(InstitutionAgent):
    """
    Convierte lenguaje natural a un formato compatible con el MSC, integrando la información.
    """
    def institution_action(self):
        self.log_institution("Translating natural language to MSC format...")
        # Simulación: se toma un nodo de entrada y se crea uno traducido.
        original_node = self.graph.get_random_node_biased()
        if original_node:
            translated_content = f"Translated content of node {original_node.id} by {self.id}"
            new_node = self.graph.add_node(content=translated_content, initial_state=original_node.state, keywords={"translated"})
            self.log_institution(f"Created translated node {new_node.id} from node {original_node.id}.")

# --- MINISTERIO DEL ENTRENAMIENTO Y DATOS ---

# Instituto de Datos Reales y Sintéticos
class DatasetAgent(InstitutionAgent):
    """
    Introduce datasets reales o sintéticos (por ejemplo, de Kaggle, OpenAI, etc.)
    al grafo para enriquecer el conocimiento.
    """
    def institution_action(self):
        self.log_institution("Injecting dataset into the graph...")
        # Simulación: crear un nodo con datos identificados como datasets.
        dataset_info = f"Dataset provided by {self.id}"
        new_node = self.graph.add_node(content=dataset_info, initial_state=0.7, keywords={"dataset", "real", "synthetic"})
        self.log_institution(f"Dataset node {new_node.id} created with content: {dataset_info}")

class AutoLabelAgent(InstitutionAgent):
    """
    Etiqueta o anota información en el grafo utilizando LLMs (o lógica simulada).
    """
    def institution_action(self):
        self.log_institution("Auto-labeling information using LLM assistance...")
        # Simulación: seleccionar un nodo y "etiquetar" su contenido.
        target_node = self.graph.get_random_node_biased()
        if target_node:
            # Ejemplo: se agregan etiquetas adicionales a los keywords del nodo.
            target_node.keywords.update({"auto-labeled", "LLM"})
            self.log_institution(f"Node {target_node.id} auto-labeled with tags: auto-labeled, LLM")
        else:
            self.log_institution("No node found for auto-labeling.")

# Instituto de Validación y Contraste
class RealVsFictionAgent(InstitutionAgent):
    """
    Distingue fuentes verificadas versus especulativas, para marcar la confiabilidad de la fuente.
    """
    def institution_action(self):
        self.log_institution("Evaluating source authenticity (Real vs Fiction)...")
        # Simulación: elegir un nodo y asignar una bandera de confiabilidad.
        candidate = self.graph.get_random_node_biased()
        if candidate:
            # Se podría asignar un campo o ajustar el estado para reflejar mayor confiabilidad.
            if "verified" in candidate.keywords:
                self.log_institution(f"Node {candidate.id} already verified as real.")
            else:
                # Supongamos que un incremento en el estado indica mayor confiabilidad.
                candidate.update_state(candidate.state + 0.1)
                candidate.keywords.add("verified")
                self.log_institution(f"Node {candidate.id} marked as real (verified) and state increased.")
        else:
            self.log_institution("No candidate node found for source evaluation.")

class SourceVerifierAgent(InstitutionAgent):
    """
    Asigna reputación y confiabilidad a nodos de acuerdo a la fiabilidad de su fuente.
    """
    def institution_action(self):
        self.log_institution("Verifying source reliability for nodes...")
        # Simulación: recorrer nodos y ajustar reputación según ciertas palabras clave.
        for node in self.graph.nodes.values():
            # Si un nodo tiene la etiqueta "verified", aumentar reputación.
            if "verified" in node.keywords:
                # Incremento en la reputación para nodos verificados.
                self.log_institution(f"Node {node.id} verified. Increasing reputation.")
                node.update_state(min(1.0, node.state + 0.05))
            else:
                # Reducir reputación (simulado, por ejemplo) para nodos sin verificación.
                node.update_state(max(0.1, node.state - 0.03))
                self.log_institution(f"Node {node.id} not verified. Decreasing state slightly for caution.")

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

        # Inicializar repositorio de código con la nueva ruta configurada
        if 'taec_runtime_path' in config:
            # Usar la ruta específica para TAEC
            code_repo_path = config['taec_runtime_path']
        else:
            # Usar la ruta por defecto en el directorio actual
            save_path = config.get('save_state')
            if save_path:
                # Si hay una ruta de guardado especificada, usar su directorio
                code_repo_path = os.path.join(os.path.dirname(save_path), 'taec_code')
            else:
                # Si no hay ninguna ruta, usar 'generated_code/taec_code' en el directorio actual
                code_repo_path = os.path.join('generated_code', 'taec_code')

        # Asegurarse de que el directorio exista
        if not os.path.exists(code_repo_path):
            os.makedirs(code_repo_path, exist_ok=True)

        self.graph.code_repository = CodeRepository(save_dir=code_repo_path)
        logging.info(f"Initialized code repository at {code_repo_path}")

        self.agents = []
        self.is_running = False
        self.simulation_thread = None
        self.step_count = 0
        self.lock = threading.Lock()
        self.start_time = time.time()  # Añadir esta línea

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

        # Inicialización de nodos semilla si el grafo está vacío
        if len(self.graph.nodes) == 0:
            logging.info("El grafo está vacío. Inicializando nodos semilla para TAEC...")
            
            # Crear nodos base con diferentes dominios de conocimiento
            domains = [
                {"name": "IA General", "keywords": {"ai", "machine_learning", "intelligence"}},
                {"name": "Procesamiento de Lenguaje Natural", "keywords": {"nlp", "language", "text"}}, 
                {"name": "Visión por Computadora", "keywords": {"vision", "image", "recognition"}},
                {"name": "Redes Neuronales", "keywords": {"neural_networks", "deep_learning", "backprop"}},
                {"name": "Grafos de Conocimiento", "keywords": {"knowledge_graphs", "semantic", "ontology"}},
                {"name": "Auto-evolución", "keywords": {"self_evolution", "taec", "metacognition"}},
                {"name": "Aprendizaje por Refuerzo", "keywords": {"reinforcement_learning", "rl", "rewards"}},
                {"name": "Síntesis Cognitiva", "keywords": {"cognitive_synthesis", "integration", "emergence"}}
            ]
            
            # Crear los nodos semilla
            nodes = []
            for domain in domains:
                node = self.graph.add_node(
                    content=f"TAEC Seed: {domain['name']}", 
                    initial_state=0.5, 
                    keywords=domain['keywords']
                )
                nodes.append(node)
                logging.info(f"Creado nodo semilla {node.id}: {domain['name']}")
            
            # Crear algunas conexiones iniciales entre nodos relacionados
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if random.random() < 0.4:  # 40% de probabilidad de conexión
                        utility = random.uniform(0.4, 0.8)
                        if self.graph.add_edge(nodes[i].id, nodes[j].id, utility=utility):
                            logging.info(f"Creada conexión entre nodos {nodes[i].id} y {nodes[j].id} con utilidad {utility:.2f}")
            
            # Actualizar embeddings iniciales
            self.graph.update_embeddings()
            logging.info(f"Grafo inicializado con {len(self.graph.nodes)} nodos semilla y {sum(len(n.connections_out) for n in self.graph.nodes.values())} conexiones")

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

        # Agregar la instanciación de agentes del Ministerio de Desarrollo Sostenible Cognitivo:
        num_node_balancer = self.config.get('num_node_balancer_agents', 1)
        for i in range(num_node_balancer):
            self.agents.append(NodeBalancerAgent(f"NB{i}", self.graph, self.config))

        num_cluster_formation = self.config.get('num_cluster_formation_agents', 1)
        for i in range(num_cluster_formation):
            self.agents.append(ClusterFormationAgent(f"CF{i}", self.graph, self.config))

        num_perspective = self.config.get('num_perspective_agents', 1)
        for i in range(num_perspective):
            self.agents.append(PerspectiveAgent(f"PA{i}", self.graph, self.config))

        num_cultural_mirror = self.config.get('num_cultural_mirror_agents', 1)
        for i in range(num_cultural_mirror):
            self.agents.append(CulturalMirrorAgent(f"CM{i}", self.graph, self.config))

        # Agregar la instanciación de agentes del Ministerio de Síntesis e Inferencia:
        num_synthesizer = self.config.get('num_synthesizer_agents', 3)  # Aumentado de 1 a 3 por defecto
        for i in range(num_synthesizer):
            self.agents.append(SynthesizerAgent(f"SYN{i}", self.graph, self.config))
        # También se pueden agregar agentes extras si se desea:
        extra_synthesizers = self.config.get('num_extra_synthesizer_agents', 1)
        for i in range(extra_synthesizers):
            self.agents.append(SynthesizerAgent(f"EXSYN{i}", self.graph, self.config))

        num_pattern_miner = self.config.get('num_pattern_miner_agents', 1)
        for i in range(num_pattern_miner):
            self.agents.append(PatternMinerAgent(f"PM{i}", self.graph, self.config))

        num_trend_detector = self.config.get('num_trend_detector_agents', 1)
        for i in range(num_trend_detector):
            self.agents.append(TrendDetectorAgent(f"TD{i}", self.graph, self.config))

        num_chrono = self.config.get('num_chrono_agents', 1)
        for i in range(num_chrono):
            self.agents.append(ChronoAgent(f"CHRO{i}", self.graph, self.config))

        # Agregar la instanciación de agentes del Ministerio de Evaluación y Aprendizaje:
        num_feedback = self.config.get('num_feedback_agents', 1)
        for i in range(num_feedback):
            self.agents.append(FeedbackLoopAgent(f"FB{i}", self.graph, self.config))

        num_memory_adjuster = self.config.get('num_memory_adjuster_agents', 1)
        for i in range(num_memory_adjuster):
            self.agents.append(MemoryAdjusterAgent(f"MA{i}", self.graph, self.config))

        num_global_metrics = self.config.get('num_global_metrics_agents', 1)
        for i in range(num_global_metrics):
            self.agents.append(GlobalMetricsAgent(f"GM{i}", self.graph, self.config))

        num_log_evaluator = self.config.get('num_log_evaluator_agents', 1)
        for i in range(num_log_evaluator):
            self.agents.append(LogEvaluatorAgent(f"LE{i}", self.graph, self.config))

        # Agregar la instanciación de agentes del Ministerio de Conectividad Global:
        num_webcrawler = self.config.get('num_webcrawler_agents', 1)
        for i in range(num_webcrawler):
            self.agents.append(WebCrawlerAgent(f"WC{i}", self.graph, self.config))

        num_rsslistener = self.config.get('num_rsslistener_agents', 1)
        for i in range(num_rsslistener):
            self.agents.append(RSSListenerAgent(f"RSS{i}", self.graph, self.config))

        num_apicollector = self.config.get('num_apicollector_agents', 1)
        for i in range(num_apicollector):
            self.agents.append(APICollectorAgent(f"API{i}", self.graph, self.config))

        num_interface = self.config.get('num_interface_agents', 1)
        for i in range(num_interface):
            self.agents.append(InterfaceAgent(f"INT{i}", self.graph, self.config))

        num_translator = self.config.get('num_translator_agents', 1)
        for i in range(num_translator):
            self.agents.append(TranslatorAgent(f"TR{i}", self.graph, self.config))

        # Agregar la instanciación de agentes del Ministerio del Entrenamiento y Datos:
        num_dataset = self.config.get('num_dataset_agents', 1)
        for i in range(num_dataset):
            self.agents.append(DatasetAgent(f"DATA{i}", self.graph, self.config))

        num_autolabel = self.config.get('num_autolabel_agents', 1)
        for i in range(num_autolabel):
            self.agents.append(AutoLabelAgent(f"AUTOLABEL{i}", self.graph, self.config))

        num_realvsfiction = self.config.get('num_realvsfiction_agents', 1)
        for i in range(num_realvsfiction):
            self.agents.append(RealVsFictionAgent(f"RVF{i}", self.graph, self.config))

        num_sourceverifier = self.config.get('num_sourceverifier_agents', 1)
        for i in range(num_sourceverifier):
            self.agents.append(SourceVerifierAgent(f"SV{i}", self.graph, self.config))

        # Agregar la instanciación de agentes TAEC:
        num_taec_agents = self.config.get('num_taec_agents', 2)  # Valor por defecto de 2
        for i in range(num_taec_agents):
            self.agents.append(TAECEvolutionAgent(f"TAEC{i}", self.graph, self.config))

        logging.info(f"Created agents: "
                     f"Proposers={config.get('num_proposers',0)}, Evaluators={config.get('num_evaluators',0)}, "
                     f"Combiners={config.get('num_combiners',0)}, Bridging={num_bridging_agents}, "
                     f"KnowledgeFetchers={num_knowledge_fetchers}, HorizonScanners={num_horizon_scanners}, "
                     f"EpistemicValidators={num_epistemic_validators}, TechnogenesisAgents={num_technogenesis_agents}, "
                     f"Inspectors={num_inspectors}, Police={num_police}, Coordinators={num_coordinators}, "
                     f"RepairAgents={num_repair}, Masters={num_master}, Students={num_students}, Scientists={num_scientists}, "
                     f"StorageAgents={num_storage}, BankAgents={num_bank}, MerchantAgents={num_merchant}, MinerAgents={num_miner}, "
                     f"PopulationRegulators={num_population_regulators}, Seeders={num_seeders}, ClusterBalancers={num_cluster_balancers}, Mediators={num_mediators}, MigrationAgents={num_migration_agents}, "
                     f"NodeBalancers={num_node_balancer}, ClusterFormers={num_cluster_formation}, Perspectives={num_perspective}, CulturalMirrors={num_cultural_mirror}, "
                     f"Synthesizers={num_synthesizer}, PatternMiners={num_pattern_miner}, TrendDetectors={num_trend_detector}, ChronoAgents={num_chrono}, "
                     f"FeedbackLoops={num_feedback}, MemoryAdjusters={num_memory_adjuster}, GlobalMetrics={num_global_metrics}, LogEvaluators={num_log_evaluator}, "
                     f"WebCrawlers={num_webcrawler}, RSSListeners={num_rsslistener}, APICollectors={num_apicollector}, Interfaces={num_interface}, Translators={num_translator}, "
                     f"DatasetAgents={num_dataset}, AutoLabelAgents={num_autolabel}, RealVsFictionAgents={num_realvsfiction}, SourceVerifierAgents={num_sourceverifier}, "
                     f"TAECEvolutionAgents={num_taec_agents}")  # Añadir esta línea

    def _simulation_loop(self):
        step_delay = self.config.get('step_delay', 0.1)
        gnn_update_frequency = self.config.get('gnn_update_frequency', 10)
        summary_frequency = self.config.get('summary_frequency', 50)
        max_steps = self.config.get('simulation_steps', None)
        is_api_mode = self.config.get('run_api', False)
        run_continuously = is_api_mode or (max_steps is None)
        gnn_training_frequency = self.config.get('gnn_training_frequency', 50)
        gnn_training_epochs = self.config.get('gnn_training_epochs', 5)

        # Use progress bar for fixed-step simulations
        if not run_continuously and max_steps is not None:
            progress_bar = tqdm(total=max_steps, desc="Simulation Progress")
        else:
            progress_bar = None

        # Use parallel processing for agent actions when appropriate
        def _process_agent_batch(agent_batch):
            results = []
            for agent_data in agent_batch:
                agent = agent_data['agent']
                cost = agent_data['cost']
                agent.omega -= cost
                try:
                    agent.act()
                except Exception as e:
                    agent_type = type(agent).__name__
                    logging.error(f"Error in {agent_type} {agent.id}: {e}")
                    # Optional recovery of agent state
                    if self.config.get('auto_recover_agents', True):
                        agent.omega = max(agent.omega, self.config.get('min_omega_recovery', 10.0))
                        logging.info(f"Auto-recovered agent {agent.id} with omega={agent.omega}")
                results.append((agent.id, cost, agent.omega))
            return results

        while self.is_running:
            self.step_count += 1
            if progress_bar:
                progress_bar.update(1)
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
                    # Actualizar contador de pasos en el adaptador si existe
                    if hasattr(self, 'sim_adapter') and self.sim_adapter:
                        self.sim_adapter.current_step = self.step_count
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
                if len(runnable_agents) >= self.config.get('parallel_batch_threshold', 5):
                    # Parallel processing for larger agent groups
                    batch_size = min(len(runnable_agents), self.config.get('max_parallel_batch', 10))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
                        batches = [runnable_agents[i:i+batch_size] for i in range(0, len(runnable_agents), batch_size)]
                        for results in executor.map(_process_agent_batch, batches):
                            for agent_id, cost, omega in results:
                                logging.debug(f"Agent {agent_id} acted (Cost: {cost:.2f}, Omega: {omega:.2f})")
                else:
                    # Original sequential processing for small numbers of agents
                    if runnable_agents:
                        chosen = random.choice(runnable_agents)
                        agent = chosen['agent']
                        cost = chosen['cost']
                        omega_before = agent.omega
                        agent.omega -= cost
                        try:
                            agent.act()
                        except Exception as e:
                            agent_type = type(agent).__name__
                            logging.error(f"Error in {agent_type} {agent.id}: {e}")
                            # Optional recovery of agent state
                            if self.config.get('auto_recover_agents', True):
                                agent.omega = max(agent.omega, self.config.get('min_omega_recovery', 10.0))
                                logging.info(f"Auto-recovered agent {agent.id} with omega={agent.omega}")
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
                if self.config.get('run_api', False) and self.step_count % self.config.get('websocket_update_frequency', 5) == 0:
                    try:
                        status = self.get_status()
                        socketio.emit('simulation_update', status)
                    except Exception as e:
                        logging.error(f"Error emitting websocket update: {e}")

                # Automatic checkpointing
                checkpoint_frequency = self.config.get('checkpoint_frequency', 1000)
                checkpoint_base_path = self.config.get('checkpoint_base_path')
                
                if checkpoint_base_path and checkpoint_frequency > 0 and self.step_count % checkpoint_frequency == 0:
                    checkpoint_path = f"{checkpoint_base_path}_step{self.step_count}"
                    logging.info(f"Creating checkpoint at step {self.step_count}: {checkpoint_path}")
                    try:
                        self.graph.save_state(checkpoint_path)
                        
                        # Optionally, clean up old checkpoints
                        max_checkpoints = self.config.get('max_checkpoints', 5)
                        if max_checkpoints > 0:
                            checkpoint_dir = os.path.dirname(checkpoint_base_path)
                            checkpoint_prefix = os.path.basename(checkpoint_base_path)
                            if os.path.exists(checkpoint_dir):
                                checkpoints = [f for f in os.listdir(checkpoint_dir) 
                                              if f.startswith(checkpoint_prefix) and f.endswith(".graphml")]
                                if len(checkpoints) > max_checkpoints:
                                    checkpoints.sort()
                                    for old_checkpoint in checkpoints[:-max_checkpoints]:
                                        old_path = os.path.join(checkpoint_dir, old_checkpoint)
                                        os.remove(old_path)
                                        logging.debug(f"Removed old checkpoint: {old_path}")
                    except Exception as e:
                        logging.error(f"Error creating checkpoint: {e}")

            time.sleep(step_delay)
        if progress_bar:
            progress_bar.close()
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
        """Devuelve el estado actual de la simulación para el visualizador"""
        with self.lock:
            try:
                # Calcular estado promedio de los nodos
                if hasattr(self, 'graph') and hasattr(self.graph, 'nodes') and self.graph.nodes:
                    avg_state = sum(node.state for node in self.graph.nodes.values()) / len(self.graph.nodes)
                    node_count = len(self.graph.nodes)
                    edge_count = sum(len(node.connections_out) for node in self.graph.nodes.values())
                else:
                    avg_state = 0
                    node_count = 0
                    edge_count = 0
                    
                # Calcular omega promedio de los agentes
                if hasattr(self, 'agents') and self.agents:
                    avg_omega = sum(agent.omega for agent in self.agents if hasattr(agent, 'omega')) / len(self.agents)
                    agent_count = len(self.agents)
                    
                    # Recopilar información de los agentes para el visualizador
                    agents_data = []
                    for agent in self.agents:
                        agents_data.append({
                            'id': agent.id,
                            'type': agent.__class__.__name__,
                            'status': 'Active' if agent.omega > 0 else 'Depleted',
                            'lastActivity': time.strftime('%H:%M:%S')
                        })
                else:
                    avg_omega = 0
                    agent_count = 0
                    agents_data = []
                    
                return {
                    "isRunning": self.is_running,
                    "currentStep": self.step_count,
                    "nodeCount": node_count,
                    "edgeCount": edge_count,
                    "agentCount": agent_count,
                    "averageState": round(avg_state, 3),
                    "averageOmega": round(avg_omega, 3),
                    "runtime": round(time.time() - self.start_time, 1) if hasattr(self, 'start_time') else 0,
                    "agents": agents_data
                }
            except Exception as e:
                logging.error(f"Error al generar estado de simulación: {e}")
                return {"error": str(e)}

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
                        logging.error(f"Error processing node {node_id}: {ex}")
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
                logging.error(f"Unexpected error in get_graph_elements_for_cytoscape: {e}")
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
        'num_node_balancers': 1,
        'num_cluster_formers': 1,
        'num_perspective_agents': 1,
        'num_cultural_mirrors': 1,
        'num_synthesizer_agents': 1,
        'num_pattern_miner_agents': 1,
        'num_trend_detector_agents': 1,
        'num_chrono_agents': 1,
        'num_feedback_loop_agents': 1,
        'num_memory_adjuster_agents': 1,
        'num_global_metrics_agents': 1,
        'num_log_evaluator_agents': 1,
        'num_webcrawler_agents': 1,
        'num_rsslistener_agents': 1,
        'num_apicollector_agents': 1,
        'num_interface_agents': 1,
        'num_translator_agents': 1,
        'num_dataset_agents': 1,
        'num_autolabel_agents': 1,
        'num_realvsfiction_agents': 1,
        'num_sourceverifier_agents': 1,
        'parallel_batch_threshold': 5,
        'max_workers': 4,
        'max_parallel_batch': 10,
        'checkpoint_frequency': 1000,
        'checkpoint_base_path': None,
        'max_checkpoints': 5,
        'max_nodes': 5000,
        'auto_recover_agents': False,
        'min_omega_recovery': 10.0,
        'websocket_update_frequency': 5,
        'num_taec_agents': 2,  # Default value for TAEC agents
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
    
    # Add config version tracking
    config['version'] = '1.1.0'
    
    # Add runtime timestamp 
    config['runtime_timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    
    # Auto-create output directories if needed
    for path_key in ['save_state', 'visualization_output_path', 'metrics_log_path']:
        if config.get(path_key):
            try:
                dir_path = os.path.dirname(config[path_key])
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    logging.info(f"Created directory: {dir_path}")
            except Exception as e:
                logging.warning(f"Could not create directory for {path_key}: {e}")
    
    return config

simulation_runner = None
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

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

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected to websocket')
    if simulation_runner:
        socketio.emit('simulation_update', simulation_runner.get_status())

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected from websocket')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MSC Simulation with GNN and optional API.")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--simulation_steps', type=int, help='Number of simulation steps to run')
    parser.add_argument('--num_proposers', type=int, help='Number of proposer agents')
    parser.add_argument('--num_evaluators', type=int, help='Number of evaluator agents')
    parser.add_argument('--num_combiners', type=int, help='Number of combiner agents')
    parser.add_argument('--step_delay', type=float, help='Delay between simulation steps in seconds')
    parser.add_argument('--save_state', type=str, help='Path to save the simulation state')
    parser.add_argument('--load_state', type=str, help='Path to load the simulation state from')
    parser.add_argument('--run_api', action='store_true', help='Run the Flask API server')
    parser.add_argument('--visualization_output_path', type=str, help='Path to output visualization')
    parser.add_argument('--metrics_log_path', type=str, help='Path to log metrics CSV')
    parser.add_argument('--checkpoint_frequency', type=int, help='Save checkpoints every N steps')
    parser.add_argument('--checkpoint_base_path', type=str, help='Base path for checkpoint files')
    parser.add_argument('--max_checkpoints', type=int, help='Maximum number of checkpoint files to keep')
    
    # Añadir argumentos para el nuevo visualizador
    parser.add_argument('--viewer_port', type=int, default=8080, 
                        help='Port for the MSC Viewer server')
    parser.add_argument('--run_viewer', action='store_true', 
                        help='Run the MSC Viewer visualization server')
    
    # Añadir argumento para el directorio de código TAEC
    parser.add_argument('--taec_runtime_path', type=str,
                        help='Directory path to store TAEC generated code')
    
    args = parser.parse_args()
    final_config = load_config(args)
    final_config['run_api'] = args.run_api
    final_config['run_viewer'] = args.run_viewer if hasattr(args, 'run_viewer') else False
    final_config['viewer_port'] = args.viewer_port if hasattr(args, 'viewer_port') else 8080
    
    # Si se proporciona taec_runtime_path, úsalo para el repositorio de código
    if hasattr(args, 'taec_runtime_path') and args.taec_runtime_path:
        final_config['taec_runtime_path'] = args.taec_runtime_path
    
    simulation_runner = SimulationRunner(final_config)

    # Iniciar la simulación
    logging.info("Iniciando simulación...")
    simulation_runner.start()

    # Integrar el nuevo visualizador MSC Viewer
    msc_viewer_server = None
    if final_config.get('run_viewer', False) and MSC_VIEWER_AVAILABLE:
        try:
            logging.info("Initializing MSC Viewer...")
            # Crear el adaptador de simulación con acceso directo al runner
            sim_adapter = SimulationAdapter(simulation_runner)
            # Guardar la referencia al adaptador en el simulador para actualizaciones
            simulation_runner.sim_adapter = sim_adapter
            
            # Iniciar el adaptador
            sim_adapter.current_step = simulation_runner.step_count
            sim_adapter.start()
            
            # Iniciar el servidor de visualización
            viewer_port = final_config.get('viewer_port', 8080)
            msc_viewer_server = MSCViewerServer(host='localhost', port=viewer_port)
            msc_viewer_server.start(simulation_runner=sim_adapter)
            
            logging.info(f"MSC Viewer started at http://localhost:{viewer_port}")
        except Exception as e:
            logging.error(f"Failed to start MSC Viewer: {e}")
            msc_viewer_server = None
    elif final_config.get('run_viewer', False):
        logging.error("MSC Viewer requested but module not available")

    time.sleep(2)
    if final_config.get('run_api', False):
        logging.info("Starting Flask+SocketIO server on http://127.0.0.1:5000")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        try:
            socketio.run(app, host='127.0.0.1', port=5000, debug=False)
        except KeyboardInterrupt:
            logging.info("Ctrl+C detected. Stopping Flask API and simulation...")
        finally:
            if msc_viewer_server:
                msc_viewer_server.stop()
            simulation_runner.stop()
    else:
        logging.info("Simulation running in background thread. Open http://localhost:8080 for visualization.")
        logging.info("Press Ctrl+C to stop.")
        try:
            while simulation_runner.is_running:
                time.sleep(0.5)
                # Si el visualizador está activo, actualizar el adaptador
                if msc_viewer_server and hasattr(simulation_runner, 'sim_adapter'):
                    simulation_runner.sim_adapter.current_step = simulation_runner.step_count
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt detected. Stopping simulation...")
            if msc_viewer_server:
                msc_viewer_server.stop()
            simulation_runner.stop()
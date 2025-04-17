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
import statistics         # <-- AÑADIR para métricas de estado
import csv                # <-- AÑADIR para escribir archivos CSV
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

    # --- NUEVO: Método Decode para Link Prediction ---
    def decode(self, z, edge_label_index):
        """Predice scores de enlace usando producto punto de embeddings."""
        # z: embeddings [N, emb_dim]
        # edge_label_index: pares de índices de nodo [2, num_pairs]
        emb_src = z[edge_label_index[0]]
        emb_dst = z[edge_label_index[1]]
        return (emb_src * emb_dst).sum(dim=-1)
    # --- Fin Decode ---

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

        # --- NUEVO: Optimizador y Loss para GNN ---
        self.gnn_optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=config.get('gnn_learning_rate', 0.01))
        self.gnn_loss_fn = torch.nn.BCEWithLogitsLoss()  # Pérdida para predicción binaria
        # --- Fin Optimizador/Loss ---

    def _prepare_pyg_data(self):
        if not self.nodes:
            return None, None
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        num_nodes = len(self.nodes)
        num_node_features = self.num_node_features
        node_features = torch.zeros((num_nodes, num_node_features), dtype=torch.float)

        # --- NUEVO: Calcular métricas estructurales adicionales, ej. PageRank ---
        # Construir grafo temporal
        temp_G = nx.DiGraph()
        for node_id in self.nodes.keys():
            temp_G.add_node(node_id)
        for node in self.nodes.values():
            for target_id in node.connections_out.keys():
                if target_id in self.nodes:
                    temp_G.add_edge(node.id, target_id)
        # Calcular PageRank (se pueden incluir otras métricas)
        pagerank = nx.pagerank(temp_G) if temp_G.number_of_nodes() > 0 else {nid: 0.0 for nid in self.nodes.keys()}
        # --- Fin métricas adicionales ---

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
            # Características base originales
            node_features[i, 0] = node.state
            node_features[i, 1] = float(len(node.connections_in))
            node_features[i, 2] = float(len(node.connections_out))
            node_features[i, 3] = float(len(node.keywords))
            # NUEVA característica: PageRank (normalizado [0,1] por ser probabilidad)
            node_features[i, 4] = pagerank.get(node_id, 0.0)
            # Si hay embeddings de texto, se añaden a continuación
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

    def train_gnn(self, num_epochs=10):
        """Entrena la GNN usando predicción de enlaces con validaciones y manejo de dispositivo."""
        # Verifica que existan suficientes nodos
        if not self.nodes or len(self.nodes) < 2:
            logging.warning("GNN Training: Not enough nodes to train.")
            return

        node_features, edge_index = self._prepare_pyg_data()
        if node_features is None or edge_index is None or edge_index.numel() == 0:
            logging.warning("GNN Training: Not enough data (nodes/edges) to train.")
            return

        num_nodes = node_features.shape[0]
        # Asigna dispositivo (GPU si está disponible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(device)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        self.gnn_model.train()
        total_loss = 0.0

        try:
            for epoch in range(num_epochs):
                self.gnn_optimizer.zero_grad()
                
                # Paso forward: obtener embeddings
                z = self.gnn_model(node_features, edge_index)
                
                # Aristas positivas: usar todas las aristas existentes
                pos_edge_label_index = edge_index
                
                # Muestreo negativo: se generan tantos ejemplos negativos como positivos
                neg_edge_label_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=pos_edge_label_index.size(1),
                    method='sparse'
                )
                if neg_edge_label_index.numel() == 0:
                    logging.warning(f"Epoch {epoch+1}: No negative samples generated, skipping this epoch.")
                    continue

                # Concatenar índices de aristas positivas y negativas
                edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=-1)
                # Crear etiquetas: 1 para positivas, 0 para negativas
                pos_labels = torch.ones(pos_edge_label_index.size(1), device=device)
                neg_labels = torch.zeros(neg_edge_label_index.size(1), device=device)
                edge_labels = torch.cat([pos_labels, neg_labels], dim=0)
                
                # Decodificar las predicciones para obtener scores de enlace
                out_scores = self.gnn_model.decode(z, edge_label_index)
                
                # Validación de dimensiones
                if out_scores.size(0) != edge_labels.size(0):
                    logging.error(
                        f"Epoch {epoch+1}: Mismatch in output scores and labels: "
                        f"{out_scores.size(0)} vs {edge_labels.size(0)}"
                    )
                    continue

                # Calcular la pérdida
                loss = self.gnn_loss_fn(out_scores, edge_labels)
                loss.backward()
                self.gnn_optimizer.step()
                
                total_loss += loss.item()
                logging.debug(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}")

        except Exception as e:
            logging.error(
                f"Error during GNN training epoch {epoch+1 if 'epoch' in locals() else 'N/A'}: {e}"
            )
            import traceback
            logging.error(traceback.format_exc())
        finally:
            self.gnn_model.eval()
            avg_loss = total_loss / num_epochs if num_epochs > 0 else float('inf')
            logging.info(f"GNN Training Finished: {num_epochs} epochs, Avg Loss = {avg_loss:.4f}")

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

    # --- NUEVO MÉTODO PARA MÉTRICAS GLOBALES ---
    def log_global_metrics(self, log_level=logging.INFO, current_step="N/A"):  # <-- Añadir current_step
        """Calcula y loguea métricas globales del grafo."""
        logging.log(log_level, f"--- Global Metrics (Step: {current_step}) ---")  # <-- Usar current_step

        num_nodes = len(self.nodes)
        if num_nodes < 2:  # No se pueden calcular muchas métricas con 0 o 1 nodo
            logging.log(log_level, "--- Global Metrics: Skipped (Graph too small) ---")
            return

        # Crear grafo NetworkX temporal para cálculos
        G = nx.DiGraph()
        node_states = []
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            G.add_node(node_id_str)
            node_states.append(node.state)
        num_edges = 0
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            for target_id, utility in node.connections_out.items():
                target_id_str = str(target_id)
                if node_id_str in G and target_id_str in G:
                    G.add_edge(node_id_str, target_id_str)
                    num_edges += 1

        # Calcular Métricas Estructurales
        try:
            density = nx.density(G)
            logging.log(log_level, f"  Density: {density:.4f}")
        except Exception as e:
            logging.warning(f"  Could not calculate density: {e}")

        try:
            if num_nodes > 0:
                avg_clustering = nx.average_clustering(G.to_undirected(as_view=True))
                logging.log(log_level, f"  Avg. Clustering Coefficient: {avg_clustering:.4f}")
            else:
                logging.log(log_level, "  Avg. Clustering Coefficient: N/A (no nodes)")
        except Exception as e:
            logging.warning(f"  Could not calculate avg clustering: {e}")

        try:
            num_components = nx.number_weakly_connected_components(G)
            logging.log(log_level, f"  Weakly Connected Components: {num_components}")
        except Exception as e:
            logging.warning(f"  Could not calculate connected components: {e}")

        # Calcular Métricas de Estado
        if node_states:
            mean_state = statistics.mean(node_states)
            median_state = statistics.median(node_states)
            stdev_state = statistics.stdev(node_states) if len(node_states) > 1 else 0.0
            min_state = min(node_states)
            max_state = max(node_states)
            logging.log(log_level, f"  Node State (sj) Stats: Mean={mean_state:.4f}, Median={median_state:.4f}, StdDev={stdev_state:.4f}, Min={min_state:.4f}, Max={max_state:.4f}")
        else:
            logging.log(log_level, "  Node State (sj) Stats: N/A (no nodes)")
    # --- Fin del método log_global_metrics ---

    # --- MÉTODO PARA OBTENER MÉTRICAS ---
    def get_global_metrics(self):
        """Calcula y devuelve métricas globales del grafo como diccionario."""
        metrics = {fieldname: None for fieldname in [
            "Nodes", "Edges", "Density", "AvgClustering", "Components",
            "MeanState", "MedianState", "StdDevState", "MinState", "MaxState"
        ]}
        num_nodes = len(self.nodes)
        metrics["Nodes"] = num_nodes

        if num_nodes == 0:
            return metrics

        # Métricas de estado
        node_states = [n.state for n in self.nodes.values()]
        if node_states:
            metrics["MeanState"] = statistics.mean(node_states)
            metrics["MedianState"] = statistics.median(node_states)
            metrics["MinState"] = min(node_states)
            metrics["MaxState"] = max(node_states)
            metrics["StdDevState"] = statistics.stdev(node_states) if num_nodes > 1 else 0.0

        # Métricas estructurales con NetworkX
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
                if num_edges > 0 and num_nodes > 1:
                    metrics["AvgClustering"] = nx.average_clustering(G)
                else:
                    metrics["AvgClustering"] = 0.0
            except Exception as cluster_e:
                logging.warning(f"Could not calculate avg clustering: {cluster_e}")
                metrics["AvgClustering"] = None

        # Redondear valores flotantes
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics[key] = round(value, 4)

        return metrics
    # --- Fin get_global_metrics ---

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

    def get_graph_elements_for_cytoscape(self):
        """Prepara los nodos y aristas en formato para Dash Cytoscape."""
        elements = []
        with self.lock:
            try:
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=0, vmax=1)
                for node_id, node in self.nodes.items():
                    try:
                        node_color = cmap(norm(node.state))
                        hex_color = '#%02x%02x%02x' % tuple([int(c * 255) for c in node_color[:3]])
                    except Exception as ex:
                        app.logger.error("Error procesando nodo %s: %s", node_id, ex)
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
                        if source_id in self.nodes and target_id in self.nodes:
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

    # --- MÉTODO visualize_graph ACTUALIZADO PARA GUARDAR ---
    def visualize_graph(self, config):
        """Genera y guarda una visualización del grafo si se especifica en config."""
        output_path = config.get('visualization_output_path', None)
        # Solo proceder si hay una ruta de salida configurada
        if not output_path:
            return

        if not self.nodes:
            logging.warning("Cannot visualize empty graph. Image not saved.")
            return

        logging.info(f"Generating graph visualization to save at: {output_path}")
        G = nx.DiGraph()
        node_labels = {}
        node_sizes = []
        node_colors = []
        # Procesar nodos
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            G.add_node(node_id_str)
            node_labels[node_id_str] = f"{node_id}\nS={node.state:.2f}"
            node_sizes.append(100 + node.state * 1500)
            node_colors.append(node.state)
        edge_list = []
        edge_weights = []
        edge_colors = []
        # Procesar aristas
        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)
            for target_id, utility in node.connections_out.items():
                target_id_str = str(target_id)
                if node_id_str in G and target_id_str in G:
                    edge_list.append((node_id_str, target_id_str))
                    edge_weights.append(1 + abs(utility) * 4)
                    edge_colors.append(utility)

        if not G.nodes:
            logging.warning("No nodes to visualize after processing. Image not saved.")
            return

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_title("MSC Graph Visualization")
        ax.axis('off')

        # Calcular layout
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
            cbar_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.5, aspect=20)
            cbar_nodes.set_label('Node State (sj)')

        if edge_colors:
            norm_edges = plt.Normalize(vmin=min(edge_colors or [-1]), vmax=max(edge_colors or [1]))
            sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm_edges)
            sm_edges.set_array([])
            cbar_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.5, aspect=20)
            cbar_edges.set_label('Edge Utility (uij)')

        try:
            plt.savefig(config.get('visualization_output_path'), bbox_inches='tight', dpi=150)
            logging.info(f"Graph visualization saved successfully to {output_path}")
        except Exception as e:
            logging.error(f"Error saving visualization to {output_path}: {e}")
        finally:
            plt.close(fig)
    # --- Fin método visualize_graph ---

class Synthesizer:
    """Clase base para los agentes."""
    def __init__(self, agent_id, graph, config):
        self.id = agent_id
        self.graph = graph
        self.config = config
        self.omega = config.get('initial_omega', 100.0)
        # --- AÑADIDO: Inicializar Reputación ---
        self.reputation = config.get('initial_reputation', 1.0)
        # --- Fin Añadido ---

    def act(self):
        raise NotImplementedError

# Clase base para Agentes Institucionales
class InstitutionAgent(Synthesizer, ABC):
    """
    Base para los agentes que actúan bajo instituciones estratégicas.
    Provee mensajes estándar y facilidades para la interacción institucional.
    """
    def act(self):
        # Por defecto, se delega a la función institucional específica.
        self.institution_action()

    @abstractmethod
    def institution_action(self):
        """Implementar la acción institucional específica."""
        pass

    def log_institution(self, message):
        logging.info(f"[{self.__class__.__name__} {self.id}] {message}")

# --- JUZGADO MSC ---
class InspectorAgent(InstitutionAgent):
    """Auditor general del sistema: Vigila la disciplina, detecta violaciones y reporta irregularidades."""
    def institution_action(self):
        self.log_institution("Revisando integridad y orden del sistema...")
        # Ejemplo: recorrer nodos para detectar anomalías en estados o conexiones
        anomalies = [node for node in self.graph.nodes.values() if node.state < 0.05]
        if anomalies:
            self.log_institution(f"Anomalías detectadas en {len(anomalies)} nodos.")
        else:
            self.log_institution("Todo en orden.")

class PoliceAgent(InstitutionAgent):
    """Responsable de sancionar abusos y actividades sospechosas (plagio, spam de Ω)."""
    def institution_action(self):
        self.log_institution("Monitoreando actividades suspectas en el grafo...")
        # Ejemplo: identificar nodos con conexiones excesivamente negativas
        offenders = [node for node in self.graph.nodes.values() 
                     if any(u < -0.8 for u in node.connections_out.values())]
        if offenders:
            for offender in offenders:
                self.log_institution(f"Aplicando sanción a nodo {offender.id}.")
                offender.update_state(offender.state * 0.9)
        else:
            self.log_institution("Ningún abuso detectado.")

class CoordinatorAgent(InstitutionAgent):
    """Mediador que facilita la resolución de conflictos entre nodos."""
    def institution_action(self):
        self.log_institution("Revisando conflictos entre nodos...")
        # Ejemplo: detectar nodos con estados muy discrepantes conectados y proponer mediación
        conflicts = []
        for node in self.graph.nodes.values():
            for target_id, utility in node.connections_out.items():
                target = self.graph.get_node(target_id)
                if target and abs(node.state - target.state) > 0.5:
                    conflicts.append((node, target))
        if conflicts:
            self.log_institution(f"Se detectaron {len(conflicts)} conflictos; mediando ajustes.")
            for node, target in conflicts:
                avg_state = (node.state + target.state) / 2
                node.update_state(avg_state)
                target.update_state(avg_state)
        else:
            self.log_institution("No se detectaron conflictos.")

class RepairAgent(InstitutionAgent):
    """Rehabilita nodos útiles: restaura reputación mediante la detección de aprendizaje comprobado."""
    def institution_action(self):
        self.log_institution("Analizando nodos para detectar deterioro injustificado...")
        # Ejemplo: identificar nodos con baja reputación pero con conexiones de alta calidad
        for node in self.graph.nodes.values():
            if node.state > 0.7 and hasattr(node, 'reputation') and node.reputation < 0.5:
                self.log_institution(f"Rehabilitando nodo {node.id}.")
                node.update_state(min(1.0, node.state + 0.1))

# --- UNIVERSIDAD MSC ---
class MasterAgent(InstitutionAgent):
    """Mentor: guía y estructura las rutas de aprendizaje de los nodos."""
    def institution_action(self):
        self.log_institution("Organizando rutas de aprendizaje y verificando progresos.")
        # Ejemplo: identificar nodos en estado bajo y sugerir conexiones con nodos de conocimiento avanzado
        weak_nodes = [node for node in self.graph.nodes.values() if node.state < 0.3]
        advanced_nodes = [node for node in self.graph.nodes.values() if node.state > 0.8]
        if weak_nodes and advanced_nodes:
            for node in weak_nodes:
                mentor = random.choice(advanced_nodes)
                if self.graph.add_edge(mentor.id, node.id, 0.8):
                    self.log_institution(f"Conectado nodo {node.id} con mentor {mentor.id}.")

class StudentAgent(InstitutionAgent):
    """Nodo en formación: absorbe y procesa conocimiento conforme a la estructura educativa."""
    def institution_action(self):
        self.log_institution("Buscando oportunidades de aprendizaje...")
        # Ejemplo: mejora su estado conectándose con nodos de aprendizaje (simulamos absorción)
        candidate = self.graph.get_random_node_biased()
        if candidate:
            increment = 0.05
            candidate.update_state(candidate.state + increment)
            self.log_institution(f"Aumentado estado del nodo {candidate.id} en {increment:.2f}.")

class ScientistAgent(InstitutionAgent):
    """Investigador: formula hipótesis emergentes y publica teorías innovadoras."""
    def institution_action(self):
        self.log_institution("Generando hipótesis y evaluando evidencia interna...")
        # Ejemplo: crear nodos con contenido teórico
        hypothesis = f"Hipótesis generada por {self.id} sobre la dinámica del sistema."
        new_node = self.graph.add_node(content=hypothesis, initial_state=0.6, keywords={"teoría", "hipótesis"})
        self.graph.add_edge(random.choice(list(self.graph.nodes.keys())), new_node.id, 0.5)
        self.log_institution(f"Generado nodo teórico {new_node.id}.")

class StorageAgent(InstitutionAgent):
    """Archivador de conocimiento: organiza, clasifica y preserva nodos relevantes."""
    def institution_action(self):
        self.log_institution("Clasificando nodos para archivado...")
        archived = 0
        for node in self.graph.nodes.values():
            if "archivar" in node.keywords or node.state > 0.9:
                # Se podría marcar el nodo para no ser modificado, etc.
                archived += 1
        self.log_institution(f"Archivados {archived} nodos relevantes.")

# --- INSTITUTO FINANCIERO MSC ---
class BankAgent(InstitutionAgent):
    """Gestiona la distribución y regulación de recursos (Ω)."""
    def institution_action(self):
        self.log_institution("Revisando balance económico interno...")
        # Ejemplo: redistribuir Omega a nodos con bajo recurso
        for agent in self.graph.agents if hasattr(self.graph, 'agents') else []:
            if agent.omega < 50:
                bonus = 10
                agent.omega += bonus
                self.log_institution(f"Otorgado bono de {bonus} Ω a agente {agent.id}.")

class MerchantAgent(InstitutionAgent):
    """Facilita el intercambio de recursos e ideas."""
    def institution_action(self):
        self.log_institution("Monitoreando transacciones de ideas y recursos...")
        # Ejemplo: si detecta exceso de acumulación en un nodo, intermedia una redistribución
        resourceful = [agent for agent in self.graph.agents if agent.omega > 150] if hasattr(self.graph, 'agents') else []
        if resourceful:
            donor = random.choice(resourceful)
            recipient = random.choice([agent for agent in self.graph.agents if agent.omega < 80])
            transfer = 20
            donor.omega -= transfer
            recipient.omega += transfer
            self.log_institution(f"Transferencia de {transfer} Ω de {donor.id} a {recipient.id} realizada.")

class MinerAgent(InstitutionAgent):
    """Extrae recursos o información inexplorada (Φ) y los redistribuye al sistema."""
    def institution_action(self):
        self.log_institution("Explorando para extraer nuevos recursos...")
        # Ejemplo: añadir Omega a nodos aleatorios que cumplan ciertos criterios
        target = self.graph.get_random_node_biased()
        if target:
            bonus = 15
            target.update_state(target.state + 0.05)
            self.log_institution(f"Añadido {bonus} Ω y aumentado estado a nodo {target.id}.")

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

def generate_code(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("¡ERROR FATAL! No se encontró la GEMINI_API_KEY en .env o variables de entorno.")
        return None
    try:
        try:
            try:
                import genai
            except ImportError:
                logging.error("The 'genai' module is not installed. Please install it to use this feature.")
                return None
            genai.configure(api_key=api_key)
        except ImportError:
            logging.error("The 'genai' module is not installed. Please install it to use this feature.")
            return None
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        else:
            logging.warning(f"Gemini API did not return text. Response: {response}")
            return ""
    except Exception as e:
        logging.error(f"Error calling the Gemini API: {e}")
        return ""

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

        # --- INICIO: Inicializar Log de Métricas CSV ---
        self.metrics_file = None
        self.metrics_writer = None
        self.metrics_fieldnames = [
            "Step", "Nodes", "Edges", "Density", "AvgClustering",
            "Components", "MeanState", "MedianState", "StdDevState",
            "MinState", "MaxState"
        ]
        metrics_path = self.config.get('metrics_log_path', None)
        if metrics_path:
            try:
                # Comprobar si el archivo existe y está vacío
                is_new_file = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
                self.metrics_file = open(metrics_path, 'a', newline='', encoding='utf-8')
                self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.metrics_fieldnames)
                if is_new_file:
                    self.metrics_writer.writeheader()
                logging.info(f"Logging global metrics to CSV: {metrics_path}")
            except Exception as e:
                logging.error(f"Failed to open metrics log file {metrics_path}: {e}")
                self.metrics_file = None
                self.metrics_writer = None
        # --- FIN Inicializar Log ---

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
        num_coders = config.get('num_coders', 1)
        num_bridging_agents = config.get('num_bridging_agents', 1)  # NUEVO
        num_knowledge_fetchers = config.get('num_knowledge_fetchers', 1)  # NUEVO
        num_horizon_scanners = config.get('num_horizon_scanners', 1)  # NUEVO
        num_epistemic_validators = config.get('num_epistemic_validators', 1)  # NUEVO
        num_technogenesis_agents = config.get('num_technogenesis_agents', 1)  # NUEVO
        for i in range(num_proposers):
            self.agents.append(ProposerAgent(f"P{i}", self.graph, config))
        for i in range(num_evaluators):
            self.agents.append(EvaluatorAgent(f"E{i}", self.graph, config))
        for i in range(num_combiners):
            self.agents.append(CombinerAgent(f"C{i}", self.graph, config))
        for i in range(num_coders):
            self.agents.append(AdvancedCoderAgent(f"CD{i}", self.graph, config))
        # --- AÑADIR Creación de BridgingAgent ---
        for i in range(num_bridging_agents):
             self.agents.append(BridgingAgent(f"B{i}", self.graph, config))
        # --- Fin Creación Bridging ---
        # --- AÑADIR Creación de KnowledgeFetcherAgent ---
        for i in range(num_knowledge_fetchers):
            self.agents.append(KnowledgeFetcherAgent(f"KF{i}", self.graph, config))
        # --- Fin Creación KnowledgeFetcher ---
        # --- AÑADIR Creación de HorizonScannerAgent ---
        for i in range(num_horizon_scanners):
            self.agents.append(HorizonScannerAgent(f"HS{i}", self.graph, config))
        # --- Fin Creación HorizonScanner ---
        # --- AÑADIR Creación de EpistemicValidatorAgent ---
        for i in range(num_epistemic_validators):
            self.agents.append(EpistemicValidatorAgent(f"EV{i}", self.graph, config))
        # --- Fin Creación EpistemicValidator ---
        # --- AÑADIR Creación de TechnogenesisAgent ---
        for i in range(num_technogenesis_agents):
            self.agents.append(TechnogenesisAgent(f"TG{i}", self.graph, config))
        # --- Fin Creación Technogenesis ---

        # Instanciar Agentes del Juzgado MSC
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

        # Instanciar Agentes de la Universidad MSC
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

        # Instanciar Agentes del Instituto Financiero MSC
        num_bank = self.config.get('num_bank', 1)
        for i in range(num_bank):
            self.agents.append(BankAgent(f"BANK{i}", self.graph, self.config))

        num_merchant = self.config.get('num_merchant', 1)
        for i in range(num_merchant):
            self.agents.append(MerchantAgent(f"MER{i}", self.graph, self.config))

        num_miner = self.config.get('num_miner', 1)
        for i in range(num_miner):
            self.agents.append(MinerAgent(f"MIN{i}", self.graph, self.config))

        logging.info(f"Created agents: Proposers={config.get('num_proposers',0)}, Evaluators={config.get('num_evaluators',0)}, Combiners={config.get('num_combiners',0)}, Bridging={num_bridging_agents}, KnowledgeFetchers={num_knowledge_fetchers}, HorizonScanners={num_horizon_scanners}, EpistemicValidators={num_epistemic_validators}, TechnogenesisAgents={num_technogenesis_agents}, Inspectors={num_inspectors}, Police={num_police}, Coordinators={num_coordinators}, RepairAgents={num_repair}, Masters={num_master}, Students={num_students}, Scientists={num_scientists}, StorageAgents={num_storage}, BankAgents={num_bank}, MerchantAgents={num_merchant}, MinerAgents={num_miner}")

    def _simulation_loop(self):
        step_delay = self.config.get('step_delay', 0.1)
        gnn_update_frequency = self.config.get('gnn_update_frequency', 10)
        summary_frequency = self.config.get('summary_frequency', 50)
        max_steps = self.config.get('simulation_steps', None)
        is_api_mode = self.config.get('run_api', False)
        run_continuously = is_api_mode or (max_steps is None)
        gnn_training_frequency = self.config.get('gnn_training_frequency', 50)  # NUEVO
        gnn_training_epochs = self.config.get('gnn_training_epochs', 5)           # NUEVO

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
                # --- Actualizar Embeddings Periódicamente ---
                if self.step_count > 0 and self.step_count % gnn_update_frequency == 0:
                    self.graph.update_embeddings()

                # --- NUEVO: Llamada a Entrenamiento GNN ---
                if self.step_count > 0 and self.step_count % gnn_training_frequency == 0:
                    logging.info(f"GNN: Training model at step {self.step_count}...")
                    self.graph.train_gnn(num_epochs=gnn_training_epochs)
                # --- Fin Llamada Entrenamiento ---

                # --- Lógica NORMAL para Selección/Acción de Agente ---
                runnable_agents = []
                agent_costs = { # Obtener costes
                    "ProposerAgent": self.config.get('proposer_cost', 1.0),
                    "EvaluatorAgent": self.config.get('evaluator_cost', 0.5),
                    "CombinerAgent": self.config.get('combiner_cost', 1.5),
                    "BridgingAgent": self.config.get('bridging_agent_cost', 2.0),
                    "KnowledgeFetcherAgent": self.config.get('knowledge_fetcher_cost', 2.5),
                    "HorizonScannerAgent": self.config.get('horizonscanner_cost', 3.0),
                    "EpistemicValidatorAgent": self.config.get('epistemic_validator_cost', 2.0),
                    "TechnogenesisAgent": self.config.get('technogenesis_cost', 2.5)
                    # Añadir otros si existen
                }
                for agent in self.agents: # Filtrar agentes que pueden actuar
                    cost = agent_costs.get(type(agent).__name__, 1.0)
                    if agent.omega >= cost:
                        runnable_agents.append({'agent': agent, 'cost': cost})

                if runnable_agents: # Si hay agentes que pueden actuar
                    chosen = random.choice(runnable_agents) # Elegir uno al azar
                    agent = chosen['agent']
                    cost = chosen['cost']
                    omega_before = agent.omega
                    agent.omega -= cost # Consumir Omega
                    agent.act() # Ejecutar acción
                    logging.debug(f"Agent {agent.id} acted (Cost: {cost:.2f}, Omega: {omega_before:.2f} -> {agent.omega:.2f})")
                else:
                    logging.warning("No agents have enough Omega to act this step!")

                # Aplicar Regeneración de Omega con reputación
                regen_rate = self.config.get('omega_regeneration_rate', 0.05)
                if regen_rate > 0:
                    for a in self.agents:
                        a.omega += regen_rate * a.reputation
                # --- FIN Lógica NORMAL ---

                # Loguear resumen periódico (sin cambios)
                if self.step_count % summary_frequency == 0:
                    self.graph.print_summary(logging.INFO)
                    # Escribir métricas si está configurado (sin cambios)
                    if self.metrics_writer:
                         metrics = self.graph.get_global_metrics();
                         if metrics: metrics["Step"] = self.step_count;
                         try: self.metrics_writer.writerow(metrics); self.metrics_file.flush()
                         except Exception as e: logging.error(f"Error writing CSV step {self.step_count}: {e}")
            # --- Fin del bloque with self.lock ---
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
            self.graph.log_global_metrics(logging.INFO, current_step=self.step_count)  # <-- Pasar current_step

            # --- NUEVO: Escribir Métricas Finales ---
            if self.metrics_writer:
                logging.info("Writing final metrics to CSV...")
                metrics = self.graph.get_global_metrics()
                if metrics:
                    metrics["Step"] = self.step_count  # Paso final
                    try:
                        self.metrics_writer.writerow(metrics)
                    except Exception as e:
                        logging.error(f"Error writing final metrics to CSV: {e}")
            # --- FIN Escritura Final ---

            # --- NUEVO: Cerrar archivo CSV al final de stop() ---
            if self.metrics_file:
                try:
                    self.metrics_file.close()
                    logging.info("Metrics log file closed.")
                except Exception as e:
                    logging.error(f"Error closing metrics log file: {e}")
            # --- FIN Cerrar Archivo ---

            # --- Llamada a guardar visualización (si path está configurado) ---
            save_vis_path = self.config.get('visualization_output_path', None)
            if save_vis_path:  # Solo intenta visualizar si se dio una ruta
                if self.graph.nodes:
                    logging.info(f"Attempting to save final graph visualization to {save_vis_path}...")
                    try:
                        self.graph.visualize_graph(self.config)  # visualize_graph usará el path de config
                    except Exception as e:
                        logging.error(f"Error during final visualization saving: {e}")
                else:
                    logging.warning("Graph is empty, skipping visualization saving.")
            # --- Fin Llamada Visualización ---
        logging.info("--- Simulation Runner Stopped ---")

    def get_status(self):
        """Devuelve el estado actual de la simulación (thread-safe)."""
        with self.lock:
             num_nodes = len(self.graph.nodes)
             num_edges = sum(len(n.connections_out) for n in self.graph.nodes.values())
             avg_state = (sum(n.state for n in self.graph.nodes.values()) / num_nodes) if num_nodes > 0 else 0
             num_agents = len(self.agents)
             avg_reputation = (sum(a.reputation for a in self.agents) / num_agents) if num_agents > 0 else 0.0
             avg_omega = (sum(a.omega for a in self.agents) / num_agents) if num_agents > 0 else 0.0
             status = {
                 "is_running": self.is_running,
                 "current_step": self.step_count,
                 "node_count": num_nodes,
                 "edge_count": num_edges,
                 "average_state": round(avg_state, 3),
                 "embeddings_count": len(self.graph.node_embeddings),
                 "average_reputation": round(avg_reputation, 3),
                 "average_omega": round(avg_omega, 2)
             }
        return status

    def get_graph_elements_for_cytoscape(self):
        """Prepara los nodos y aristas en formato para Dash Cytoscape."""
        elements = []
        with self.lock:
            try:
                cmap = plt.cm.viridis
                norm = plt.Normalize(vmin=0, vmax=1)
                for node_id, node in self.graph.nodes.items():
                    try:
                        node_color = cmap(norm(node.state))
                        hex_color = '#%02x%02x%02x' % tuple([int(c * 255) for c in node_color[:3]])
                    except Exception as ex:
                        app.logger.error("Error procesando nodo %s: %s", node_id, ex)
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
        # --- Nuevos parámetros para entrenamiento ---
        'gnn_training_frequency': 50,
        'gnn_training_epochs': 5,
        'gnn_learning_rate': 0.01,
        # --- Nuevos Defaults para Omega ---
        'initial_omega': 100.0,
        'proposer_cost': 1.0,
        'evaluator_cost': 0.5,
        'combiner_cost': 1.5,
        'omega_regeneration_rate': 0.1,
        # --- Nuevos Defaults para Recompensas Omega ---
        'evaluator_reward_factor': 2.0,
        'evaluator_reward_threshold': 0.05,
        'proposer_reward_factor': 0.5,
        'combiner_reward_factor': 1.0,
        # --- Nuevo Default para Guardar Visualización ---
        'visualization_output_path': None,
        # --- Nuevo Default para Log de Métricas ---
        'metrics_log_path': None, # Default: no guardar log
        # --- Nuevos Defaults para BridgingAgent ---
        'num_bridging_agents': 1,         # Número de agentes puente
        'bridging_agent_cost': 2.0,         # Coste Omega por intento de puente
        'bridging_similarity_threshold': 0.75, # Umbral similitud [0,1] para crear puente
        'bridging_adjusted_threshold': 0.65, # Umbral ajustado para similitud
        # --- Fin Nuevos Defaults ---
        # --- Nuevo Default para KnowledgeFetcher ---
        'knowledge_fetcher_cost': 2.5, # Coste Omega por búsqueda/creación
        'num_knowledge_fetchers': 1,   # Default 1 fetcher
        # --- Fin Nuevo Default ---
        # --- Nuevos Defaults para Reputación Psi ---
        'initial_reputation': 1.0,     # Reputación inicial
        'proposer_psi_reward': 0.02,   # Recompensa Ψ por proponer nodo/enlace
        'evaluator_psi_reward': 0.01,  # Recompensa Ψ por evaluar nodo
        'combiner_psi_reward': 0.03,   # Recompensa Ψ por combinación exitosa
        'bridging_psi_reward': 0.05,   # Recompensa Ψ por crear puente exitoso
        'fetcher_psi_reward': 0.04,    # Recompensa Ψ por añadir nodo de info
        'horizonscanner_cost': 3.0,    # Coste Omega por HorizonScanner
        'horizonscanner_psi_reward': 0.05, # Recompensa Ψ por HorizonScanner
        'num_horizon_scanners': 1,     # Número de HorizonScanner
        'epistemic_validator_cost': 2.0, # Coste Omega por EpistemicValidator
        'epistemic_validator_omega_reward': 0.05, # Recompensa Omega por EpistemicValidator
        'epistemic_validator_psi_reward': 0.01, # Recompensa Ψ por EpistemicValidator
        'num_epistemic_validators': 1, # Número de EpistemicValidator
        'technogenesis_cost': 2.5, # Coste Omega por TechnogenesisAgent
        'technogenesis_omega_reward': 0.05, # Recompensa Omega por TechnogenesisAgent
        'technogenesis_psi_reward': 0.01, # Recompensa Ψ por TechnogenesisAgent
        'num_technogenesis_agents': 1, # Número de TechnogenesisAgent
        # --- Fin Nuevos Defaults ---
        # --- Nuevos Defaults para Agentes Institucionales ---
        'num_inspectors': 1, # Número de InspectorAgents
        'num_police': 1, # Número de PoliceAgents
        'num_coordinators': 1, # Número de CoordinatorAgents
        'num_repair': 1, # Número de RepairAgents
        'num_master': 1, # Número de MasterAgents
        'num_students': 1, # Número de StudentAgents
        'num_scientists': 1, # Número de ScientistAgents
        'num_storage': 1, # Número de StorageAgents
        'num_bank': 1, # Número de BankAgents
        'num_merchant': 1, # Número de MerchantAgents
        'num_miner': 1, # Número de MinerAgents
        # --- Fin Nuevos Defaults ---
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
    """Endpoint API para obtener elementos del grafo para Cytoscape."""
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
    parser.add_argument('--simulation_steps', type=int, default=None,
                        help='Run for a fixed number of steps (default: continuous if --run_api is off).')
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
    parser.add_argument('--visualize_graph', action='store_true', help='Show graph plot at the end (only in fixed-step mode without --run_api).')
    parser.add_argument('--save_state', type=str, help='Base path to save simulation state on stop.')
    parser.add_argument('--load_state', type=str, help='Base path to load simulation state from.')
    # --- Nuevo Arg para Guardar Visualización ---
    parser.add_argument('--visualization_output_path', type=str, default=None,
                        help='Path to save the final graph visualization image (e.g., graph.png).')
    # --- Fin Nuevo Arg ---
    # --- Nuevo Arg para Log de Métricas ---
    parser.add_argument('--metrics_log_path', type=str, default=None,
                        help='Path to save the global metrics CSV log (e.g., metrics.csv).')
    # --- Fin Nuevo Arg ---
    parser.add_argument('--run_api', action='store_true', help='Run Flask API server (runs continuously).')
    parser.add_argument('--summary_frequency', type=int, help='Log summary frequency (steps).')
    # --- Nuevos args para Omega ---
    parser.add_argument('--initial_omega', type=float, help='Initial Omega resource for agents.')
    parser.add_argument('--proposer_cost', type=float, help='Omega cost for Proposer action.')
    parser.add_argument('--evaluator_cost', type=float, help='Omega cost for Evaluator action.')
    parser.add_argument('--combiner_cost', type=float, help='Omega cost for Combiner action.')
    parser.add_argument('--omega_regeneration_rate', type=float, help='Omega regenerated per agent per step.')
    # --- Nuevos args para Recompensas Omega ---
    parser.add_argument('--evaluator_reward_factor', type=float, help='Omega reward multiplier for evaluation.')
    parser.add_argument('--evaluator_reward_threshold', type=float, help='Min state increase for evaluator reward.')
    parser.add_argument('--proposer_reward_factor', type=float, help='Omega reward multiplier for proposer.')
    parser.add_argument('--combiner_reward_factor', type=float, help='Omega reward multiplier for combiner.')
    # --- Fin Nuevos Args ---
    # --- Nuevos argumentos para entrenamiento GNN ---
    parser.add_argument('--gnn_training_frequency', type=int, help='Frequency (steps) to train GNN.')
    parser.add_argument('--gnn_training_epochs', type=int, help='Epochs per GNN training session.')
    parser.add_argument('--gnn_learning_rate', type=float, help='Learning rate for GNN optimizer.')
    # --- Fin Nuevos ---
    # --- Nuevos args para BridgingAgent ---
    parser.add_argument('--num_bridging_agents', type=int, help='Number of BridgingAgents.')
    parser.add_argument('--bridging_agent_cost', type=float, help='Omega cost for BridgingAgent action.')
    parser.add_argument('--bridging_similarity_threshold', type=float, help='Embedding similarity threshold [0,1] for bridging.')
    parser.add_argument('--bridging_adjusted_threshold', type=float, help='Adjusted similarity threshold for bridging.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para KnowledgeFetcher ---
    parser.add_argument('--knowledge_fetcher_cost', type=float, help='Omega cost for KnowledgeFetcher action.')
    parser.add_argument('--num_knowledge_fetchers', type=int, default=1, help='Number of KnowledgeFetcher agents.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para Reputación Psi ---
    parser.add_argument('--initial_reputation', type=float, help='Initial Psi reputation for agents.')
    parser.add_argument('--proposer_psi_reward', type=float, help='Psi reward for Proposer action.')
    parser.add_argument('--evaluator_psi_reward', type=float, help='Psi reward for Evaluator action.')
    parser.add_argument('--combiner_psi_reward', type=float, help='Psi reward for Combiner action.')
    parser.add_argument('--bridging_psi_reward', type=float, help='Psi reward for Bridging action.')
    parser.add_argument('--fetcher_psi_reward', type=float, help='Psi reward for KnowledgeFetcher action.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para HorizonScanner ---
    parser.add_argument('--horizonscanner_cost', type=float, help='Omega cost for HorizonScanner action.')
    parser.add_argument('--horizonscanner_psi_reward', type=float, help='Psi reward for HorizonScanner action.')
    parser.add_argument('--num_horizon_scanners', type=int, default=1, help='Number of HorizonScanner agents.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para EpistemicValidator ---
    parser.add_argument('--epistemic_validator_cost', type=float, help='Omega cost for EpistemicValidator action.')
    parser.add_argument('--epistemic_validator_omega_reward', type=float, help='Omega reward for EpistemicValidator action.')
    parser.add_argument('--epistemic_validator_psi_reward', type=float, help='Psi reward for EpistemicValidator action.')
    parser.add_argument('--num_epistemic_validators', type=int, default=1, help='Number of EpistemicValidator agents.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para TechnogenesisAgent ---
    parser.add_argument('--technogenesis_cost', type=float, help='Omega cost for TechnogenesisAgent action.')
    parser.add_argument('--technogenesis_omega_reward', type=float, help='Omega reward for TechnogenesisAgent action.')
    parser.add_argument('--technogenesis_psi_reward', type=float, help='Psi reward for TechnogenesisAgent action.')
    parser.add_argument('--num_technogenesis_agents', type=int, default=1, help='Number of TechnogenesisAgent agents.')
    # --- Fin Nuevos Args ---
    # --- Nuevos args para Agentes Institucionales ---
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
    # --- Fin Nuevos Args ---

    args = parser.parse_args()
    final_config = load_config(args)
    final_config['run_api'] = args.run_api

    simulation_runner = SimulationRunner(final_config)
    simulation_runner.start()

    # Esperamos unos segundos para asegurar que la simulación se inicia y la API responda adecuadamente
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
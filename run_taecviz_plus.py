import os
import logging
import socketserver
import sys
import threading
from http.server import SimpleHTTPRequestHandler
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para renderizar sin interfaz gráfica
import json
from msc_simulation import SimulationRunner, load_config
import argparse
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clase TAECVizPlusHandler basada en SimpleHTTPRequestHandler
class TAECVizPlusHandler(SimpleHTTPRequestHandler):
    
    def do_GET(self):
        """Maneja peticiones GET al servidor"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_dashboard_html().encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if hasattr(self.server, 'simulation_runner'):
                status = self.server.simulation_runner.get_status()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.wfile.write(json.dumps({'error': 'No simulation runner attached'}).encode())
        elif self.path == '/api/graph':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if hasattr(self.server, 'simulation_runner'):
                elements = self.server.simulation_runner.get_graph_elements_for_cytoscape()
                self.wfile.write(json.dumps(elements).encode())
            else:
                self.wfile.write(json.dumps([]).encode())
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if hasattr(self.server, 'simulation_runner') and hasattr(self.server.simulation_runner, 'graph'):
                metrics = self.server.simulation_runner.graph.get_global_metrics()
                self.wfile.write(json.dumps(metrics).encode())
            else:
                self.wfile.write(json.dumps({}).encode())
        elif self.path == '/api/agents':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            agent_data = self.get_agents_data()
            self.wfile.write(json.dumps(agent_data).encode())
        elif self.path == '/api/heatmap':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            heatmap_data = self.generate_heatmap()
            self.wfile.write(heatmap_data)
        elif self.path == '/api/time_series':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            time_series = self.get_time_series_data()
            self.wfile.write(json.dumps(time_series).encode())
        elif self.path == '/api/network_analysis':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            analysis = self.get_network_analysis()
            self.wfile.write(json.dumps(analysis).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
    
    def get_agents_data(self):
        """Obtiene datos sobre los agentes en la simulación"""
        agent_data = []
        if not hasattr(self.server, 'simulation_runner') or not self.server.simulation_runner:
            return agent_data
            
        try:
            for agent in self.server.simulation_runner.agents:
                agent_info = {
                    "id": agent.id,
                    "type": type(agent).__name__,
                    "omega": round(getattr(agent, 'omega', 0), 2),
                    "reputation": round(getattr(agent, 'reputation', 1.0), 2)
                }
                agent_data.append(agent_info)
        except Exception as e:
            logging.error(f"Error getting agent data: {e}")
            
        return agent_data
    
    def get_network_analysis(self):
        """Analiza la red para obtener métricas adicionales"""
        analysis_data = {
            "communities": [],
            "centrality": {},
            "influential_nodes": []
        }
        
        if not hasattr(self.server, 'simulation_runner') or not self.server.simulation_runner:
            return analysis_data
            
        try:
            import networkx as nx
            if len(self.server.simulation_runner.graph.nodes) > 0:
                # Construir grafo NetworkX
                G = nx.DiGraph()
                for node_id in self.server.simulation_runner.graph.nodes:
                    G.add_node(node_id)
                
                for node_id, node in self.server.simulation_runner.graph.nodes.items():
                    for target_id in node.connections_out:
                        G.add_edge(node_id, target_id)
                
                # Calcular centralidad
                try:
                    centrality = nx.betweenness_centrality(G)
                    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                    analysis_data["centrality"] = {str(k): round(v, 3) for k, v in top_central}
                except:
                    pass
                
                # Obtener nodos influyentes basados en PageRank
                try:
                    pagerank = nx.pagerank(G)
                    top_influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
                    analysis_data["influential_nodes"] = [{"id": str(k), "rank": round(v, 3)} for k, v in top_influential]
                except:
                    pass
                
                # Detectar comunidades
                try:
                    # Usar algoritmo de detección de comunidades
                    undirected_G = G.to_undirected()
                    from networkx.algorithms import community
                    communities = community.greedy_modularity_communities(undirected_G)
                    
                    for i, comm in enumerate(communities):
                        if i >= 5:  # Limitamos a 5 comunidades para no sobrecargar
                            break
                        analysis_data["communities"].append({
                            "id": i,
                            "size": len(comm),
                            "nodes": [str(n) for n in list(comm)[:10]]  # Primeros 10 nodos como muestra
                        })
                except Exception as e:
                    logging.error(f"Community detection error: {e}")
        
        except Exception as e:
            logging.error(f"Network analysis error: {e}")
        
        return analysis_data
    
    def generate_heatmap(self):
        """Genera un mapa de calor del estado de los nodos"""
        try:
            if not hasattr(self.server, 'simulation_runner') or not self.server.simulation_runner:
                raise Exception("No simulation runner available")
                
            graph = self.server.simulation_runner.graph
            if not graph.nodes:
                raise Exception("Graph has no nodes")
            
            # Crear heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Organizar nodos en una cuadrícula
            node_count = len(graph.nodes)
            grid_size = int(node_count ** 0.5) + 1
            
            # Crear matriz para el heatmap
            data = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            for i, node_id in enumerate(graph.nodes):
                x, y = i % grid_size, i // grid_size
                data[y][x] = graph.nodes[node_id].state
            
            # Generar heatmap
            im = ax.imshow(data, cmap='viridis')
            plt.colorbar(im, ax=ax, label='Node State')
            ax.set_title('Node State Heatmap')
            
            # Guardar en buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return buf.read()
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return buf.read()
    
    def get_time_series_data(self):
        """Obtiene datos de series temporales para gráficos"""
        time_series = {
            "step_values": [],
            "metrics": {}
        }
        
        if not hasattr(self.server, 'simulation_runner') or not self.server.simulation_runner:
            return time_series
        
        try:
            # Verificar si existe historial de métricas
            if hasattr(self.server.simulation_runner, 'metrics_history'):
                metrics_history = self.server.simulation_runner.metrics_history
                time_series["step_values"] = list(metrics_history.keys())
                
                # Inicializar cada métrica como una lista vacía
                for step in metrics_history:
                    for metric_name in metrics_history[step]:
                        if metric_name not in time_series["metrics"]:
                            time_series["metrics"][metric_name] = []
                
                # Llenar las series temporales
                for step in sorted(metrics_history.keys()):
                    for metric_name, value in metrics_history[step].items():
                        time_series["metrics"][metric_name].append(value)
                        
            elif hasattr(self.server.simulation_runner, 'history'):
                # Alternativa si no hay metrics_history pero hay history
                history = self.server.simulation_runner.history
                if history:
                    time_series["step_values"] = list(range(len(history)))
                    
                    # Extraer métricas comunes en todas las entradas
                    if history:
                        first_entry = history[0]
                        for key in first_entry:
                            if key not in ['step']:
                                time_series["metrics"][key] = []
                        
                        for entry in history:
                            for key in time_series["metrics"]:
                                if key in entry:
                                    time_series["metrics"][key].append(entry[key])
                                else:
                                    time_series["metrics"][key].append(None)
        
        except Exception as e:
            logging.error(f"Error getting time series data: {e}")
        
        return time_series
    
    def generate_dashboard_html(self):
        """Genera el HTML para el dashboard mejorado"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TAECViz+ - Advanced MSC Framework Visualization</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.23.0/dist/cytoscape.min.js"></script>
            <style>
                body { padding-top: 60px; }
                .graph-container { height: 500px; border: 1px solid #ddd; }
                .card { margin-bottom: 20px; }
                .status-indicator { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
                .status-running { background-color: #28a745; }
                .status-stopped { background-color: #dc3545; }
                .metric-value { font-size: 1.5rem; font-weight: bold; }
                .metric-title { font-size: 0.9rem; color: #6c757d; }
                .node-info { font-size: 0.9rem; }
                .table-sm td, .table-sm th { padding: 0.3rem; }
                #time-series-chart { height: 300px; }
                #heatmap-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">
                        <i class="bi bi-braces"></i> TAECViz+
                    </a>
                    <div class="d-flex align-items-center">
                        <span class="text-light me-2">Status:</span>
                        <span class="status-indicator me-1" id="status-light"></span>
                        <span id="status-text" class="text-light me-3">Connecting...</span>
                        <span class="text-light me-2">Step:</span>
                        <span id="step-counter" class="text-light me-3">0</span>
                        <button class="btn btn-sm btn-outline-light me-2" id="refresh-btn">
                            <i class="bi bi-arrow-clockwise"></i> Refresh
                        </button>
                    </div>
                </div>
            </nav>

            <div class="container-fluid">
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <i class="bi bi-info-circle"></i> Simulation Metrics
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Nodes</div>
                                        <div class="metric-value" id="node-count">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Edges</div>
                                        <div class="metric-value" id="edge-count">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Agents</div>
                                        <div class="metric-value" id="agent-count">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Avg. State</div>
                                        <div class="metric-value" id="avg-state">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Avg. Omega</div>
                                        <div class="metric-value" id="avg-omega">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Runtime</div>
                                        <div class="metric-value" id="runtime">-</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header bg-info text-white">
                                <i class="bi bi-graph-up"></i> Advanced Metrics
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Density</div>
                                        <div class="metric-value" id="density">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Clustering</div>
                                        <div class="metric-value" id="clustering">-</div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="metric-title">Components</div>
                                        <div class="metric-value" id="components">-</div>
                                    </div>
                                    <div class="col-6">
                                        <div class="metric-title">State StdDev</div>
                                        <div class="metric-value" id="state-stddev">-</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span><i class="bi bi-diagram-3"></i> Knowledge Graph</span>
                                <div>
                                    <button class="btn btn-sm btn-outline-secondary" id="reset-view-btn">
                                        <i class="bi bi-arrows-angle-expand"></i> Reset View
                                    </button>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="graph-container" class="graph-container"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-3">
                        <div class="card mb-4">
                            <div class="card-header bg-success text-white">
                                <i class="bi bi-people"></i> Agent Status
                            </div>
                            <div class="card-body p-0" style="max-height: 250px; overflow-y: auto;">
                                <table class="table table-sm table-hover mb-0" id="agents-table">
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Type</th>
                                            <th>Omega</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="3" class="text-center">Loading...</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <i class="bi bi-info-circle"></i> Node Information
                            </div>
                            <div class="card-body">
                                <div id="node-info">
                                    <p class="text-muted text-center">Click on a node to see details</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-info text-white">
                                <i class="bi bi-graph-up"></i> Time Series Data
                            </div>
                            <div class="card-body">
                                <canvas id="time-series-chart"></canvas>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <i class="bi bi-hdd-network"></i> Network Analysis
                            </div>
                            <div class="card-body">
                                <h6>Top Central Nodes</h6>
                                <ul id="central-nodes" class="small"></ul>
                                <h6>Communities</h6>
                                <div id="communities-info" class="small"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <i class="bi bi-thermometer-half"></i> Node State Heatmap
                            </div>
                            <div class="card-body">
                                <div id="heatmap-container" class="text-center">
                                    <img src="/api/heatmap" alt="Node State Heatmap" class="img-fluid">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // Controlador principal del dashboard
                const Dashboard = {
                    // Estado de la aplicación
                    state: {
                        isRunning: false,
                        currentStep: 0,
                        selectedNode: null,
                        refreshInterval: null,
                        cy: null,
                        timeSeriesChart: null
                    },
                    
                    // Inicializar dashboard
                    init: function() {
                        this.setupEventListeners();
                        this.initGraph();
                        this.refreshData();
                        this.startAutoRefresh();
                    },
                    
                    // Configurar event listeners
                    setupEventListeners: function() {
                        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshData());
                        document.getElementById('reset-view-btn').addEventListener('click', () => this.resetGraphView());
                    },
                    
                    // Inicializar gráfico de conocimiento
                    initGraph: function() {
                        this.state.cy = cytoscape({
                            container: document.getElementById('graph-container'),
                            style: [
                                {
                                    selector: 'node',
                                    style: {
                                        'background-color': 'data(color)',
                                        'label': 'data(label)',
                                        'width': 'data(width)',
                                        'height': 'data(height)',
                                        'text-wrap': 'wrap',
                                        'text-max-width': '80px',
                                        'font-size': '8px',
                                        'text-valign': 'center',
                                        'text-halign': 'center'
                                    }
                                },
                                {
                                    selector: 'edge',
                                    style: {
                                        'width': 'data(width)',
                                        'line-color': 'data(color)',
                                        'curve-style': 'bezier',
                                        'target-arrow-shape': 'triangle',
                                        'target-arrow-color': 'data(color)',
                                        'arrow-scale': 0.7
                                    }
                                },
                                {
                                    selector: 'node:selected',
                                    style: {
                                        'border-width': '3px',
                                        'border-color': '#ff0000',
                                        'border-style': 'solid'
                                    }
                                }
                            ],
                            layout: {
                                name: 'cose',
                                animate: false,
                                nodeDimensionsIncludeLabels: true
                            }
                        });
                        
                        // Evento de clic en nodo
                        this.state.cy.on('tap', 'node', (evt) => {
                            const node = evt.target;
                            this.displayNodeInfo(node.data());
                            this.state.selectedNode = node.id();
                        });
                        
                        // Evento de clic en el fondo para deseleccionar
                        this.state.cy.on('tap', function(evt) {
                            if (evt.target === this.state.cy) {
                                document.getElementById('node-info').innerHTML = 
                                    '<p class="text-muted text-center">Click on a node to see details</p>';
                                this.state.selectedNode = null;
                            }
                        }.bind(this));
                    },
                    
                    // Reiniciar vista del grafo
                    resetGraphView: function() {
                        if (this.state.cy) {
                            this.state.cy.fit();
                            this.state.cy.center();
                        }
                    },
                    
                    // Mostrar información del nodo
                    displayNodeInfo: function(nodeData) {
                        const infoContainer = document.getElementById('node-info');
                        let content = `
                            <div class="node-info">
                                <p><strong>ID:</strong> ${nodeData.id}</p>
                                <p><strong>State:</strong> ${nodeData.state?.toFixed(3) || 'N/A'}</p>
                                <p><strong>Keywords:</strong> ${nodeData.keywords || 'None'}</p>
                            </div>
                        `;
                        infoContainer.innerHTML = content;
                    },
                    
                    // Actualizar el gráfico con datos del servidor
                    updateGraph: function(elements) {
                        if (!this.state.cy) return;
                        
                        // Guardar posiciones de nodos existentes
                        const positions = {};
                        this.state.cy.nodes().forEach(node => {
                            positions[node.id()] = { x: node.position('x'), y: node.position('y') };
                        });
                        
                        // Procesar elementos para Cytoscape
                        const cytoscapeElements = elements.map(elem => {
                            // Conservar posiciones si el nodo ya existe
                            if (elem.data.id && positions[elem.data.id]) {
                                return {
                                    data: elem.data,
                                    style: elem.style,
                                    position: positions[elem.data.id]
                                };
                            }
                            return {
                                data: elem.data,
                                style: elem.style
                            };
                        });
                        
                        // Cargar elementos en Cytoscape
                        this.state.cy.elements().remove();
                        this.state.cy.add(cytoscapeElements);
                        
                        // Aplicar layout solo a nuevos nodos
                        const existingNodes = this.state.cy.nodes().filter(node => positions[node.id()]);
                        const newNodes = this.state.cy.nodes().filter(node => !positions[node.id()]);
                        
                        if (newNodes.length > 0 && existingNodes.length > 0) {
                            newNodes.layout({
                                name: 'cose',
                                animate: false,
                                randomize: true,
                                nodeDimensionsIncludeLabels: true
                            }).run();
                        } 
                        else if (this.state.cy.elements().length > 0) {
                            this.state.cy.layout({
                                name: 'cose',
                                animate: false,
                                nodeDimensionsIncludeLabels: true
                            }).run();
                        }
                        
                        // Restaurar nodo seleccionado
                        if (this.state.selectedNode) {
                            const node = this.state.cy.getElementById(this.state.selectedNode);
                            if (node.length > 0) {
                                node.select();
                                this.displayNodeInfo(node.data());
                            }
                        }
                    },
                    
                    // Actualizar tabla de agentes
                    updateAgentsTable: function(agents) {
                        const tableBody = document.getElementById('agents-table').getElementsByTagName('tbody')[0];
                        let tableHTML = '';
                        
                        if (agents && agents.length > 0) {
                            agents.forEach(agent => {
                                tableHTML += `<tr>
                                    <td>${agent.id}</td>
                                    <td>${agent.type}</td>
                                    <td>${agent.omega}</td>
                                </tr>`;
                            });
                        } else {
                            tableHTML = '<tr><td colspan="3" class="text-center">No agents available</td></tr>';
                        }
                        
                        tableBody.innerHTML = tableHTML;
                    },
                    
                    // Actualizar métricas avanzadas
                    updateAdvancedMetrics: function(metrics) {
                        if (!metrics) return;
                        
                        document.getElementById('density').textContent = 
                            metrics.Density !== null ? metrics.Density.toFixed(4) : 'N/A';
                        document.getElementById('clustering').textContent = 
                            metrics.AvgClustering !== null ? metrics.AvgClustering.toFixed(4) : 'N/A';
                        document.getElementById('components').textContent = 
                            metrics.Components !== null ? metrics.Components : 'N/A';
                        document.getElementById('state-stddev').textContent = 
                            metrics.StdDevState !== null ? metrics.StdDevState.toFixed(4) : 'N/A';
                    },
                    
                    // Actualizar análisis de red
                    updateNetworkAnalysis: function(analysisData) {
                        if (!analysisData) return;
                        
                        // Actualizar nodos centrales
                        const centralNodes = document.getElementById('central-nodes');
                        centralNodes.innerHTML = '';
                        
                        if (analysisData.centrality && Object.keys(analysisData.centrality).length > 0) {
                            Object.entries(analysisData.centrality).slice(0, 5).forEach(([nodeId, value]) => {
                                const li = document.createElement('li');
                                li.textContent = `Node ${nodeId}: ${value}`;
                                centralNodes.appendChild(li);
                            });
                        } else {
                            centralNodes.innerHTML = '<li>No data available</li>';
                        }
                        
                        // Actualizar información de comunidades
                        const communitiesInfo = document.getElementById('communities-info');
                        communitiesInfo.innerHTML = '';
                        
                        if (analysisData.communities && analysisData.communities.length > 0) {
                            analysisData.communities.forEach(community => {
                                const div = document.createElement('div');
                                div.innerHTML = `<p><strong>Community ${community.id}:</strong> ${community.size} nodes</p>`;
                                communitiesInfo.appendChild(div);
                            });
                        } else {
                            communitiesInfo.innerHTML = '<p>No community data available</p>';
                        }
                    },
                    
                    // Actualizar gráfico de series temporales
                    updateTimeSeriesChart: function(timeSeriesData) {
                        if (!timeSeriesData || !timeSeriesData.step_values || timeSeriesData.step_values.length === 0) {
                            return;
                        }
                        
                        const ctx = document.getElementById('time-series-chart').getContext('2d');
                        
                        // Preparar datasets
                        const datasets = [];
                        const colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff', '#ff9f40'];
                        
                        let colorIndex = 0;
                        for (const [metricName, values] of Object.entries(timeSeriesData.metrics)) {
                            // Limitar a 6 métricas máximo para no saturar
                            if (colorIndex >= colors.length) break;
                            
                            datasets.push({
                                label: metricName,
                                data: values,
                                borderColor: colors[colorIndex % colors.length],
                                backgroundColor: colors[colorIndex % colors.length] + '33',
                                fill: false,
                                tension: 0.3
                            });
                            
                            colorIndex++;
                        }
                        
                        // Destruir gráfico anterior si existe
                        if (this.state.timeSeriesChart) {
                            this.state.timeSeriesChart.destroy();
                        }
                        
                        // Crear nuevo gráfico
                        this.state.timeSeriesChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: timeSeriesData.step_values,
                                datasets: datasets
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: true
                                    }
                                }
                            }
                        });
                    },
                    
                    // Actualizar imagen del mapa de calor
                    updateHeatmap: function() {
                        const heatmapImg = document.querySelector('#heatmap-container img');
                        if (heatmapImg) {
                            // Agregar timestamp para evitar caché
                            heatmapImg.src = `/api/heatmap?t=${Date.now()}`;
                        }
                    },
                    
                    // Obtener y actualizar todos los datos
                    refreshData: async function() {
                        try {
                            // Obtener estado de la simulación
                            const statusResponse = await fetch('/api/status');
                            const statusData = await statusResponse.json();
                            
                            // Actualizar indicadores de estado
                            this.updateStatusIndicators(statusData);
                            
                            // Obtener datos del grafo
                            const graphResponse = await fetch('/api/graph');
                            const graphData = await graphResponse.json();
                            this.updateGraph(graphData);
                            
                            // Obtener métricas avanzadas
                            const metricsResponse = await fetch('/api/metrics');
                            const metricsData = await metricsResponse.json();
                            this.updateAdvancedMetrics(metricsData);
                            
                            // Obtener datos de agentes
                            const agentsResponse = await fetch('/api/agents');
                            const agentsData = await agentsResponse.json();
                            this.updateAgentsTable(agentsData);
                            
                            // Obtener análisis de red
                            const networkAnalysisResponse = await fetch('/api/network_analysis');
                            const networkAnalysisData = await networkAnalysisResponse.json();
                            this.updateNetworkAnalysis(networkAnalysisData);
                            
                            // Obtener datos de series temporales
                            const timeSeriesResponse = await fetch('/api/time_series');
                            const timeSeriesData = await timeSeriesResponse.json();
                            this.updateTimeSeriesChart(timeSeriesData);
                            
                            // Actualizar mapa de calor
                            this.updateHeatmap();
                            
                        } catch (error) {
                            console.error('Error refreshing data:', error);
                        }
                    },
                    
                    // Actualizar indicadores de estado
                    updateStatusIndicators: function(status) {
                        if (!status) return;
                        
                        this.state.isRunning = status.is_running;
                        this.state.currentStep = status.current_step;
                        
                        // Actualizar texto e indicador visual
                        const statusText = document.getElementById('status-text');
                        const statusLight = document.getElementById('status-light');
                        
                        statusText.textContent = status.is_running ? 'Running' : 'Stopped';
                        statusLight.className = 'status-indicator me-1 ' + 
                            (status.is_running ? 'status-running' : 'status-stopped');
                        
                        // Actualizar contadores
                        document.getElementById('step-counter').textContent = status.current_step || 0;
                        document.getElementById('node-count').textContent = status.node_count || 0;
                        document.getElementById('edge-count').textContent = status.edge_count || 0;
                        document.getElementById('agent-count').textContent = status.agent_count || 0;
                        document.getElementById('avg-state').textContent = 
                            status.average_state !== undefined ? status.average_state.toFixed(3) : '-';
                        document.getElementById('avg-omega').textContent = 
                            status.average_omega !== undefined ? status.average_omega.toFixed(1) : '-';
                        document.getElementById('runtime').textContent = 
                            status.simulation_time !== undefined ? status.simulation_time + 's' : '-';
                    },
                    
                    // Iniciar actualización automática
                    startAutoRefresh: function() {
                        const refreshInterval = 5000; // 5 segundos
                        this.state.refreshInterval = setInterval(() => this.refreshData(), refreshInterval);
                    },
                    
                    // Detener actualización automática
                    stopAutoRefresh: function() {
                        if (this.state.refreshInterval) {
                            clearInterval(this.state.refreshInterval);
                        }
                    }
                };
                
                // Iniciar dashboard cuando el DOM esté listo
                document.addEventListener('DOMContentLoaded', () => Dashboard.init());
            </script>
        </body>
        </html>
        """

def start_taecviz_plus(simulation_runner, port=8080):
    """Inicia el servidor de TAECViz+ conectado a la simulación"""
    logging.info(f"Iniciando TAECViz+ en puerto {port}...")
    server = socketserver.TCPServer(("", port), TAECVizPlusHandler)
    server.simulation_runner = simulation_runner
    
    # Ejecutar servidor en un hilo separado
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    logging.info(f"TAECViz+ iniciado en http://localhost:{port}")
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar TAECViz+ conectado a una simulación MSC")
    parser.add_argument('--port', type=int, default=8080, help='Puerto en el que ejecutar el servidor TAECViz+')
    parser.add_argument('--config', type=str, default='config.yaml', help='Archivo de configuración para la simulación')
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args)
    
    # Crear y arrancar simulación
    simulation = SimulationRunner(config)
    simulation.start()
    
    # Iniciar TAECViz+
    server = start_taecviz_plus(simulation, args.port)
    
    print(f"TAECViz+ iniciado en http://localhost:{args.port}")
    print("Presiona Ctrl+C para detener")
    
    try:
        # Mantener programa en ejecución
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Deteniendo servicios...")
        simulation.stop()
        server.shutdown()
        server.server_close()
        print("Servidor y simulación detenidos.")
import os
import json
import threading
import logging
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MSCViewerHandler(SimpleHTTPRequestHandler):
    """Handler for MSC Simulation Viewer"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_dashboard_html().encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if hasattr(self.server, 'simulation_runner'):
                status = self.server.simulation_runner.get_status()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.wfile.write(json.dumps({'error': 'No simulation runner attached'}).encode())
        elif self.path == '/api/graph':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if hasattr(self.server, 'simulation_runner'):
                graph_data = self.server.simulation_runner.get_graph_data()
                self.wfile.write(json.dumps(graph_data).encode())
            else:
                self.wfile.write(json.dumps({'nodes': [], 'edges': []}).encode())
        else:
            # Handle static files or return 404
            super().do_GET()
    
    def generate_dashboard_html(self):
        """Generate the HTML for the MSC Viewer dashboard"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MSC Simulation Viewer</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.23.0/dist/cytoscape.min.js"></script>
            <style>
                body { padding-top: 60px; background-color: #f8f9fa; }
                .graph-container { height: 600px; border: 1px solid #ddd; background-color: #fff; }
                .card { margin-bottom: 20px; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); }
                .status-indicator { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
                .status-running { background-color: #28a745; }
                .status-stopped { background-color: #dc3545; }
                .metric-value { font-size: 1.5rem; font-weight: bold; }
                .metric-title { font-size: 0.9rem; color: #6c757d; }
                .node-info { font-size: 0.9rem; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">
                        <i class="bi bi-braces"></i> MSC Simulation Viewer
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
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-9">
                        <div class="card h-100">
                            <div class="card-header d-flex justify-content-between align-items-center bg-info text-white">
                                <span><i class="bi bi-diagram-3"></i> Knowledge Graph</span>
                                <div>
                                    <button class="btn btn-sm btn-outline-light" id="layout-btn">
                                        <i class="bi bi-grid-3x3"></i> Change Layout
                                    </button>
                                    <button class="btn btn-sm btn-outline-light ms-2" id="reset-view-btn">
                                        <i class="bi bi-arrows-angle-expand"></i> Reset View
                                    </button>
                                </div>
                            </div>
                            <div class="card-body p-0">
                                <div id="graph-container" class="graph-container"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-success text-white">
                                <i class="bi bi-people"></i> Active Agents
                            </div>
                            <div class="card-body p-0" style="max-height: 250px; overflow-y: auto;">
                                <table class="table table-sm table-hover mb-0" id="agents-table">
                                    <thead>
                                        <tr>
                                            <th>ID</th>
                                            <th>Type</th>
                                            <th>Status</th>
                                            <th>Last Activity</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="4" class="text-center">Loading...</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <i class="bi bi-clipboard-data"></i> Node Details
                            </div>
                            <div class="card-body">
                                <div id="node-details">
                                    <p class="text-muted text-center">Select a node to view details</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                // Dashboard controller
                const Dashboard = {
                    state: {
                        isRunning: false,
                        currentStep: 0,
                        selectedNode: null,
                        refreshInterval: null,
                        cy: null,
                        layouts: ['cose', 'circle', 'grid', 'breadthfirst', 'concentric'],
                        currentLayout: 0,
                        forceLayout: false,
                        _initialLayoutDone: false
                    },
                    
                    init: function() {
                        this.setupEventListeners();
                        this.initGraph();
                        this.refreshData();
                        this.startAutoRefresh();
                    },
                    
                    setupEventListeners: function() {
                        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshData());
                        document.getElementById('reset-view-btn').addEventListener('click', () => this.resetGraphView());
                        document.getElementById('layout-btn').addEventListener('click', () => this.changeLayout());
                    },
                    
                    initGraph: function() {
                        this.state.cy = cytoscape({
                            container: document.getElementById('graph-container'),
                            style: [
                                {
                                    selector: 'node',
                                    style: {
                                        'background-color': 'data(color)',
                                        'label': 'data(label)',
                                        'color': '#fff',
                                        'text-outline-color': '#000',
                                        'text-outline-width': 1,
                                        'font-size': 12,
                                        'text-valign': 'center',
                                        'text-halign': 'center',
                                        'width': 'data(size)',
                                        'height': 'data(size)'
                                    }
                                },
                                {
                                    selector: 'edge',
                                    style: {
                                        'width': 'data(width)',
                                        'line-color': 'data(color)',
                                        'curve-style': 'bezier',
                                        'target-arrow-color': 'data(color)',
                                        'target-arrow-shape': 'triangle',
                                        'arrow-scale': 0.8
                                    }
                                }
                            ],
                            layout: {
                                name: this.state.layouts[this.state.currentLayout],
                                fit: true,
                                padding: 30,
                                animate: false,  // Deshabilitar animaciones iniciales
                                randomize: false // Usar posiciones previas cuando sea posible
                            },
                            // Opciones de renderizado optimizadas
                            minZoom: 0.1,
                            maxZoom: 2.5,
                            wheelSensitivity: 0.2,
                            pixelRatio: 'auto',
                            hideEdgesOnViewport: true,  // Ocultar bordes durante zoom/pan
                            hideLabelsOnViewport: true,  // Ocultar etiquetas durante zoom/pan
                            textureOnViewport: true,     // Usar textura durante zoom/pan
                            motionBlur: false            // Deshabilitar motion blur para mejor rendimiento
                        });
                        
                        this.state.cy.on('tap', 'node', event => {
                            const node = event.target;
                            this.showNodeDetails(node.data());
                        });
                        
                        this.state.cy.on('tap', function(event) {
                            if (event.target === this) {
                                // Clicked on background
                                document.getElementById('node-details').innerHTML = 
                                    '<p class="text-muted text-center">Select a node to view details</p>';
                            }
                        });
                    },
                    
                    refreshData: function() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                this.updateStatusDisplay(data);
                            })
                            .catch(error => {
                                console.error('Error fetching status:', error);
                                document.getElementById('status-light').className = 'status-indicator status-stopped';
                                document.getElementById('status-text').textContent = 'Disconnected';
                            });
                            
                        fetch('/api/graph')
                            .then(response => response.json())
                            .then(graphData => {
                                this.updateGraphDisplay(graphData);
                            })
                            .catch(error => {
                                console.error('Error fetching graph data:', error);
                            });
                    },
                    
                    updateStatusDisplay: function(data) {
                        document.getElementById('status-light').className = 
                            `status-indicator ${data.isRunning ? 'status-running' : 'status-stopped'}`;
                        document.getElementById('status-text').textContent = 
                            data.isRunning ? 'Running' : 'Stopped';
                        document.getElementById('step-counter').textContent = data.currentStep || 0;
                        
                        // Update metrics
                        if (data.metadata && data.metadata.filtered) {
                            document.getElementById('node-count').innerHTML = 
                                `${data.nodeCount || 0}<span class="text-muted small"> (mostrando ${data.metadata.shownNodes}/${data.metadata.totalNodes})</span>`;
                            document.getElementById('edge-count').innerHTML = 
                                `${data.edgeCount || 0}<span class="text-muted small"> (mostrando ${data.metadata.shownEdges}/${data.metadata.totalEdges})</span>`;
                        } else {
                            document.getElementById('node-count').textContent = data.nodeCount || 0;
                            document.getElementById('edge-count').textContent = data.edgeCount || 0;
                        }
                        
                        document.getElementById('agent-count').textContent = data.agentCount || 0;
                        document.getElementById('avg-state').textContent = 
                            typeof data.averageState === 'number' ? data.averageState.toFixed(3) : '-';
                            
                        // Update agents table
                        const agentsTable = document.getElementById('agents-table').querySelector('tbody');
                        if (data.agents && data.agents.length > 0) {
                            agentsTable.innerHTML = '';
                            data.agents.forEach(agent => {
                                const row = document.createElement('tr');
                                row.innerHTML = `
                                    <td>${agent.id}</td>
                                    <td>${agent.type}</td>
                                    <td>${agent.status}</td>
                                    <td>${agent.lastActivity || '-'}</td>
                                `;
                                agentsTable.appendChild(row);
                            });
                        } else {
                            agentsTable.innerHTML = '<tr><td colspan="4" class="text-center">No agents data</td></tr>';
                        }
                    },
                    
                    updateGraphDisplay: function(graphData) {
                        if (!graphData.nodes || !graphData.edges) return;
                        
                        const cy = this.state.cy;
                        
                        // Mostrar mensaje si los datos están filtrados
                        if (graphData.metadata && graphData.metadata.filtered) {
                            const container = document.getElementById('graph-container');
                            let noticeEl = container.querySelector('.filter-notice');
                            if (!noticeEl) {
                                noticeEl = document.createElement('div');
                                noticeEl.className = 'filter-notice';
                                noticeEl.style.cssText = 'position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.7);color:white;padding:5px;border-radius:3px;z-index:999;';
                                container.appendChild(noticeEl);
                            }
                            noticeEl.textContent = `Mostrando ${graphData.metadata.shownNodes} de ${graphData.metadata.totalNodes} nodos`;
                        }
                        
                        // Recordar posiciones de nodos existentes
                        const positions = {};
                        cy.nodes().forEach(node => {
                            positions[node.id()] = { x: node.position('x'), y: node.position('y') };
                        });
                        
                        // Batch operations for better performance
                        cy.batch(() => {
                            // Remove elements that are no longer in the data
                            const currentIds = new Set(graphData.nodes.map(n => n.id.toString()));
                            cy.nodes().forEach(node => {
                                if (!currentIds.has(node.id())) {
                                    node.remove();
                                }
                            });
                            
                            // Update or add nodes
                            graphData.nodes.forEach(nodeData => {
                                const id = nodeData.id.toString();
                                let node = cy.getElementById(id);
                                
                                if (node.length > 0) {
                                    // Update existing node
                                    node.data(nodeData);
                                } else {
                                    // Add new node with position
                                    const nodeElement = {
                                        data: nodeData,
                                        position: positions[id] || { x: Math.random() * 800, y: Math.random() * 600 }
                                    };
                                    cy.add({ group: 'nodes', data: nodeData });
                                }
                            });
                            
                            // Update edges similarly
                            cy.edges().remove(); // Remove all edges and re-add for simplicity
                            cy.add(graphData.edges.map(edge => ({ group: 'edges', data: edge })));
                        });
                        
                        // Only run layout if specifically requested or if it's the first load
                        if (this.state.forceLayout || cy.nodes().length <= 50 || !this._initialLayoutDone) {
                            this._initialLayoutDone = true;
                            this.state.forceLayout = false;
                            
                            // Use a more efficient layout for large graphs
                            const layoutName = cy.nodes().length > 100 ? 'grid' : this.state.layouts[this.state.currentLayout];
                            
                            cy.layout({
                                name: layoutName,
                                fit: true,
                                padding: 30,
                                animate: false,
                                randomize: false
                            }).run();
                        }
                    },
                    
                    showNodeDetails: function(nodeData) {
                        const detailsContainer = document.getElementById('node-details');
                        detailsContainer.innerHTML = `
                            <h5>Node ${nodeData.id}</h5>
                            <div class="mb-2">
                                <span class="badge bg-info">State: ${parseFloat(nodeData.state).toFixed(3)}</span>
                            </div>
                            <div class="mb-2">
                                <h6>Content:</h6>
                                <p>${nodeData.content || 'No content'}</p>
                            </div>
                            <div>
                                <h6>Keywords:</h6>
                                <p>${nodeData.keywords ? nodeData.keywords : 'No keywords'}</p>
                            </div>
                        `;
                    },
                    
                    resetGraphView: function() {
                        this.state.cy.fit();
                    },
                    
                    changeLayout: function() {
                        this.state.currentLayout = (this.state.currentLayout + 1) % this.state.layouts.length;
                        this.state.forceLayout = true;
                        
                        const layoutName = this.state.cy.nodes().length > 100 ? 
                            (this.state.layouts[this.state.currentLayout] === 'cose' ? 'grid' : this.state.layouts[this.state.currentLayout]) : 
                            this.state.layouts[this.state.currentLayout];
                        
                        this.state.cy.layout({
                            name: layoutName,
                            fit: true,
                            padding: 30,
                            animate: this.state.cy.nodes().length < 100,
                            animationDuration: 500,
                            randomize: false,
                            nodeDimensionsIncludeLabels: false
                        }).run();
                    },
                    
                    startAutoRefresh: function() {
                        this.state.refreshInterval = setInterval(() => this.refreshData(), 5000);
                    }
                };

                document.addEventListener('DOMContentLoaded', () => Dashboard.init());
            </script>
        </body>
        </html>
        """

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread"""
    allow_reuse_address = True

class MSCViewerServer:
    """Server for MSC Simulation visualization"""
    
    def __init__(self, host='localhost', port=8080):
        """Initialize the visualization server"""
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.simulation_runner = None
        
    def start(self, simulation_runner=None):
        """Start the visualization server"""
        if self.server_thread and self.server_thread.is_alive():
            logging.warning("Server is already running")
            return False
            
        # Store simulation runner reference
        self.simulation_runner = simulation_runner
            
        # Create server
        handler = MSCViewerHandler
        self.server = ThreadedHTTPServer((self.host, self.port), handler)
        self.server.simulation_runner = self.simulation_runner
        
        # Start in a separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        logging.info(f"MSC Viewer server started at http://{self.host}:{self.port}")
        return True
        
    def stop(self):
        """Stop the visualization server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logging.info("MSC Viewer server stopped")
            return True
        return False

class SimulationAdapter:
    """Adapter for connecting MSC Simulation with the viewer"""
    
    def __init__(self, simulation):
        """Initialize with a reference to the MSC simulation"""
        self.simulation = simulation
        self.is_running = False
        self.current_step = 0
        self.start_time = None
    
    def start(self):
        """Start simulation tracking"""
        self.is_running = True
        self.start_time = time.time()
    
    def stop(self):
        """Stop simulation tracking"""
        self.is_running = False
    
    def update(self, step):
        """Update simulation status"""
        self.current_step = step
        
    def get_status(self):
        """Get current simulation status"""
        if not self.simulation:
            return {
                'isRunning': self.is_running,
                'currentStep': self.current_step,
                'error': 'No simulation attached'
            }
            
        # Get status info directly from the simulation runner
        return self.simulation.get_status()
    
    def get_graph_data(self):
        """Get current graph data for visualization with performance optimizations"""
        if not self.simulation or not hasattr(self.simulation, 'graph'):
            return {'nodes': [], 'edges': []}
            
        # Acceder al grafo directamente
        graph = self.simulation.graph
        
        # Limitar el número de nodos y conexiones para visualización
        MAX_NODES = 500  # Límite razonable para visualización fluida
        MAX_EDGES = 1500
        
        nodes = []
        edges = []
        
        # Cmap para colorear nodos - usar precalculados para mayor velocidad
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=1)
        
        # Crear un caché de colores para estados comunes
        color_cache = {}
        for i in range(0, 101):
            state = i / 100
            node_color = cmap(norm(state))
            color_cache[state] = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
        
        # Seleccionar nodos importantes si hay demasiados
        selected_nodes = {}
        if len(graph.nodes) > MAX_NODES:
            # Estrategia: tomar nodos con más conexiones o mayor estado
            nodes_by_importance = sorted(
                graph.nodes.items(),
                key=lambda x: (len(x[1].connections_out) + len(x[1].connections_in)) * x[1].state,
                reverse=True
            )
            selected_nodes = {nid: node for nid, node in nodes_by_importance[:MAX_NODES]}
        else:
            selected_nodes = graph.nodes
            
        # Convertir los nodos seleccionados
        for node_id, node in selected_nodes.items():
            try:
                # Redondear estado a 2 decimales para usar el caché
                rounded_state = round(node.state * 100) / 100
                if rounded_state in color_cache:
                    hex_color = color_cache[rounded_state]
                else:
                    node_color = cmap(norm(node.state))
                    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
                    
                # Simplificar etiquetas para mejor rendimiento
                label = f'{node_id}\n{node.state:.2f}'
                
                # Limitar keywords para mejor rendimiento
                if hasattr(node, 'keywords') and node.keywords:
                    keywords_list = sorted(list(node.keywords))[:5]  # Limitar a 5 keywords
                    keywords_str = ", ".join(keywords_list)
                    if len(keywords_list) < len(node.keywords):
                        keywords_str += f"... (+{len(node.keywords) - len(keywords_list)})"
                else:
                    keywords_str = ""
                    
                # Limitar contenido para mejor rendimiento
                content = getattr(node, 'content', "")
                if len(content) > 100:
                    content = content[:97] + "..."
                    
                nodes.append({
                    'id': str(node_id),
                    'label': label,
                    'state': node.state,
                    'keywords': keywords_str,
                    'content': content,
                    'color': hex_color,
                    'size': 20 + (node.state * 20)  # Tamaño reducido para mejor rendimiento
                })
            except Exception as e:
                # En caso de error, añadir un nodo básico
                nodes.append({
                    'id': str(node_id),
                    'label': f'Node {node_id}',
                    'color': "#6c757d",
                    'size': 20
                })
                
        # Convertir las conexiones con límites de desempeño
        edge_count = 0
        node_ids = set(str(nid) for nid in selected_nodes.keys())
        
        for source_id, node in selected_nodes.items():
            if hasattr(node, 'connections_out'):
                # Ordenar por utilidad para priorizar conexiones importantes
                connections = sorted(
                    ((target, util) for target, util in node.connections_out.items() if str(target) in node_ids),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                # Limitar conexiones por nodo
                max_edges_per_node = min(20, max(3, int(MAX_EDGES / max(1, len(selected_nodes)))))
                
                for target_id, utility in connections[:max_edges_per_node]:
                    if str(target_id) in node_ids and edge_count < MAX_EDGES:
                        # Simplificar anchuras para mejor rendimiento
                        width = 1 + min(3, abs(utility) * 2)
                        
                        # Color basado en signo con menor variedad para mejor rendimiento
                        if utility < -0.3:
                            color = 'red'
                        elif utility > 0.3:
                            color = 'blue'
                        else:
                            color = 'grey'
                            
                        edges.append({
                            'source': str(source_id),
                            'target': str(target_id),
                            'utility': round(utility, 2),  # Redondear para reducir datos
                            'width': width,
                            'color': color
                        })
                        edge_count += 1
                        
                    if edge_count >= MAX_EDGES:
                        break
        
        # Agregar información sobre el filtrado
        metadata = {
            'filtered': len(graph.nodes) > MAX_NODES,
            'totalNodes': len(graph.nodes),
            'totalEdges': sum(len(n.connections_out) for n in graph.nodes.values()),
            'shownNodes': len(nodes),
            'shownEdges': len(edges)
        }
        
        return {'nodes': nodes, 'edges': edges, 'metadata': metadata}

# Main function to run the server standalone
def main():
    """Run the MSC Viewer server standalone"""
    import argparse
    parser = argparse.ArgumentParser(description='MSC Simulation Viewer')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()
    
    server = MSCViewerServer(host=args.host, port=args.port)
    server.start()
    
    try:
        logging.info(f"MSC Viewer server started at http://{args.host}:{args.port}")
        logging.info("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        server.stop()

if __name__ == "__main__":
    main()
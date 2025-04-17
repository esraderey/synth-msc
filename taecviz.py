import os
import sys
import json
import time
import logging
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import socket
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TAECVizHandler(SimpleHTTPRequestHandler):
    """Manejador HTTP personalizado para la visualización de TAEC"""
    
    def __init__(self, *args, **kwargs):
        # El simulation_runner debe obtenerse del servidor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Maneja las solicitudes GET para la API de visualización"""
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/status':
            self.serve_status()
        elif self.path == '/api/graph':
            self.serve_graph()
        elif self.path == '/api/taec':
            self.serve_taec_status()
        elif self.path.startswith('/static/'):
            self.serve_static()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Sirve la página principal del dashboard"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = self.generate_dashboard_html()
        self.wfile.write(html.encode())
    
    def serve_status(self):
        """Sirve información de estado de la simulación"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        status = self.get_simulation_status()
        self.wfile.write(json.dumps(status).encode())
    
    def serve_graph(self):
        """Sirve los datos del grafo en formato JSON"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        graph_data = self.get_graph_data()
        self.wfile.write(json.dumps(graph_data).encode())
    
    def serve_taec_status(self):
        """Sirve información sobre el estado de TAEC"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        taec_data = self.get_taec_data()
        self.wfile.write(json.dumps(taec_data).encode())
    
    def serve_static(self):
        """Sirve archivos estáticos (CSS, JS, etc.)"""
        self.send_response(200)
        content_type = 'text/css' if self.path.endswith('.css') else 'application/javascript'
        self.send_header('Content-type', content_type)
        self.end_headers()
        
        # Aquí simplemente enviamos contenido vacío para evitar errores
        self.wfile.write(b"/* Placeholder */")
    
    def get_simulation_status(self):
        """Obtiene el estado actual de la simulación"""
        if hasattr(self.server, 'simulation_runner') and self.server.simulation_runner:
            try:
                # Si tiene un método get_status, úsalo
                if hasattr(self.server.simulation_runner, 'get_status'):
                    return self.server.simulation_runner.get_status()
                
                # Si no, crea un estado básico
                return {
                    "is_running": self.server.simulation_runner.is_running if hasattr(self.server.simulation_runner, 'is_running') else True,
                    "current_step": self.server.simulation_runner.current_step if hasattr(self.server.simulation_runner, 'current_step') else 0,
                    "node_count": len(self.server.simulation_runner.graph.nodes) if hasattr(self.server.simulation_runner, 'graph') and hasattr(self.server.simulation_runner.graph, 'nodes') else 0,
                    "average_state": "N/A"
                }
            except Exception as e:
                logging.error(f"Error al obtener el estado de la simulación: {e}")
                return {"error": f"Error al obtener estado: {str(e)}"}
        return {"error": "Simulation runner not available"}
    
    def get_graph_data(self):
        """Obtiene los datos del grafo directamente de la instancia de simulación"""
        if not hasattr(self.server, 'simulation_runner') or not self.server.simulation_runner or not hasattr(self.server.simulation_runner, 'graph'):
            return {"nodes": [], "links": []}
        
        # Acceder directamente al grafo de la simulación
        graph = self.server.simulation_runner.graph
        
        nodes = []
        links = []
        
        # Crear un mapa de colores para los estados de los nodos
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=1)
        
        # Convertir nodos a formato de visualización
        for node_id, node in graph.nodes.items():
            try:
                node_color = cmap(norm(node.state))
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in node_color[:3])
                
                node_data = {
                    "id": str(node_id),
                    "label": f"Node {node_id}",
                    "state": node.state,
                    "color": hex_color,
                    "size": 10 + node.state * 30
                }
                
                if hasattr(node, 'keywords') and node.keywords:
                    node_data["keywords"] = list(node.keywords)
                
                nodes.append(node_data)
                
                # Convertir conexiones a enlaces
                for target_id, utility in node.connections_out.items():
                    if target_id in graph.nodes:
                        link_color = 'red' if utility < 0 else 'blue'
                        links.append({
                            "source": str(node_id),
                            "target": str(target_id),
                            "value": abs(utility),
                            "color": link_color
                        })
            except Exception as e:
                logging.error(f"Error processing node {node_id}: {e}")
        
        return {"nodes": nodes, "links": links}
    
    def get_taec_data(self):
        """Obtiene información sobre el estado de TAEC"""
        taec_data = {
            "active": False,
            "modules": [],
            "diagnostics": [],
            "optimizations": []
        }
        
        if hasattr(self.server, 'simulation_runner') and self.server.simulation_runner:
            # Buscar agentes TAEC en la simulación
            taec_agents = [a for a in self.server.simulation_runner.agents 
                          if hasattr(a, '__class__') and a.__class__.__name__ == 'SelfEvolvingSystem']
            
            if taec_agents:
                taec = taec_agents[0]
                taec_data["active"] = True
                
                # Información sobre módulos generados
                if hasattr(taec, 'generated_modules'):
                    taec_data["modules"] = [
                        {"name": name, "timestamp": timestamp}
                        for name, timestamp in taec.generated_modules.items()
                    ]
                
                # Información sobre diagnósticos realizados
                if hasattr(taec, 'diagnostics_history'):
                    taec_data["diagnostics"] = taec.diagnostics_history
                
                # Información sobre optimizaciones realizadas
                if hasattr(taec, 'optimizations_history'):
                    taec_data["optimizations"] = taec.optimizations_history
        
        return taec_data
    
    def generate_dashboard_html(self):
        """Genera el HTML para el dashboard"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TAECViz - MSC Framework Visualization</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 0; 
                    background-color: #f5f5f5;
                }
                .container {
                    display: flex;
                    height: 100vh;
                }
                .sidebar {
                    width: 300px;
                    background-color: #2c3e50;
                    color: white;
                    padding: 15px;
                    box-sizing: border-box;
                    overflow-y: auto;
                }
                .main {
                    flex-grow: 1;
                    padding: 15px;
                    box-sizing: border-box;
                    overflow: hidden;
                }
                .graph-container {
                    width: 100%;
                    height: 100%;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                h1, h2, h3 {
                    margin-top: 0;
                    color: #ecf0f1;
                }
                .status-panel {
                    background-color: #34495e;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }
                .taec-panel {
                    background-color: #34495e;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 15px;
                }
                .stat-item {
                    margin-bottom: 8px;
                }
                .stat-label {
                    color: #bdc3c7;
                    font-size: 12px;
                }
                .stat-value {
                    font-weight: bold;
                    font-size: 16px;
                }
            </style>
            <!-- Añadir D3.js para visualización -->
            <script src="https://d3js.org/d3.v7.min.js"></script>
        </head>
        <body>
            <div class="container">
                <div class="sidebar">
                    <h1>TAECViz</h1>
                    <p>Visualización avanzada para el framework MSC</p>
                    
                    <div class="status-panel">
                        <h2>Estado de la Simulación</h2>
                        <div id="simulation-stats"></div>
                    </div>
                    
                    <div class="taec-panel">
                        <h2>TAEC Status</h2>
                        <div id="taec-stats"></div>
                    </div>
                </div>
                
                <div class="main">
                    <div class="graph-container" id="graph-view"></div>
                </div>
            </div>
            
            <script>
                // Función para actualizar los datos
                function updateData() {
                    // Actualizar estado de simulación
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            let statsHtml = '';
                            for (const [key, value] of Object.entries(data)) {
                                statsHtml += `
                                    <div class="stat-item">
                                        <div class="stat-label">${key}</div>
                                        <div class="stat-value">${value}</div>
                                    </div>
                                `;
                            }
                            document.getElementById('simulation-stats').innerHTML = statsHtml;
                        })
                        .catch(error => console.error('Error fetching status:', error));
                    
                    // Actualizar estado de TAEC
                    fetch('/api/taec')
                        .then(response => response.json())
                        .then(data => {
                            let taecHtml = '';
                            taecHtml += `<div class="stat-item">
                                <div class="stat-label">TAEC Active</div>
                                <div class="stat-value">${data.active ? 'Yes' : 'No'}</div>
                            </div>`;
                            
                            taecHtml += `<div class="stat-item">
                                <div class="stat-label">Modules Generated</div>
                                <div class="stat-value">${data.modules.length}</div>
                            </div>`;
                            
                            taecHtml += `<div class="stat-item">
                                <div class="stat-label">Recent Diagnostics</div>
                                <div class="stat-value">${data.diagnostics.length}</div>
                            </div>`;
                            
                            document.getElementById('taec-stats').innerHTML = taecHtml;
                        })
                        .catch(error => console.error('Error fetching TAEC data:', error));
                    
                    // Actualizar visualización del grafo
                    fetch('/api/graph')
                        .then(response => response.json())
                        .then(data => updateGraph(data))
                        .catch(error => console.error('Error fetching graph data:', error));
                }
                
                // Función para actualizar el grafo
                function updateGraph(data) {
                    const width = document.getElementById('graph-view').clientWidth;
                    const height = document.getElementById('graph-view').clientHeight;
                    
                    document.getElementById('graph-view').innerHTML = '';
                    
                    // Crear el SVG
                    const svg = d3.select('#graph-view')
                        .append('svg')
                        .attr('width', width)
                        .attr('height', height);
                    
                    // Crear la simulación de fuerzas
                    const simulation = d3.forceSimulation(data.nodes)
                        .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
                        .force('charge', d3.forceManyBody().strength(-400))
                        .force('center', d3.forceCenter(width / 2, height / 2))
                        .on('tick', ticked);
                    
                    // Crear los enlaces
                    const link = svg.append('g')
                        .selectAll('line')
                        .data(data.links)
                        .enter()
                        .append('line')
                        .attr('stroke-width', d => Math.sqrt(d.value) * 2)
                        .attr('stroke', d => d.color || '#999');
                    
                    // Crear los nodos
                    const node = svg.append('g')
                        .selectAll('circle')
                        .data(data.nodes)
                        .enter()
                        .append('circle')
                        .attr('r', d => d.size || 10)
                        .attr('fill', d => d.color || '#69b3a2')
                        .call(d3.drag()
                            .on('start', dragstarted)
                            .on('drag', dragged)
                            .on('end', dragended));
                    
                    // Añadir etiquetas a los nodos
                    const label = svg.append('g')
                        .selectAll('text')
                        .data(data.nodes)
                        .enter()
                        .append('text')
                        .text(d => d.label)
                        .attr('font-size', 10)
                        .attr('dx', 12)
                        .attr('dy', 4);
                    
                    // Función para actualizar posiciones
                    function ticked() {
                        link
                            .attr('x1', d => d.source.x)
                            .attr('y1', d => d.source.y)
                            .attr('x2', d => d.target.x)
                            .attr('y2', d => d.target.y);
                        
                        node
                            .attr('cx', d => d.x)
                            .attr('cy', d => d.y);
                        
                        label
                            .attr('x', d => d.x)
                            .attr('y', d => d.y);
                    }
                    
                    // Funciones para el arrastre de nodos
                    function dragstarted(event, d) {
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }
                    
                    function dragged(event, d) {
                        d.fx = event.x;
                        d.fy = event.y;
                    }
                    
                    function dragended(event, d) {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }
                }
                
                // Actualizar datos cada 2 segundos
                updateData();
                setInterval(updateData, 2000);
            </script>
        </body>
        </html>
        """

class CustomHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, simulation_runner=None):
        self.simulation_runner = simulation_runner
        super().__init__(server_address, RequestHandlerClass)

class TAECVizServer:
    """Servidor para la visualización de TAEC"""
    
    def __init__(self, simulation_runner, host='localhost', port=8082):
        self.simulation_runner = simulation_runner
        self.host = host
        self.port = self.find_available_port(port)
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def find_available_port(self, start_port):
        """Encuentra un puerto disponible a partir del puerto inicial"""
        port = start_port
        max_port = start_port + 100  # Buscar hasta 100 puertos
        
        while port < max_port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((self.host, port)) != 0:
                    return port
            port += 1
        
        logging.warning(f"No se encontró un puerto disponible entre {start_port} y {max_port}")
        return start_port  # Devolver el puerto original y esperar lo mejor
    
    def start(self):
        """Inicia el servidor de visualización"""
        if self.is_running:
            return
        
        # Crear un handler personalizado que tenga acceso al simulation_runner
        handler_class = type('CustomTAECVizHandler', 
                            (TAECVizHandler,), 
                            {})
        
        # Iniciar el servidor en un hilo separado
        self.server = CustomHTTPServer((self.host, self.port), handler_class, simulation_runner=self.simulation_runner)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.is_running = True
        
        url = f"http://{self.host}:{self.port}"
        logging.info(f"TAECViz iniciado en {url}")
        
        # Abrir el navegador automáticamente
        try:
            webbrowser.open(url)
        except:
            logging.info(f"Por favor, abre manualmente: {url}")
    
    def stop(self):
        """Detiene el servidor de visualización"""
        if not self.is_running:
            return
        
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        self.is_running = False
        logging.info("TAECViz detenido")


# Función para integrar TAECViz con la simulación
def integrate_taecviz_with_simulation(simulation_runner):
    """Integra TAECViz con la simulación MSC"""
    viz_server = TAECVizServer(simulation_runner)
    viz_server.start()
    return viz_server


if __name__ == "__main__":
    print("Este módulo debe ser importado por msc_simulation.py para usarse.")
    print("Ejecución de ejemplo:")
    print("  from taecviz import integrate_taecviz_with_simulation")
    print("  viz_server = integrate_taecviz_with_simulation(simulation_runner)")
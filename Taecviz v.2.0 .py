#!/usr/bin/env python3
"""
TAECViz v2.0 - Sistema de Visualización Avanzado para MSC Framework
Características principales:
- Dashboard unificado para MSC + TAEC + SCED
- Visualización 3D del grafo con WebGL
- Monitoreo en tiempo real de evoluciones
- Visualización de memoria cuántica
- Análisis de métricas con ML
- Control interactivo del sistema
- Exportación de datos y reportes
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import webbrowser
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.gen
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles

# Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURACIÓN ===

@dataclass
class TAECVizConfig:
    """Configuración para TAECViz"""
    host: str = "localhost"
    port: int = 8888
    msc_api_url: str = "http://localhost:5000"
    update_interval: float = 1.0  # segundos
    max_graph_nodes: int = 1000
    enable_3d: bool = True
    enable_quantum_viz: bool = True
    enable_ml_analytics: bool = True
    theme: str = "dark"  # dark, light, auto
    
class VizMetricsCollector:
    """Recolector de métricas para visualización"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.alerts = deque(maxlen=100)
        self.predictions = {}
        
    def add_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Añade una métrica"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def add_alert(self, level: str, message: str, details: Optional[Dict] = None):
        """Añade una alerta"""
        self.alerts.append({
            'level': level,  # info, warning, error, critical
            'message': message,
            'details': details or {},
            'timestamp': time.time()
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de métricas"""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                recent_values = [v['value'] for v in list(values)[-100:]]
                summary[name] = {
                    'current': values[-1]['value'],
                    'avg': np.mean(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'trend': self._calculate_trend(recent_values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendencia de valores"""
        if len(values) < 3:
            return 'stable'
        
        # Regresión lineal simple
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'

# === HANDLERS WEB ===

class BaseHandler(tornado.web.RequestHandler):
    """Handler base con funcionalidad común"""
    
    def set_default_headers(self):
        """Configura headers CORS"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, Content-Type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")
    
    def options(self):
        """Maneja requests OPTIONS para CORS"""
        self.set_status(204)
        self.finish()
    
    def write_json(self, data: Any):
        """Escribe respuesta JSON"""
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))

class DashboardHandler(BaseHandler):
    """Sirve el dashboard principal"""
    
    async def get(self):
        """Sirve la página HTML del dashboard"""
        self.set_header("Content-Type", "text/html")
        html = self.generate_dashboard_html()
        self.write(html)
    
    def generate_dashboard_html(self) -> str:
        """Genera el HTML del dashboard mejorado"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TAECViz 2.0 - MSC Framework Dashboard</title>
    
    <!-- CSS Framework -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    
    <!-- Librerías de visualización -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.2.1/dist/chart.umd.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        :root {
            --primary-color: #3B82F6;
            --secondary-color: #10B981;
            --danger-color: #EF4444;
            --warning-color: #F59E0B;
            --dark-bg: #1F2937;
            --darker-bg: #111827;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--darker-bg);
            color: #F3F4F6;
        }
        
        .glass-morphism {
            background: rgba(31, 41, 55, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(75, 85, 99, 0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        }
        
        .tab-active {
            background: var(--primary-color);
            color: white;
        }
        
        .alert-enter {
            animation: slideInRight 0.3s ease-out;
        }
        
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .graph-container {
            position: relative;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        
        #graph-3d-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        
        .loading-spinner {
            border: 4px solid rgba(59, 130, 246, 0.1);
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .quantum-visualization {
            background: radial-gradient(circle at center, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        .evolution-timeline {
            position: relative;
            padding: 20px 0;
        }
        
        .evolution-node {
            position: absolute;
            width: 12px;
            height: 12px;
            background: var(--secondary-color);
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .evolution-node:hover {
            transform: scale(1.5);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.6);
        }
        
        .console-output {
            background: #000;
            color: #0F0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            border: 1px solid #333;
        }
        
        .heatmap-cell {
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        .heatmap-cell:hover {
            stroke: white;
            stroke-width: 2;
        }
    </style>
</head>
<body class="bg-gray-900">
    <!-- Header -->
    <header class="glass-morphism px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
                <i class="fas fa-brain text-3xl text-blue-500"></i>
                <h1 class="text-2xl font-bold">TAECViz 2.0</h1>
            </div>
            <span class="text-sm text-gray-400">MSC Framework Dashboard</span>
        </div>
        
        <div class="flex items-center space-x-4">
            <div class="flex items-center space-x-2">
                <div id="connection-status" class="w-3 h-3 rounded-full bg-green-500"></div>
                <span class="text-sm">Connected</span>
            </div>
            
            <button onclick="toggleTheme()" class="p-2 rounded hover:bg-gray-700">
                <i class="fas fa-moon"></i>
            </button>
            
            <button onclick="toggleFullscreen()" class="p-2 rounded hover:bg-gray-700">
                <i class="fas fa-expand"></i>
            </button>
        </div>
    </header>
    
    <!-- Main Content -->
    <div class="flex h-screen pt-16">
        <!-- Sidebar -->
        <aside class="w-64 glass-morphism p-4 overflow-y-auto">
            <div class="space-y-6">
                <!-- System Status -->
                <div>
                    <h3 class="text-lg font-semibold mb-3 flex items-center">
                        <i class="fas fa-heartbeat mr-2 text-green-500"></i>
                        System Status
                    </h3>
                    <div id="system-status" class="space-y-2"></div>
                </div>
                
                <!-- Quick Actions -->
                <div>
                    <h3 class="text-lg font-semibold mb-3 flex items-center">
                        <i class="fas fa-bolt mr-2 text-yellow-500"></i>
                        Quick Actions
                    </h3>
                    <div class="space-y-2">
                        <button onclick="triggerEvolution()" class="w-full px-3 py-2 bg-blue-600 rounded hover:bg-blue-700 transition">
                            <i class="fas fa-dna mr-2"></i>Trigger Evolution
                        </button>
                        <button onclick="createCheckpoint()" class="w-full px-3 py-2 bg-green-600 rounded hover:bg-green-700 transition">
                            <i class="fas fa-save mr-2"></i>Create Checkpoint
                        </button>
                        <button onclick="analyzeSystem()" class="w-full px-3 py-2 bg-purple-600 rounded hover:bg-purple-700 transition">
                            <i class="fas fa-microscope mr-2"></i>Analyze System
                        </button>
                    </div>
                </div>
                
                <!-- Metrics Summary -->
                <div>
                    <h3 class="text-lg font-semibold mb-3 flex items-center">
                        <i class="fas fa-chart-line mr-2 text-blue-500"></i>
                        Key Metrics
                    </h3>
                    <div id="metrics-summary" class="space-y-3"></div>
                </div>
                
                <!-- Active Agents -->
                <div>
                    <h3 class="text-lg font-semibold mb-3 flex items-center">
                        <i class="fas fa-robot mr-2 text-purple-500"></i>
                        Active Agents
                    </h3>
                    <div id="active-agents" class="space-y-2"></div>
                </div>
            </div>
        </aside>
        
        <!-- Main Panel -->
        <main class="flex-1 p-4 overflow-hidden">
            <!-- Tabs -->
            <div class="flex space-x-1 mb-4">
                <button onclick="switchTab('graph')" id="tab-graph" class="tab-active px-4 py-2 rounded-t transition">
                    <i class="fas fa-project-diagram mr-2"></i>Graph View
                </button>
                <button onclick="switchTab('quantum')" id="tab-quantum" class="px-4 py-2 rounded-t hover:bg-gray-700 transition">
                    <i class="fas fa-atom mr-2"></i>Quantum Memory
                </button>
                <button onclick="switchTab('evolution')" id="tab-evolution" class="px-4 py-2 rounded-t hover:bg-gray-700 transition">
                    <i class="fas fa-dna mr-2"></i>Evolution
                </button>
                <button onclick="switchTab('analytics')" id="tab-analytics" class="px-4 py-2 rounded-t hover:bg-gray-700 transition">
                    <i class="fas fa-chart-bar mr-2"></i>Analytics
                </button>
                <button onclick="switchTab('console')" id="tab-console" class="px-4 py-2 rounded-t hover:bg-gray-700 transition">
                    <i class="fas fa-terminal mr-2"></i>Console
                </button>
            </div>
            
            <!-- Tab Content -->
            <div class="glass-morphism rounded-lg h-full p-4">
                <!-- Graph View -->
                <div id="content-graph" class="h-full">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold">Knowledge Graph Visualization</h2>
                        <div class="flex space-x-2">
                            <button onclick="toggleGraphMode()" class="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600">
                                <i class="fas fa-cube mr-1"></i>3D Mode
                            </button>
                            <button onclick="resetGraphView()" class="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600">
                                <i class="fas fa-undo mr-1"></i>Reset
                            </button>
                            <select id="graph-layout" onchange="changeGraphLayout()" class="px-3 py-1 bg-gray-700 rounded">
                                <option value="force">Force Layout</option>
                                <option value="hierarchical">Hierarchical</option>
                                <option value="circular">Circular</option>
                                <option value="grid">Grid</option>
                            </select>
                        </div>
                    </div>
                    <div class="graph-container" style="height: calc(100% - 60px);">
                        <div id="graph-3d-container"></div>
                        <svg id="graph-2d-container" width="100%" height="100%"></svg>
                    </div>
                </div>
                
                <!-- Quantum Memory View -->
                <div id="content-quantum" class="h-full hidden">
                    <div class="grid grid-cols-2 gap-4 h-full">
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Quantum States</h3>
                            <div id="quantum-states" class="quantum-visualization rounded p-4 h-96 overflow-y-auto"></div>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Entanglement Network</h3>
                            <div id="entanglement-network" class="h-96"></div>
                        </div>
                    </div>
                    <div class="mt-4">
                        <h3 class="text-lg font-semibold mb-3">Memory Heatmap</h3>
                        <div id="memory-heatmap" class="h-48"></div>
                    </div>
                </div>
                
                <!-- Evolution View -->
                <div id="content-evolution" class="h-full hidden">
                    <div class="space-y-4">
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Evolution Timeline</h3>
                            <div class="evolution-timeline bg-gray-800 rounded p-4 h-32 relative" id="evolution-timeline"></div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Fitness Progress</h3>
                                <canvas id="fitness-chart" height="200"></canvas>
                            </div>
                            <div>
                                <h3 class="text-lg font-semibold mb-3">Evolution Strategies</h3>
                                <div id="strategy-distribution"></div>
                            </div>
                        </div>
                        
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Generated Code</h3>
                            <div id="generated-code" class="console-output"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Analytics View -->
                <div id="content-analytics" class="h-full hidden">
                    <div class="grid grid-cols-2 gap-4 h-full">
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Performance Metrics</h3>
                            <div id="performance-metrics"></div>
                        </div>
                        <div>
                            <h3 class="text-lg font-semibold mb-3">Predictions</h3>
                            <div id="predictions-chart"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Console View -->
                <div id="content-console" class="h-full hidden">
                    <div class="h-full flex flex-col">
                        <div class="flex-1 console-output" id="console-output"></div>
                        <div class="mt-2">
                            <input type="text" id="console-input" class="w-full px-3 py-2 bg-gray-800 rounded" 
                                   placeholder="Enter MSC-Lang command..." onkeypress="handleConsoleInput(event)">
                        </div>
                    </div>
                </div>
            </div>
        </main>
        
        <!-- Right Panel - Alerts & Notifications -->
        <aside class="w-80 glass-morphism p-4 overflow-y-auto">
            <h3 class="text-lg font-semibold mb-3 flex items-center">
                <i class="fas fa-bell mr-2 text-yellow-500"></i>
                Alerts & Notifications
            </h3>
            <div id="alerts-container" class="space-y-2"></div>
        </aside>
    </div>
    
    <!-- Floating Action Button -->
    <div class="fixed bottom-8 right-8">
        <button onclick="openCommandPalette()" 
                class="w-14 h-14 bg-blue-600 rounded-full shadow-lg hover:bg-blue-700 transition transform hover:scale-110">
            <i class="fas fa-terminal text-white"></i>
        </button>
    </div>
    
    <!-- Command Palette Modal -->
    <div id="command-palette" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-gray-800 rounded-lg p-6 w-96 max-w-full">
            <h3 class="text-lg font-semibold mb-4">Command Palette</h3>
            <input type="text" id="command-search" class="w-full px-3 py-2 bg-gray-700 rounded mb-4" 
                   placeholder="Type a command..." oninput="filterCommands()">
            <div id="command-list" class="space-y-2 max-h-64 overflow-y-auto"></div>
            <div class="mt-4 flex justify-end">
                <button onclick="closeCommandPalette()" class="px-4 py-2 bg-gray-600 rounded hover:bg-gray-700">
                    Cancel
                </button>
            </div>
        </div>
    </div>
    
    <script>
        // === GLOBAL STATE ===
        let ws = null;
        let currentTab = 'graph';
        let graphMode = '2d';
        let scene, camera, renderer; // Three.js
        let simulation = null; // D3 force simulation
        let metricsCharts = {};
        let consoleHistory = [];
        let currentHistoryIndex = -1;
        
        // === WEBSOCKET CONNECTION ===
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateConnectionStatus(true);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                // Reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'metrics_update':
                    updateMetrics(data.metrics);
                    break;
                case 'graph_update':
                    updateGraph(data.graph);
                    break;
                case 'evolution_update':
                    updateEvolution(data.evolution);
                    break;
                case 'alert':
                    showAlert(data.alert);
                    break;
                case 'quantum_update':
                    updateQuantumView(data.quantum);
                    break;
                case 'console_output':
                    appendConsoleOutput(data.output);
                    break;
            }
        }
        
        // === UI FUNCTIONS ===
        function switchTab(tabName) {
            // Hide all content
            document.querySelectorAll('[id^="content-"]').forEach(el => {
                el.classList.add('hidden');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('[id^="tab-"]').forEach(el => {
                el.classList.remove('tab-active');
                el.classList.add('hover:bg-gray-700');
            });
            
            // Show selected content and activate tab
            document.getElementById(`content-${tabName}`).classList.remove('hidden');
            document.getElementById(`tab-${tabName}`).classList.add('tab-active');
            document.getElementById(`tab-${tabName}`).classList.remove('hover:bg-gray-700');
            
            currentTab = tabName;
            
            // Initialize tab-specific content
            if (tabName === 'graph' && !simulation) {
                initializeGraphVisualization();
            } else if (tabName === 'evolution') {
                initializeEvolutionCharts();
            } else if (tabName === 'analytics') {
                initializeAnalytics();
            }
        }
        
        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connection-status');
            if (connected) {
                statusEl.className = 'w-3 h-3 rounded-full bg-green-500';
            } else {
                statusEl.className = 'w-3 h-3 rounded-full bg-red-500';
            }
        }
        
        function showAlert(alert) {
            const container = document.getElementById('alerts-container');
            const alertEl = document.createElement('div');
            
            const bgColor = {
                'info': 'bg-blue-600',
                'warning': 'bg-yellow-600',
                'error': 'bg-red-600',
                'success': 'bg-green-600'
            }[alert.level] || 'bg-gray-600';
            
            alertEl.className = `p-3 rounded ${bgColor} alert-enter`;
            alertEl.innerHTML = `
                <div class="flex items-start justify-between">
                    <div>
                        <div class="font-semibold">${alert.message}</div>
                        ${alert.details ? `<div class="text-sm mt-1">${alert.details}</div>` : ''}
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-2">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            container.insertBefore(alertEl, container.firstChild);
            
            // Auto-remove after 10 seconds
            setTimeout(() => alertEl.remove(), 10000);
        }
        
        // === GRAPH VISUALIZATION ===
        function initializeGraphVisualization() {
            if (graphMode === '3d') {
                initialize3DGraph();
            } else {
                initialize2DGraph();
            }
        }
        
        function initialize3DGraph() {
            const container = document.getElementById('graph-3d-container');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111827);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
            camera.position.z = 100;
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Controls
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        }
        
        function initialize2DGraph() {
            const svg = d3.select('#graph-2d-container');
            const width = svg.node().clientWidth;
            const height = svg.node().clientHeight;
            
            // Clear existing
            svg.selectAll('*').remove();
            
            // Create groups
            const g = svg.append('g');
            
            // Zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });
            
            svg.call(zoom);
            
            // Force simulation
            simulation = d3.forceSimulation()
                .force('link', d3.forceLink().id(d => d.id).distance(50))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(20));
        }
        
        function updateGraph(graphData) {
            if (!graphData || !graphData.nodes) return;
            
            if (graphMode === '3d') {
                update3DGraph(graphData);
            } else {
                update2DGraph(graphData);
            }
            
            // Update sidebar stats
            updateGraphStats(graphData);
        }
        
        function update2DGraph(graphData) {
            const svg = d3.select('#graph-2d-container');
            const g = svg.select('g');
            
            // Update links
            const links = g.selectAll('.link')
                .data(graphData.links, d => `${d.source}-${d.target}`);
            
            links.exit().remove();
            
            const linksEnter = links.enter()
                .append('line')
                .attr('class', 'link')
                .style('stroke', d => d.value > 0 ? '#3B82F6' : '#EF4444')
                .style('stroke-width', d => Math.abs(d.value) * 3)
                .style('opacity', 0.6);
            
            // Update nodes
            const nodes = g.selectAll('.node')
                .data(graphData.nodes, d => d.id);
            
            nodes.exit().remove();
            
            const nodesEnter = nodes.enter()
                .append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
            nodesEnter.append('circle')
                .attr('r', d => 5 + d.state * 20)
                .style('fill', d => d3.interpolateViridis(d.state))
                .style('stroke', '#fff')
                .style('stroke-width', 2);
            
            nodesEnter.append('text')
                .text(d => d.label || `Node ${d.id}`)
                .attr('x', 0)
                .attr('y', -20)
                .style('text-anchor', 'middle')
                .style('fill', '#fff')
                .style('font-size', '12px');
            
            // Tooltip
            nodesEnter
                .on('mouseover', function(event, d) {
                    showNodeTooltip(event, d);
                })
                .on('mouseout', hideTooltip);
            
            // Update simulation
            if (simulation) {
                simulation.nodes(graphData.nodes);
                simulation.force('link').links(graphData.links);
                simulation.alpha(0.3).restart();
                
                simulation.on('tick', () => {
                    g.selectAll('.link')
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    g.selectAll('.node')
                        .attr('transform', d => `translate(${d.x},${d.y})`);
                });
            }
        }
        
        function update3DGraph(graphData) {
            // Clear existing objects
            while(scene.children.length > 2) { // Keep lights
                scene.remove(scene.children[2]);
            }
            
            // Create node geometry
            const nodeGeometry = new THREE.SphereGeometry(1, 32, 32);
            
            // Create nodes
            const nodeObjects = {};
            graphData.nodes.forEach(node => {
                const material = new THREE.MeshPhongMaterial({
                    color: new THREE.Color(d3.interpolateViridis(node.state)),
                    emissive: new THREE.Color(0x0000ff),
                    emissiveIntensity: node.state * 0.3
                });
                
                const mesh = new THREE.Mesh(nodeGeometry, material);
                mesh.scale.setScalar(1 + node.state * 2);
                
                // Random initial position
                mesh.position.set(
                    (Math.random() - 0.5) * 100,
                    (Math.random() - 0.5) * 100,
                    (Math.random() - 0.5) * 100
                );
                
                scene.add(mesh);
                nodeObjects[node.id] = mesh;
            });
            
            // Create edges
            graphData.links.forEach(link => {
                const sourceNode = nodeObjects[link.source];
                const targetNode = nodeObjects[link.target];
                
                if (sourceNode && targetNode) {
                    const geometry = new THREE.BufferGeometry().setFromPoints([
                        sourceNode.position,
                        targetNode.position
                    ]);
                    
                    const material = new THREE.LineBasicMaterial({
                        color: link.value > 0 ? 0x3B82F6 : 0xEF4444,
                        opacity: 0.6,
                        transparent: true
                    });
                    
                    const line = new THREE.Line(geometry, material);
                    scene.add(line);
                }
            });
        }
        
        // === QUANTUM VISUALIZATION ===
        function updateQuantumView(quantumData) {
            if (!quantumData) return;
            
            // Update quantum states
            const statesContainer = document.getElementById('quantum-states');
            statesContainer.innerHTML = '';
            
            quantumData.states.forEach(state => {
                const stateEl = document.createElement('div');
                stateEl.className = 'mb-4 p-3 bg-gray-800 rounded';
                stateEl.innerHTML = `
                    <div class="flex justify-between items-center mb-2">
                        <span class="font-semibold">${state.address}</span>
                        <span class="text-sm text-gray-400">Coherence: ${state.coherence.toFixed(3)}</span>
                    </div>
                    <div class="grid grid-cols-4 gap-2 text-xs">
                        ${state.amplitudes.map((amp, i) => `
                            <div class="text-center">
                                <div>|${i}⟩</div>
                                <div>${amp.toFixed(3)}</div>
                            </div>
                        `).join('')}
                    </div>
                `;
                statesContainer.appendChild(stateEl);
            });
            
            // Update entanglement network
            drawEntanglementNetwork(quantumData.entanglements);
            
            // Update memory heatmap
            drawMemoryHeatmap(quantumData.memoryMap);
        }
        
        function drawEntanglementNetwork(entanglements) {
            const container = document.getElementById('entanglement-network');
            const width = container.clientWidth;
            const height = 384; // h-96 = 24rem = 384px
            
            // Clear existing
            container.innerHTML = '';
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // Create quantum-style visualization
            const g = svg.append('g');
            
            // Nodes represent quantum states
            const nodes = entanglements.nodes.map((node, i) => ({
                id: node.id,
                x: width / 2 + Math.cos(i * 2 * Math.PI / entanglements.nodes.length) * 150,
                y: height / 2 + Math.sin(i * 2 * Math.PI / entanglements.nodes.length) * 150
            }));
            
            // Draw entanglement connections
            entanglements.links.forEach(link => {
                const source = nodes.find(n => n.id === link.source);
                const target = nodes.find(n => n.id === link.target);
                
                if (source && target) {
                    // Quantum-style curved connection
                    const path = g.append('path')
                        .attr('d', `M ${source.x} ${source.y} Q ${width/2} ${height/2} ${target.x} ${target.y}`)
                        .style('fill', 'none')
                        .style('stroke', `rgba(139, 92, 246, ${link.strength})`)
                        .style('stroke-width', 2);
                    
                    // Animate the connection
                    path.style('stroke-dasharray', '5,5')
                        .style('stroke-dashoffset', 0)
                        .style('animation', 'dash 2s linear infinite');
                }
            });
            
            // Draw nodes
            nodes.forEach(node => {
                const nodeG = g.append('g')
                    .attr('transform', `translate(${node.x},${node.y})`);
                
                // Outer glow
                nodeG.append('circle')
                    .attr('r', 15)
                    .style('fill', 'none')
                    .style('stroke', 'rgba(139, 92, 246, 0.3)')
                    .style('stroke-width', 20)
                    .style('filter', 'blur(10px)');
                
                // Inner circle
                nodeG.append('circle')
                    .attr('r', 8)
                    .style('fill', '#8B5CF6')
                    .style('stroke', '#fff')
                    .style('stroke-width', 2);
                
                // Label
                nodeG.append('text')
                    .text(node.id)
                    .attr('y', -15)
                    .style('text-anchor', 'middle')
                    .style('fill', '#fff')
                    .style('font-size', '12px');
            });
        }
        
        function drawMemoryHeatmap(memoryMap) {
            const container = document.getElementById('memory-heatmap');
            const width = container.clientWidth;
            const height = 192; // h-48 = 12rem = 192px
            
            // Clear existing
            container.innerHTML = '';
            
            if (!memoryMap || memoryMap.length === 0) return;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const rows = memoryMap.length;
            const cols = memoryMap[0].length;
            const cellWidth = width / cols;
            const cellHeight = height / rows;
            
            // Color scale
            const colorScale = d3.scaleSequential(d3.interpolateInferno)
                .domain([0, 1]);
            
            // Draw heatmap
            memoryMap.forEach((row, i) => {
                row.forEach((value, j) => {
                    svg.append('rect')
                        .attr('class', 'heatmap-cell')
                        .attr('x', j * cellWidth)
                        .attr('y', i * cellHeight)
                        .attr('width', cellWidth - 1)
                        .attr('height', cellHeight - 1)
                        .style('fill', colorScale(value))
                        .on('mouseover', function(event) {
                            d3.select(this).style('stroke', 'white').style('stroke-width', 2);
                            showTooltip(event, `Memory[${i},${j}]: ${value.toFixed(3)}`);
                        })
                        .on('mouseout', function() {
                            d3.select(this).style('stroke', 'none');
                            hideTooltip();
                        });
                });
            });
        }
        
        // === EVOLUTION VISUALIZATION ===
        function updateEvolution(evolutionData) {
            if (!evolutionData) return;
            
            // Update timeline
            updateEvolutionTimeline(evolutionData.history);
            
            // Update fitness chart
            updateFitnessChart(evolutionData.fitness);
            
            // Update strategy distribution
            updateStrategyDistribution(evolutionData.strategies);
            
            // Update generated code
            if (evolutionData.generatedCode) {
                const codeContainer = document.getElementById('generated-code');
                codeContainer.innerHTML = `<pre>${evolutionData.generatedCode}</pre>`;
            }
        }
        
        function updateEvolutionTimeline(history) {
            const timeline = document.getElementById('evolution-timeline');
            timeline.innerHTML = '';
            
            if (!history || history.length === 0) return;
            
            const width = timeline.clientWidth;
            const nodeWidth = width / history.length;
            
            history.forEach((event, i) => {
                const node = document.createElement('div');
                node.className = 'evolution-node';
                node.style.left = `${i * nodeWidth + nodeWidth/2}px`;
                node.style.top = '50%';
                node.style.transform = 'translate(-50%, -50%)';
                
                // Color based on success
                if (event.success) {
                    node.style.background = '#10B981';
                } else {
                    node.style.background = '#EF4444';
                }
                
                node.onclick = () => showEvolutionDetails(event);
                
                timeline.appendChild(node);
            });
        }
        
        function updateFitnessChart(fitnessData) {
            const ctx = document.getElementById('fitness-chart').getContext('2d');
            
            if (metricsCharts.fitness) {
                metricsCharts.fitness.destroy();
            }
            
            metricsCharts.fitness = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: fitnessData.map((_, i) => `Gen ${i}`),
                    datasets: [{
                        label: 'Best Fitness',
                        data: fitnessData.map(d => d.best),
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Average Fitness',
                        data: fitnessData.map(d => d.average),
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }
        
        function updateStrategyDistribution(strategies) {
            const container = document.getElementById('strategy-distribution');
            
            const data = [{
                type: 'pie',
                values: strategies.map(s => s.count),
                labels: strategies.map(s => s.name),
                hole: .4,
                textposition: 'inside',
                textinfo: 'percent',
                marker: {
                    colors: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
                }
            }];
            
            const layout = {
                height: 300,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#fff' },
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    font: { color: '#fff' }
                }
            };
            
            Plotly.newPlot(container, data, layout, {displayModeBar: false});
        }
        
        // === ANALYTICS ===
        function initializeAnalytics() {
            // Performance metrics
            updatePerformanceMetrics();
            
            // Predictions chart
            updatePredictionsChart();
        }
        
        function updatePerformanceMetrics() {
            const container = document.getElementById('performance-metrics');
            
            // Create 3D surface plot for performance landscape
            const z = [];
            for(let i = 0; i < 50; i++) {
                z[i] = [];
                for(let j = 0; j < 50; j++) {
                    z[i][j] = Math.sin(i/5) * Math.cos(j/5) + Math.random() * 0.2;
                }
            }
            
            const data = [{
                type: 'surface',
                z: z,
                colorscale: 'Viridis'
            }];
            
            const layout = {
                height: 400,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                scene: {
                    xaxis: { title: 'Parameter 1', color: '#fff' },
                    yaxis: { title: 'Parameter 2', color: '#fff' },
                    zaxis: { title: 'Performance', color: '#fff' }
                }
            };
            
            Plotly.newPlot(container, data, layout, {displayModeBar: false});
        }
        
        function updatePredictionsChart() {
            const container = document.getElementById('predictions-chart');
            
            // Generate sample prediction data
            const actual = [];
            const predicted = [];
            const confidence = [];
            const x = [];
            
            for(let i = 0; i < 100; i++) {
                x.push(i);
                const base = Math.sin(i/10) * 50 + 50;
                actual.push(base + Math.random() * 10 - 5);
                predicted.push(base + Math.random() * 5 - 2.5);
                confidence.push(5 + Math.random() * 5);
            }
            
            const data = [
                {
                    x: x,
                    y: actual,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Actual',
                    line: { color: '#10B981' }
                },
                {
                    x: x,
                    y: predicted,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Predicted',
                    line: { color: '#3B82F6' }
                },
                {
                    x: x,
                    y: predicted.map((p, i) => p + confidence[i]),
                    type: 'scatter',
                    mode: 'lines',
                    line: { width: 0 },
                    showlegend: false
                },
                {
                    x: x,
                    y: predicted.map((p, i) => p - confidence[i]),
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tonexty',
                    fillcolor: 'rgba(59, 130, 246, 0.2)',
                    line: { width: 0 },
                    showlegend: false
                }
            ];
            
            const layout = {
                height: 400,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#fff' },
                xaxis: {
                    title: 'Time',
                    color: '#fff',
                    gridcolor: 'rgba(255, 255, 255, 0.1)'
                },
                yaxis: {
                    title: 'Value',
                    color: '#fff',
                    gridcolor: 'rgba(255, 255, 255, 0.1)'
                },
                legend: {
                    x: 0,
                    y: 1,
                    font: { color: '#fff' }
                }
            };
            
            Plotly.newPlot(container, data, layout, {displayModeBar: false});
        }
        
        // === CONSOLE ===
        function appendConsoleOutput(output) {
            const consoleEl = document.getElementById('console-output');
            const timestamp = new Date().toLocaleTimeString();
            
            consoleEl.innerHTML += `<div><span class="text-gray-500">[${timestamp}]</span> ${output}</div>`;
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
        
        function handleConsoleInput(event) {
            if (event.key === 'Enter') {
                const input = document.getElementById('console-input');
                const command = input.value.trim();
                
                if (command) {
                    // Add to history
                    consoleHistory.push(command);
                    currentHistoryIndex = consoleHistory.length;
                    
                    // Display command
                    appendConsoleOutput(`<span class="text-blue-400">&gt; ${command}</span>`);
                    
                    // Send command
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'console_command',
                            command: command
                        }));
                    }
                    
                    input.value = '';
                }
            } else if (event.key === 'ArrowUp') {
                // Navigate history up
                if (currentHistoryIndex > 0) {
                    currentHistoryIndex--;
                    document.getElementById('console-input').value = consoleHistory[currentHistoryIndex];
                }
            } else if (event.key === 'ArrowDown') {
                // Navigate history down
                if (currentHistoryIndex < consoleHistory.length - 1) {
                    currentHistoryIndex++;
                    document.getElementById('console-input').value = consoleHistory[currentHistoryIndex];
                } else {
                    currentHistoryIndex = consoleHistory.length;
                    document.getElementById('console-input').value = '';
                }
            }
        }
        
        // === COMMAND PALETTE ===
        const commands = [
            { name: 'Trigger Evolution', action: triggerEvolution, icon: 'fa-dna' },
            { name: 'Create Checkpoint', action: createCheckpoint, icon: 'fa-save' },
            { name: 'Analyze System', action: analyzeSystem, icon: 'fa-microscope' },
            { name: 'Export Data', action: exportData, icon: 'fa-download' },
            { name: 'Reset Graph View', action: resetGraphView, icon: 'fa-undo' },
            { name: 'Toggle 3D Mode', action: toggleGraphMode, icon: 'fa-cube' },
            { name: 'Show Performance Report', action: showPerformanceReport, icon: 'fa-chart-line' },
            { name: 'Run Quantum Optimization', action: runQuantumOptimization, icon: 'fa-atom' }
        ];
        
        function openCommandPalette() {
            document.getElementById('command-palette').classList.remove('hidden');
            document.getElementById('command-search').focus();
            
            // Display all commands
            filterCommands();
        }
        
        function closeCommandPalette() {
            document.getElementById('command-palette').classList.add('hidden');
        }
        
        function filterCommands() {
            const search = document.getElementById('command-search').value.toLowerCase();
            const container = document.getElementById('command-list');
            
            container.innerHTML = '';
            
            commands
                .filter(cmd => cmd.name.toLowerCase().includes(search))
                .forEach(cmd => {
                    const cmdEl = document.createElement('div');
                    cmdEl.className = 'flex items-center space-x-3 p-2 rounded hover:bg-gray-700 cursor-pointer';
                    cmdEl.onclick = () => {
                        cmd.action();
                        closeCommandPalette();
                    };
                    
                    cmdEl.innerHTML = `
                        <i class="fas ${cmd.icon} w-5"></i>
                        <span>${cmd.name}</span>
                    `;
                    
                    container.appendChild(cmdEl);
                });
        }
        
        // === ACTION FUNCTIONS ===
        function triggerEvolution() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'trigger_evolution' }));
                showAlert({
                    level: 'info',
                    message: 'Evolution triggered',
                    details: 'System evolution process has been initiated'
                });
            }
        }
        
        function createCheckpoint() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'create_checkpoint' }));
                showAlert({
                    level: 'success',
                    message: 'Checkpoint created',
                    details: 'System state has been saved'
                });
            }
        }
        
        function analyzeSystem() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'analyze_system' }));
                showAlert({
                    level: 'info',
                    message: 'System analysis started',
                    details: 'Running comprehensive system diagnostics'
                });
            }
        }
        
        function exportData() {
            window.open('/api/export', '_blank');
        }
        
        function resetGraphView() {
            if (simulation) {
                simulation.alpha(1).restart();
            }
            
            if (graphMode === '3d' && camera) {
                camera.position.set(0, 0, 100);
                camera.lookAt(0, 0, 0);
            }
        }
        
        function toggleGraphMode() {
            graphMode = graphMode === '2d' ? '3d' : '2d';
            
            // Update UI
            document.getElementById('graph-2d-container').style.display = graphMode === '2d' ? 'block' : 'none';
            document.getElementById('graph-3d-container').style.display = graphMode === '3d' ? 'block' : 'none';
            
            // Reinitialize visualization
            initializeGraphVisualization();
        }
        
        function runQuantumOptimization() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'quantum_optimization' }));
                showAlert({
                    level: 'info',
                    message: 'Quantum optimization started',
                    details: 'Running quantum algorithms for system optimization'
                });
            }
        }
        
        function showPerformanceReport() {
            // Would open a detailed performance report modal
            showAlert({
                level: 'info',
                message: 'Performance Report',
                details: 'Detailed performance metrics available in Analytics tab'
            });
            switchTab('analytics');
        }
        
        // === UTILITY FUNCTIONS ===
        function toggleTheme() {
            // Implement theme switching
            document.body.classList.toggle('light-theme');
        }
        
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }
        
        function showNodeTooltip(event, node) {
            // Create tooltip element if it doesn't exist
            let tooltip = document.getElementById('node-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'node-tooltip';
                tooltip.className = 'absolute bg-gray-800 p-3 rounded shadow-lg text-sm z-50';
                document.body.appendChild(tooltip);
            }
            
            tooltip.innerHTML = `
                <div class="font-semibold mb-1">${node.label || `Node ${node.id}`}</div>
                <div>State: ${node.state.toFixed(3)}</div>
                <div>Connections: ${node.connections || 0}</div>
                ${node.keywords ? `<div>Keywords: ${node.keywords.join(', ')}</div>` : ''}
            `;
            
            tooltip.style.left = `${event.pageX + 10}px`;
            tooltip.style.top = `${event.pageY + 10}px`;
            tooltip.style.display = 'block';
        }
        
        function hideTooltip() {
            const tooltip = document.getElementById('node-tooltip');
            if (tooltip) {
                tooltip.style.display = 'none';
            }
        }
        
        function showTooltip(event, text) {
            let tooltip = document.getElementById('generic-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'generic-tooltip';
                tooltip.className = 'absolute bg-gray-800 p-2 rounded shadow-lg text-sm z-50';
                document.body.appendChild(tooltip);
            }
            
            tooltip.textContent = text;
            tooltip.style.left = `${event.pageX + 10}px`;
            tooltip.style.top = `${event.pageY + 10}px`;
            tooltip.style.display = 'block';
        }
        
        function showEvolutionDetails(event) {
            showAlert({
                level: 'info',
                message: `Evolution ${event.id}`,
                details: `Strategy: ${event.strategy}, Fitness: ${event.fitness.toFixed(3)}`
            });
        }
        
        function updateMetrics(metrics) {
            const container = document.getElementById('metrics-summary');
            container.innerHTML = '';
            
            Object.entries(metrics).forEach(([key, value]) => {
                const metricEl = document.createElement('div');
                metricEl.className = 'metric-card p-3 rounded';
                
                const trend = value.trend === 'increasing' ? '↑' : 
                             value.trend === 'decreasing' ? '↓' : '→';
                const trendColor = value.trend === 'increasing' ? 'text-green-400' : 
                                  value.trend === 'decreasing' ? 'text-red-400' : 'text-gray-400';
                
                metricEl.innerHTML = `
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">${key}</span>
                        <span class="${trendColor}">${trend}</span>
                    </div>
                    <div class="text-2xl font-semibold mt-1">${value.current.toFixed(2)}</div>
                    <div class="text-xs text-gray-500 mt-1">
                        Avg: ${value.avg.toFixed(2)} | Min: ${value.min.toFixed(2)} | Max: ${value.max.toFixed(2)}
                    </div>
                `;
                
                container.appendChild(metricEl);
            });
        }
        
        function updateGraphStats(graphData) {
            const statusEl = document.getElementById('system-status');
            statusEl.innerHTML = `
                <div class="flex justify-between">
                    <span class="text-gray-400">Nodes</span>
                    <span>${graphData.nodes.length}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-400">Edges</span>
                    <span>${graphData.links.length}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-400">Avg State</span>
                    <span>${(graphData.nodes.reduce((sum, n) => sum + n.state, 0) / graphData.nodes.length).toFixed(3)}</span>
                </div>
            `;
        }
        
        // === DRAG FUNCTIONS FOR D3 ===
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
        
        // === INITIALIZATION ===
        document.addEventListener('DOMContentLoaded', () => {
            connectWebSocket();
            
            // Initialize with graph view
            switchTab('graph');
            
            // Start update loop for smooth animations
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'request_update' }));
                }
            }, 1000);
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    switch(e.key) {
                        case 'k':
                            e.preventDefault();
                            openCommandPalette();
                            break;
                        case 's':
                            e.preventDefault();
                            createCheckpoint();
                            break;
                        case 'e':
                            e.preventDefault();
                            triggerEvolution();
                            break;
                    }
                } else if (e.key === 'Escape') {
                    closeCommandPalette();
                }
            });
        });
    </script>
</body>
</html>
        """

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket handler para comunicación en tiempo real"""
    
    clients = set()
    
    def initialize(self, app_instance):
        self.app = app_instance
    
    def check_origin(self, origin):
        """Permite conexiones de cualquier origen (configurar en producción)"""
        return True
    
    def open(self):
        """Cliente conectado"""
        WebSocketHandler.clients.add(self)
        logger.info(f"WebSocket client connected. Total clients: {len(WebSocketHandler.clients)}")
        
        # Enviar estado inicial
        self.send_initial_state()
    
    def on_message(self, message):
        """Mensaje recibido del cliente"""
        try:
            data = json.loads(message)
            asyncio.create_task(self.handle_message(data))
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
    
    async def handle_message(self, data):
        """Maneja mensajes del cliente"""
        msg_type = data.get('type')
        
        if msg_type == 'request_update':
            await self.send_update()
        elif msg_type == 'trigger_evolution':
            await self.app.trigger_evolution()
        elif msg_type == 'create_checkpoint':
            await self.app.create_checkpoint()
        elif msg_type == 'analyze_system':
            await self.app.analyze_system()
        elif msg_type == 'console_command':
            result = await self.app.execute_console_command(data.get('command', ''))
            self.write_message(json.dumps({
                'type': 'console_output',
                'output': result
            }))
        elif msg_type == 'quantum_optimization':
            await self.app.run_quantum_optimization()
    
    def on_close(self):
        """Cliente desconectado"""
        WebSocketHandler.clients.remove(self)
        logger.info(f"WebSocket client disconnected. Total clients: {len(WebSocketHandler.clients)}")
    
    def send_initial_state(self):
        """Envía el estado inicial al cliente"""
        try:
            initial_data = {
                'type': 'initial_state',
                'graph': self.app.get_graph_data(),
                'metrics': self.app.metrics_collector.get_metrics_summary(),
                'system_status': self.app.get_system_status()
            }
            self.write_message(json.dumps(initial_data))
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
    
    async def send_update(self):
        """Envía actualización al cliente"""
        try:
            update_data = {
                'type': 'update',
                'graph': self.app.get_graph_data(),
                'metrics': self.app.metrics_collector.get_metrics_summary(),
                'quantum': self.app.get_quantum_data(),
                'evolution': self.app.get_evolution_data()
            }
            self.write_message(json.dumps(update_data))
        except Exception as e:
            logger.error(f"Error sending update: {e}")
    
    @classmethod
    def broadcast(cls, message):
        """Broadcast a todos los clientes"""
        for client in cls.clients:
            try:
                client.write_message(message)
            except:
                logger.error("Error broadcasting to client")

# === HANDLERS DE API ===

class SystemStatusHandler(BaseHandler):
    """API endpoint para estado del sistema"""
    
    def initialize(self, app_instance):
        self.app = app_instance
    
    async def get(self):
        """Obtiene estado del sistema"""
        status = self.app.get_system_status()
        self.write_json(status)

class GraphDataHandler(BaseHandler):
    """API endpoint para datos del grafo"""
    
    def initialize(self, app_instance):
        self.app = app_instance
    
    async def get(self):
        """Obtiene datos del grafo"""
        graph_data = self.app.get_graph_data()
        self.write_json(graph_data)

class ExportHandler(BaseHandler):
    """API endpoint para exportar datos"""
    
    def initialize(self, app_instance):
        self.app = app_instance
    
    async def get(self):
        """Exporta datos del sistema"""
        export_data = await self.app.export_data()
        
        self.set_header('Content-Type', 'application/json')
        self.set_header('Content-Disposition', 
                       f'attachment; filename="msc_export_{int(time.time())}.json"')
        self.write(json.dumps(export_data, indent=2))

# === APLICACIÓN PRINCIPAL ===

class TAECVizApplication:
    """Aplicación principal de TAECViz"""
    
    def __init__(self, config: TAECVizConfig, msc_framework_instance=None):
        self.config = config
        self.msc_framework = msc_framework_instance
        self.metrics_collector = VizMetricsCollector()
        
        # Cache de datos
        self.graph_cache = None
        self.quantum_cache = None
        self.evolution_cache = None
        
        # Estado
        self.is_running = False
        
        # API client para MSC Framework
        self.api_client = MSCAPIClient(config.msc_api_url)
        
        # Configurar aplicación Tornado
        self.app = self._create_tornado_app()
        
        # Executor para operaciones pesadas
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _create_tornado_app(self):
        """Crea la aplicación Tornado"""
        handlers = [
            (r"/", DashboardHandler),
            (r"/ws", WebSocketHandler, dict(app_instance=self)),
            (r"/api/status", SystemStatusHandler, dict(app_instance=self)),
            (r"/api/graph", GraphDataHandler, dict(app_instance=self)),
            (r"/api/export", ExportHandler, dict(app_instance=self)),
        ]
        
        settings = {
            "debug": True,
            "autoreload": False,
            "compress_response": True,
            "websocket_ping_interval": 10,
            "websocket_ping_timeout": 30,
        }
        
        return tornado.web.Application(handlers, **settings)
    
    async def start(self):
        """Inicia la aplicación"""
        self.is_running = True
        
        # Iniciar servidor
        self.app.listen(self.config.port, self.config.host)
        logger.info(f"TAECViz 2.0 running on http://{self.config.host}:{self.config.port}")
        
        # Iniciar loop de actualización
        tornado.ioloop.IOLoop.current().spawn_callback(self._update_loop)
        
        # Abrir navegador
        try:
            webbrowser.open(f"http://{self.config.host}:{self.config.port}")
        except:
            pass
    
    async def _update_loop(self):
        """Loop de actualización de datos"""
        while self.is_running:
            try:
                # Actualizar datos desde MSC Framework
                await self._update_from_msc()
                
                # Broadcast actualizaciones
                update_data = {
                    'type': 'metrics_update',
                    'metrics': self.metrics_collector.get_metrics_summary()
                }
                WebSocketHandler.broadcast(json.dumps(update_data))
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
            
            await asyncio.sleep(self.config.update_interval)
    
    async def _update_from_msc(self):
        """Actualiza datos desde MSC Framework"""
        if self.msc_framework:
            # Actualización directa desde instancia
            self._update_graph_cache()
            self._update_metrics()
        else:
            # Actualización via API
            try:
                status = await self.api_client.get_status()
                graph = await self.api_client.get_graph()
                
                if graph:
                    self.graph_cache = self._process_graph_data(graph)
                
                if status:
                    self._update_metrics_from_status(status)
                    
            except Exception as e:
                logger.error(f"Error updating from MSC API: {e}")
    
    def _update_graph_cache(self):
        """Actualiza cache del grafo desde instancia local"""
        if not self.msc_framework or not hasattr(self.msc_framework, 'graph'):
            return
        
        graph = self.msc_framework.graph
        nodes = []
        links = []
        
        # Procesar nodos
        for node_id, node in graph.nodes.items():
            nodes.append({
                'id': str(node_id),
                'label': getattr(node, 'content', f'Node {node_id}')[:30],
                'state': getattr(node, 'state', 0.5),
                'keywords': list(getattr(node, 'keywords', [])),
                'connections': len(getattr(node, 'connections_out', {}))
            })
        
        # Procesar enlaces
        for node_id, node in graph.nodes.items():
            for target_id, utility in getattr(node, 'connections_out', {}).items():
                links.append({
                    'source': str(node_id),
                    'target': str(target_id),
                    'value': abs(utility)
                })
        
        self.graph_cache = {
            'nodes': nodes[:self.config.max_graph_nodes],
            'links': links
        }
    
    def _update_metrics(self):
        """Actualiza métricas desde instancia local"""
        if not self.msc_framework:
            return
        
        # Métricas del grafo
        if hasattr(self.msc_framework, 'graph'):
            graph = self.msc_framework.graph
            node_count = len(graph.nodes)
            
            if node_count > 0:
                avg_state = sum(getattr(n, 'state', 0) for n in graph.nodes.values()) / node_count
                self.metrics_collector.add_metric('avg_node_state', avg_state)
            
            self.metrics_collector.add_metric('node_count', node_count)
        
        # Métricas de TAEC si está disponible
        if hasattr(self.msc_framework, 'agents'):
            taec_agents = [a for a in self.msc_framework.agents 
                          if hasattr(a, 'evolution_count')]
            
            if taec_agents:
                total_evolutions = sum(a.evolution_count for a in taec_agents)
                self.metrics_collector.add_metric('total_evolutions', total_evolutions)
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Obtiene datos del grafo"""
        return self.graph_cache or {'nodes': [], 'links': []}
    
    def get_quantum_data(self) -> Dict[str, Any]:
        """Obtiene datos cuánticos"""
        if not self.quantum_cache:
            # Generar datos de ejemplo
            self.quantum_cache = {
                'states': [
                    {
                        'address': f'qubit_{i}',
                        'coherence': 0.9 - i * 0.1,
                        'amplitudes': [
                            complex(np.cos(i * np.pi/8), np.sin(i * np.pi/8)).real
                            for _ in range(4)
                        ]
                    }
                    for i in range(5)
                ],
                'entanglements': {
                    'nodes': [{'id': f'q{i}'} for i in range(5)],
                    'links': [
                        {'source': 'q0', 'target': 'q1', 'strength': 0.8},
                        {'source': 'q1', 'target': 'q2', 'strength': 0.6},
                        {'source': 'q2', 'target': 'q3', 'strength': 0.7},
                        {'source': 'q3', 'target': 'q4', 'strength': 0.5},
                        {'source': 'q0', 'target': 'q4', 'strength': 0.9}
                    ]
                },
                'memoryMap': [
                    [np.random.random() for _ in range(10)]
                    for _ in range(10)
                ]
            }
        
        return self.quantum_cache
    
    def get_evolution_data(self) -> Dict[str, Any]:
        """Obtiene datos de evolución"""
        if not self.evolution_cache:
            # Generar datos de ejemplo
            self.evolution_cache = {
                'history': [
                    {
                        'id': i,
                        'success': np.random.random() > 0.3,
                        'strategy': np.random.choice(['synthesis', 'optimization', 'exploration']),
                        'fitness': 0.5 + 0.5 * (i / 20)
                    }
                    for i in range(20)
                ],
                'fitness': [
                    {
                        'best': 0.5 + 0.5 * (i / 50) + np.random.random() * 0.1,
                        'average': 0.3 + 0.3 * (i / 50) + np.random.random() * 0.1
                    }
                    for i in range(50)
                ],
                'strategies': [
                    {'name': 'Synthesis', 'count': 45},
                    {'name': 'Optimization', 'count': 30},
                    {'name': 'Exploration', 'count': 25}
                ],
                'generatedCode': '''synth advanced_synthesis {
    node alpha { state => 0.9; }
    node beta { state => 0.7; }
    alpha <-> beta;
    evolve alpha "optimization";
}'''
            }
        
        return self.evolution_cache
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema"""
        return {
            'running': self.is_running,
            'connected_clients': len(WebSocketHandler.clients),
            'cache_size': {
                'graph': len(self.graph_cache['nodes']) if self.graph_cache else 0,
                'quantum': 1 if self.quantum_cache else 0,
                'evolution': 1 if self.evolution_cache else 0
            },
            'uptime': int(time.time()),
            'version': '2.0.0'
        }
    
    async def trigger_evolution(self):
        """Dispara evolución en el sistema"""
        if self.msc_framework:
            # Evolución local
            # Buscar agente TAEC
            taec_agents = [a for a in self.msc_framework.agents 
                          if hasattr(a, 'evolve_system')]
            
            if taec_agents:
                result = await taec_agents[0].evolve_system()
                self.metrics_collector.add_alert(
                    'info', 
                    'Evolution completed',
                    {'success': result.get('success', False)}
                )
        else:
            # Evolución via API
            result = await self.api_client.trigger_evolution()
            
        # Broadcast resultado
        WebSocketHandler.broadcast(json.dumps({
            'type': 'alert',
            'alert': {
                'level': 'success' if result else 'error',
                'message': 'Evolution completed' if result else 'Evolution failed'
            }
        }))
    
    async def create_checkpoint(self):
        """Crea checkpoint del sistema"""
        if self.msc_framework:
            # Implementar creación de checkpoint
            pass
        else:
            result = await self.api_client.create_checkpoint()
        
        self.metrics_collector.add_alert('success', 'Checkpoint created')
    
    async def analyze_system(self):
        """Analiza el sistema"""
        analysis_results = {
            'graph_health': 0.0,
            'evolution_efficiency': 0.0,
            'quantum_coherence': 0.0
        }
        
        if self.graph_cache:
            # Análisis del grafo
            nodes = self.graph_cache['nodes']
            if nodes:
                analysis_results['graph_health'] = np.mean([n['state'] for n in nodes])
        
        # Broadcast resultados
        WebSocketHandler.broadcast(json.dumps({
            'type': 'alert',
            'alert': {
                'level': 'info',
                'message': 'System analysis complete',
                'details': json.dumps(analysis_results)
            }
        }))
    
    async def execute_console_command(self, command: str) -> str:
        """Ejecuta comando de consola"""
        try:
            # Parsear comando
            parts = command.split()
            if not parts:
                return "Empty command"
            
            cmd = parts[0].lower()
            
            # Comandos disponibles
            if cmd == 'help':
                return """Available commands:
- status: Show system status
- evolve: Trigger evolution
- checkpoint: Create checkpoint
- analyze: Analyze system
- clear: Clear console
- metrics [name]: Show specific metric
- graph info: Show graph information"""
            
            elif cmd == 'status':
                status = self.get_system_status()
                return json.dumps(status, indent=2)
            
            elif cmd == 'evolve':
                await self.trigger_evolution()
                return "Evolution triggered"
            
            elif cmd == 'checkpoint':
                await self.create_checkpoint()
                return "Checkpoint created"
            
            elif cmd == 'analyze':
                await self.analyze_system()
                return "Analysis started"
            
            elif cmd == 'clear':
                return "<CLEAR>"
            
            elif cmd == 'metrics' and len(parts) > 1:
                metric_name = parts[1]
                metrics = self.metrics_collector.get_metrics_summary()
                if metric_name in metrics:
                    return json.dumps(metrics[metric_name], indent=2)
                else:
                    return f"Metric '{metric_name}' not found"
            
            elif cmd == 'graph' and len(parts) > 1 and parts[1] == 'info':
                if self.graph_cache:
                    info = {
                        'nodes': len(self.graph_cache['nodes']),
                        'edges': len(self.graph_cache['links'])
                    }
                    return json.dumps(info, indent=2)
                else:
                    return "No graph data available"
            
            else:
                return f"Unknown command: {cmd}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def run_quantum_optimization(self):
        """Ejecuta optimización cuántica"""
        # Implementar optimización cuántica
        self.metrics_collector.add_alert(
            'info',
            'Quantum optimization started',
            {'algorithm': 'VQE', 'qubits': 8}
        )
    
    async def export_data(self) -> Dict[str, Any]:
        """Exporta todos los datos del sistema"""
        return {
            'timestamp': time.time(),
            'version': '2.0.0',
            'graph': self.get_graph_data(),
            'quantum': self.get_quantum_data(),
            'evolution': self.get_evolution_data(),
            'metrics': self.metrics_collector.get_metrics_summary(),
            'alerts': list(self.metrics_collector.alerts)
        }

# === CLIENTE API PARA MSC FRAMEWORK ===

class MSCAPIClient:
    """Cliente para comunicarse con MSC Framework API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
    
    async def _get_session(self):
        """Obtiene o crea sesión aiohttp"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_status(self) -> Optional[Dict[str, Any]]:
        """Obtiene estado del sistema"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/system/health") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Error getting status: {e}")
        return None
    
    async def get_graph(self) -> Optional[Dict[str, Any]]:
        """Obtiene datos del grafo"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/graph/status") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Error getting graph: {e}")
        return None
    
    async def trigger_evolution(self) -> bool:
        """Dispara evolución"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/agents/taec/evolve") as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Error triggering evolution: {e}")
        return False
    
    async def create_checkpoint(self) -> bool:
        """Crea checkpoint"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/simulation/checkpoint") as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
        return False
    
    async def close(self):
        """Cierra la sesión"""
        if self.session:
            await self.session.close()

# === FUNCIÓN DE INTEGRACIÓN ===

def integrate_taecviz_with_msc(msc_instance=None, config: Optional[Dict[str, Any]] = None):
    """Integra TAECViz 2.0 con MSC Framework"""
    
    # Configuración
    viz_config = TAECVizConfig(
        host=config.get('host', 'localhost') if config else 'localhost',
        port=config.get('port', 8888) if config else 8888,
        msc_api_url=config.get('msc_api_url', 'http://localhost:5000') if config else 'http://localhost:5000',
        enable_3d=config.get('enable_3d', True) if config else True,
        enable_quantum_viz=config.get('enable_quantum_viz', True) if config else True,
        enable_ml_analytics=config.get('enable_ml_analytics', True) if config else True
    )
    
    # Crear aplicación
    app = TAECVizApplication(viz_config, msc_instance)
    
    # Iniciar en el IOLoop de Tornado
    async def start_app():
        await app.start()
    
    # Configurar y ejecutar
    tornado.ioloop.IOLoop.current().run_sync(start_app)
    
    # Mantener ejecutando
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("TAECViz shutting down...")
        app.is_running = False
        if app.api_client.session:
            tornado.ioloop.IOLoop.current().run_sync(app.api_client.close)

# === MAIN ===

if __name__ == "__main__":
    # Ejemplo de uso standalone
    print("TAECViz 2.0 - MSC Framework Visualization")
    print("==========================================")
    print()
    print("Starting visualization server...")
    print("Open your browser at http://localhost:8888")
    print()
    print("Press Ctrl+C to stop")
    
    integrate_taecviz_with_msc()
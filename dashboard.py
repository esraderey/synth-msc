import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import requests  # Para llamar a la API
import plotly.graph_objects as go  # Podríamos usarlo para otras gráficas
import time
import dash_cytoscape as cyto  # <-- NUEVA IMPORTACIÓN para grafos

# --- Configuración de la App Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

SIMULATION_API_URL = "http://127.0.0.1:5000"  # URL del backend

# --- Layout de la Aplicación (Actualizado) ---
app.layout = dbc.Container([
    html.H1("MSC Simulation Dashboard", className="text-center my-4"),
    dbc.Row([
        # Columna para Estado
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Status"),
                dbc.CardBody(
                    # Usamos un Div para poder poner múltiples líneas fácilmente
                    html.Div(id='status-text', children="Loading status...")
                )
            ])
        ], width=12, md=4, lg=3, className="mb-3"),  # Ajustar ancho

        # Columna para Visualización del Grafo
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Graph Visualization"),
                dbc.CardBody([
                    # --- Reemplazado dcc.Graph con cyto.Cytoscape ---
                    cyto.Cytoscape(
                        id='cytoscape-graph',
                        layout={
                            'name': 'cose',  # Layout automático común (alternativas: 'grid', 'circle', 'breadthfirst')
                            'idealEdgeLength': 100,
                            'nodeOverlap': 20,
                            'padding': 30
                        },
                        style={
                            'width': '100%',
                            'height': '600px',
                            'border': '1px solid #ddd',
                            'border-radius': '5px'
                        },
                        elements=[],  # Empezar vacío, se llena con el callback
                        # Stylesheet para definir apariencia de nodos/aristas basado en datos
                        stylesheet=[
                            {  # Estilo base para nodos
                                'selector': 'node',
                                'style': {
                                    'background-color': '#66a3ff',  # Color azul base
                                    'label': 'data(label)',         # Usa la etiqueta definida en los datos
                                    'width': '20px',                # Tamaño base
                                    'height': '20px',
                                    'font-size': '8px',
                                    'color': '#333',                # Color del texto de la etiqueta
                                    'text-valign': 'center',
                                    'text-halign': 'center'
                                }
                            },
                            {  # Estilo condicional para nodos basado en estado 'state'
                                'selector': 'node[state]',  # Selecciona nodos que tienen el atributo 'state'
                                'style': {
                                    # Mapea estado [0,1] a un gradiente de color (ej: azul claro a amarillo/verde)
                                    'background-color': 'mapData(state, 0, 1, #adebff, #ccff99)',
                                    # Mapea estado [0,1] a tamaño (ej: 15px a 50px)
                                    'width': 'mapData(state, 0, 1, 15, 50)',
                                    'height': 'mapData(state, 0, 1, 15, 50)'
                                }
                            },
                            {  # Estilo base para aristas
                                'selector': 'edge',
                                'style': {
                                    'line-color': '#ccc',  # Color gris base
                                    'width': 1,            # Grosor base
                                    'curve-style': 'bezier',  # Líneas curvas
                                    'target-arrow-shape': 'triangle',  # Forma de la flecha
                                    'target-arrow-color': '#ccc'
                                }
                            },
                            {  # Estilo condicional para aristas basado en utilidad 'utility'
                                'selector': 'edge[utility]',
                                'style': {
                                    # Mapea utilidad [-1, 1] a un gradiente de color (ej: rojo -> gris -> azul)
                                    'line-color': 'mapData(utility, -1, 1, red, blue)',
                                    # Opcional: mapear grosor al valor absoluto, si se desea
                                    # 'width': 'mapData(utility, -1, 1, 5, 1)'  
                                }
                            }
                        ]
                    )
                    # --- Fin cyto.Cytoscape ---
                ])
            ])
        ], width=12, md=8, lg=9)  # Ajustar ancho
    ]),
    # Componente para actualizar periódicamente
    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,  # Actualizar cada 5 segundos
        n_intervals=0
    )
], fluid=True)

# --- Callback para Actualizar Datos (Modificado) ---
@app.callback(
    Output('status-text', 'children'),      # Output 1: Texto de estado
    Output('cytoscape-graph', 'elements'),    # Output 2: Elementos del grafo Cytoscape
    Input('interval-component', 'n_intervals')  # Input: Temporizador
)
def update_dashboard(n):
    """Consulta AMBAS APIs (/status y /graph_data) y actualiza el dashboard."""

    # Valores por defecto en caso de error
    status_data = {"error": "Could not load status"}
    cytoscape_elements = []  # Lista vacía para el grafo por defecto

    # 1. Obtener Estado de la Simulación
    try:
        response_status = requests.get(f"{SIMULATION_API_URL}/status", timeout=4)  # Timeout un poco más largo
        response_status.raise_for_status()
        status_data = response_status.json()
        status_text_content = [
            f"Running: {status_data.get('is_running', 'N/A')}", html.Br(),
            f"Current Step: {status_data.get('current_step', 'N/A')}", html.Br(),
            f"Nodes: {status_data.get('node_count', 'N/A')}", html.Br(),
            f"Edges: {status_data.get('edge_count', 'N/A')}", html.Br(),
            f"Avg State: {status_data.get('average_state', 'N/A')}", html.Br(),
            f"Embeddings: {status_data.get('embeddings_count', 'N/A')}"
        ]
    except requests.exceptions.RequestException as e:
        status_text_content = f"Error getting status: {e}"
    except Exception as e:
        status_text_content = f"Error processing status: {e}"

    # 2. Obtener Datos del Grafo para Cytoscape
    try:
        response_graph = requests.get(f"{SIMULATION_API_URL}/graph_data", timeout=4)
        response_graph.raise_for_status()
        cytoscape_elements = response_graph.json()  # La API ya devuelve el formato correcto
    except requests.exceptions.RequestException as e:
        # Mantener el texto de estado aunque falle la carga del grafo
        status_text_content = [html.Div(status_text_content), html.Br(), html.Div(f"Error getting graph data: {e}", style={'color': 'red'})]
    except Exception as e:
        status_text_content = [html.Div(status_text_content), html.Br(), html.Div(f"Error processing graph data: {e}", style={'color': 'red'})]

    # Devolver ambos outputs
    return status_text_content, cytoscape_elements

# --- Ejecutar la App Dash ---
if __name__ == '__main__':
    print(f"Dash app running on http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)  # debug=True ayuda a ver errores de Dash

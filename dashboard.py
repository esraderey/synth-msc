import dash
from dash import dcc, html, Input, Output, State # Input, Output, State son necesarios para callbacks
import dash_bootstrap_components as dbc
import requests # Para llamar a la API
import plotly.graph_objects as go
import time
import dash_cytoscape as cyto # Importar Cytoscape

# --- Configuración App Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "MSC Dashboard" # Título de la pestaña del navegador

SIMULATION_API_URL = "http://127.0.0.1:5000" # URL del backend

# --- Layout de la Aplicación ---
app.layout = dbc.Container([
    html.H1("MSC Simulation Dashboard", className="text-center my-4"),
    dbc.Row([
        # Columna para Estado
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Status"),
                dbc.CardBody(
                    html.Div(id='status-text', children="Connecting to simulation API...")
                )
            ])
        ], width=12, md=4, lg=3, className="mb-3"),

        # Columna para Visualización del Grafo
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Graph Visualization"),
                dbc.CardBody([
                    cyto.Cytoscape(
                        id='cytoscape-graph',
                        layout={'name': 'cose',
                                'idealEdgeLength': 100,
                                'nodeOverlap': 20,
                                'padding': 30,
                                'animate': False, # Desactivar animación de layout en cada update
                                'randomize': False
                               },
                        style={'width': '100%', 'height': '600px', 'border': '1px solid #ddd', 'border-radius': '5px'},
                        elements=[], # Empezar vacío
                        stylesheet=[ # Estilos definidos antes
                            {'selector': 'node', 'style': {'label': 'data(label)','font-size': '8px','color': '#333','text-valign': 'center','text-halign': 'center'}},
                            {'selector': 'node[state]', 'style': {'background-color': 'mapData(state, 0, 1, #adebff, #ccff99)','width': 'mapData(state, 0, 1, 15, 50)','height': 'mapData(state, 0, 1, 15, 50)'}},
                            {'selector': 'edge', 'style': {'width': 1, 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#ccc'}},
                            {'selector': 'edge[utility]', 'style': {'line-color': 'mapData(utility, -1, 1, red, blue)', 'width': 'mapData(utility, -1, 1, 3, 1)'}} # Grosor inverso a utilidad? Ajustar mapeo si es necesario, ej 'mapData(abs(utility), 0, 1, 1, 5)'
                        ]
                    )
                ])
            ])
        ], width=12, md=8, lg=9)
    ]),

    # Componente Interval para actualizaciones periódicas
    dcc.Interval(
        id='interval-component',
        interval=5*1000, # Cada 5 segundos
        n_intervals=0
    )
], fluid=True)

# --- Callback ACTUALIZADO para Estado Y Grafo ---
@app.callback(
    Output('status-text', 'children'),      # Output 1
    Output('cytoscape-graph', 'elements'), # Output 2 <-- NUEVO
    Input('interval-component', 'n_intervals') # Trigger
)
def update_dashboard(n):
    """Consulta AMBAS APIs (/status y /graph_data) y actualiza el dashboard."""

    # Valores por defecto
    status_text_content = "Failed to load status."
    cytoscape_elements = [] # Grafo vacío si falla

    # 1. Obtener Estado
    try:
        response_status = requests.get(f"{SIMULATION_API_URL}/status", timeout=4)
        response_status.raise_for_status()
        status_data = response_status.json()
        status_text_content = [
            f"Running: {status_data.get('is_running', 'N/A')}", html.Br(),
            f"Step: {status_data.get('current_step', 'N/A')}", html.Br(),
            f"Nodes: {status_data.get('node_count', 'N/A')}", html.Br(),
            f"Edges: {status_data.get('edge_count', 'N/A')}", html.Br(),
            f"Avg State: {status_data.get('average_state', 'N/A')}", html.Br(),
            f"Embeddings: {status_data.get('embeddings_count', 'N/A')}"
        ]
    except requests.exceptions.RequestException as e:
        status_text_content = f"Error Status API: {e}"
    except Exception as e:
        status_text_content = f"Error Status processing: {e}"

    # 2. Obtener Datos del Grafo (NUEVO)
    try:
        response_graph = requests.get(f"{SIMULATION_API_URL}/graph_data", timeout=4)
        response_graph.raise_for_status()
        cytoscape_elements = response_graph.json() # La API ya devuelve el formato correcto
    except requests.exceptions.RequestException as e:
        # Añadir mensaje de error al status si falla la carga del grafo
        error_msg = f"Error Graph API: {e}"
        if isinstance(status_text_content, list): status_text_content.append(html.Br()); status_text_content.append(html.Span(error_msg, style={'color':'red'}))
        else: status_text_content = html.Div([html.Span(status_text_content), html.Br(), html.Span(error_msg, style={'color':'red'})])
        # Devolver grafo vacío en caso de error
        cytoscape_elements = []
    except Exception as e:
        error_msg = f"Error Graph processing: {e}"
        if isinstance(status_text_content, list): status_text_content.append(html.Br()); status_text_content.append(html.Span(error_msg, style={'color':'red'}))
        else: status_text_content = html.Div([html.Span(status_text_content), html.Br(), html.Span(error_msg, style={'color':'red'})])
        cytoscape_elements = []

    # Devolver AMBOS outputs para actualizar el dashboard
    return status_text_content, cytoscape_elements


# --- Ejecutar la App Dash ---
if __name__ == '__main__':
    print(f"Dash app running on http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

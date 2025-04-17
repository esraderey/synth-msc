import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import requests
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MSC Simulation Dashboard"

# Layout del dashboard con un componente Cytoscape, un área de status y controles de filtro
app.layout = html.Div([
    html.H1("MSC Simulation Dashboard"),
    
    # Área de status actual de la simulación
    html.Div(id='status-div', children="Status will be updated...", style={'marginBottom': '20px'}),
    
    # Componente Cytoscape para visualizar el grafo
    cyto.Cytoscape(
        id='cytoscape-graph',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'},
        elements=[]
    ),
    
    # Control de actualización automática
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Actualiza cada 2 segundos
        n_intervals=0
    ),
    
    # Filtros para explorar nodos
    html.Div([
        html.Label("Node Filter:"),
        dcc.Input(id='node-filter', type='text', placeholder='Tipo de identificador o contenido'),
        html.Button("Apply Filter", id='btn-filter')
    ], style={'marginTop': '20px'})
])

def fetch_status():
    try:
        response = requests.get("http://127.0.0.1:5000/status")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        return {}

def fetch_graph_data():
    try:
        response = requests.get("http://127.0.0.1:5000/graph_data")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        return []

# Callback para actualizar el área de status
@app.callback(
    Output('status-div', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_status(n):
    status = fetch_status()
    if not status:
        raise PreventUpdate
    return html.Div([
        html.P(f"Step: {status.get('current_step', 'N/A')}"),
        html.P(f"Nodes: {status.get('node_count', 'N/A')}"),
        html.P(f"Edges: {status.get('edge_count', 'N/A')}"),
        html.P(f"Avg State: {status.get('average_state', 'N/A')}"),
        html.P(f"Avg Omega: {status.get('average_omega', 'N/A')}")
    ])

# Callback para actualizar los elementos del grafo en Cytoscape, con capacidad de filtro
@app.callback(
    Output('cytoscape-graph', 'elements'),
    [Input('interval-component', 'n_intervals'),
     Input('btn-filter', 'n_clicks')],
    State('node-filter', 'value')
)
def update_cytoscape(n_intervals, n_clicks, filter_value):
    elements = fetch_graph_data()
    # Si hay filtro, se realiza una selección simple de nodos cuyo ID contenga el valor de filtro
    if filter_value:
        filtered = []
        for ele in elements:
            if 'data' in ele and 'id' in ele['data']:
                if filter_value.lower() in ele['data']['id'].lower():
                    filtered.append(ele)
            else:
                filtered.append(ele)
        return filtered
    return elements

if __name__ == '__main__':
    app.run_server(debug=True)

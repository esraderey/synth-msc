import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import requests
from dash.exceptions import PreventUpdate

MSC_API_URL = "http://127.0.0.1:5000"  # Volver al puerto predeterminado

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
        layout={'name': 'cose', 'animate': False, 'nodeRepulsion': 8000},
        style={'width': '100%', 'height': '600px'},
        elements=[],
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': 'data(background-color)',
                    'width': 'data(width)',
                    'height': 'data(height)',
                    'text-wrap': 'wrap',
                    'text-max-width': '80px',
                    'font-size': '10px'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 'data(width)',
                    'line-color': 'data(line-color)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': 'data(line-color)'
                }
            }
        ]
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
        response = requests.get(f"{MSC_API_URL}/status")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        return {}

def fetch_graph_data():
    try:
        print(f"Fetching graph data from {MSC_API_URL}/graph_data...")
        response = requests.get(f"{MSC_API_URL}/graph_data", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Received data: {data[:100]}...")  # Print part of the response for debugging
            
            # Check if we need to transform the data format
            if isinstance(data, dict) and 'elements' in data:
                return data['elements']
            return data
        else:
            print(f"Error: Status code {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print(f"Connection refused. Make sure MSC API is running on {MSC_API_URL}")
        return []
    except Exception as e:
        print(f"Exception fetching graph data: {str(e)}")
        return []

def check_api_status():
    try:
        response = requests.get(f"{MSC_API_URL}/status", timeout=2)
        return response.status_code == 200
    except:
        return False

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
    
    # Debug: imprimir primeros 2 elementos para ver su estructura
    if elements and len(elements) > 2:
        print("Sample elements:")
        print(elements[0])
        print(elements[1])
    
    # Convertir datos si tienen un formato incorrecto
    formatted_elements = []
    for ele in elements:
        if 'data' in ele:
            # Corregir formato de estilo si está anidado incorrectamente
            if 'style' in ele:
                for style_key, style_val in ele['style'].items():
                    ele['data'][style_key] = style_val
                del ele['style']
            formatted_elements.append(ele)
        else:
            # Si falta estructura de datos
            print(f"Elemento con formato incorrecto: {ele}")
    
    # Aplicar filtro si es necesario
    if filter_value:
        filtered = [ele for ele in formatted_elements if 'data' in ele and 'id' in ele['data'] 
                   and filter_value.lower() in str(ele['data']['id']).lower()]
        return filtered
    
    return formatted_elements if formatted_elements else elements

# Callback para actualizar el estilo del área de status según el estado de conexión
@app.callback(
    Output('status-div', 'style'),
    Input('interval-component', 'n_intervals')
)
def update_connection_status(n):
    if check_api_status():
        return {'marginBottom': '20px', 'backgroundColor': '#e6ffe6', 'padding': '10px'}
    else:
        return {'marginBottom': '20px', 'backgroundColor': '#ffe6e6', 'padding': '10px', 'color': 'red'}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

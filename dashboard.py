import dash
from dash import dcc, html, Input, Output
import dash_cytoscape as cyto
import plotly.graph_objs as go
import requests
import pandas as pd
import os

# Base URL where msc_simulation's Flask server is running
BASE_URL = "http://127.0.0.1:5000"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose Flask server if needed by deployment

app.layout = html.Div([
    html.H1("MSC Simulation Dashboard"),
    html.Div(id='simulation-status'),
    html.Button("Refresh Data", id='refresh-button', n_clicks=0),
    html.H2("Graph Visualization"),
    cyto.Cytoscape(
        id='cytoscape-graph',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '500px'},
        elements=[]
    ),
    html.H2("Metrics Evolution"),
    dcc.Graph(id='metrics-graph'),
    html.H2("Controls"),
    html.Button("Stop Simulation", id='stop-button', n_clicks=0),
    html.Button("Start Simulation", id='start-button', n_clicks=0)
])

@app.callback(
    Output('simulation-status', 'children'),
    Input('refresh-button', 'n_clicks')
)
def update_status(n_clicks):
    try:
        res = requests.get(f"{BASE_URL}/status")
        status = res.json()
        return html.Div([
            html.P(f"Simulation Running: {status.get('is_running')}"),
            html.P(f"Current Step: {status.get('current_step')}"),
            html.P(f"Node Count: {status.get('node_count')}"),
            html.P(f"Edge Count: {status.get('edge_count')}"),
            html.P(f"Average State: {status.get('average_state')}"),
            html.P(f"Average Reputation: {status.get('average_reputation')}"),
            html.P(f"Average Omega: {status.get('average_omega')}")
        ])
    except Exception as e:
        return f"Error fetching status: {e}"

@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('refresh-button', 'n_clicks')
)
def update_graph(n_clicks):
    try:
        res = requests.get(f"{BASE_URL}/graph_data")
        elements = res.json()
        return elements
    except Exception as e:
        return []

@app.callback(
    Output('metrics-graph', 'figure'),
    Input('refresh-button', 'n_clicks')
)
def update_metrics(n_clicks):
    # CSV metrics file path can be configured via an environment variable
    metrics_path = os.getenv("METRICS_CSV_PATH", "metrics.csv")
    if not os.path.exists(metrics_path):
        return {'data': [], 'layout': go.Layout(title="Metrics Evolution (CSV not found)")}
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            return {'data': [], 'layout': go.Layout(title="Metrics Evolution (No Data)")}
        figure = go.Figure()
        if "Step" in df.columns and "Nodes" in df.columns:
            figure.add_trace(go.Scatter(x=df["Step"], y=df["Nodes"],
                                        mode='lines+markers', name='Nodes'))
        if "Step" in df.columns and "Edges" in df.columns:
            figure.add_trace(go.Scatter(x=df["Step"], y=df["Edges"],
                                        mode='lines+markers', name='Edges'))
        if "Step" in df.columns and "MeanState" in df.columns:
            figure.add_trace(go.Scatter(x=df["Step"], y=df["MeanState"],
                                        mode='lines+markers', name='MeanState'))
        figure.update_layout(title="Global Metrics Evolution",
                             xaxis_title="Step", yaxis_title="Value")
        return figure
    except Exception as e:
        return {'data': [], 'layout': go.Layout(title=f"Error reading CSV: {e}")}

@app.callback(
    Output('stop-button', 'children'),
    Output('start-button', 'children'),
    Input('stop-button', 'n_clicks'),
    Input('start-button', 'n_clicks')
)
def control_simulation(stop_n, start_n):
    # These are placeholders; integration with simulation control endpoints is expected.
    # For example, POST requests to /control/stop or /control/start could be invoked here.
    return "Stop Simulation", "Start Simulation"

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

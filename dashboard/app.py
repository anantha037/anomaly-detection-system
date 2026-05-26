import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
import torch
from pathlib import Path
import sys

from data_loader import load_test_data, fetch_predictions, compute_sensor_errors

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.autoencoder import LSTMAutoencoder

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "AnomalyGuard"

# Fetch Initial Data
X_test, y_test = load_test_data()

# Load model for sensor error computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMAutoencoder(n_features=8, latent_dim=32, n_layers=2)
model_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "autoencoder.pth"
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)

def get_api_status():
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        return r.status_code == 200
    except:
        return False

# Layout
app.layout = dbc.Container([
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("AnomalyGuard", className="text-white mt-4 fw-bold"),
            html.H4("Multivariate Time Series Anomaly Detection", className="text-secondary"),
        ], width=9),
        dbc.Col([
            html.Div(id="api-status-indicator", className="mt-5 text-end")
        ], width=3)
    ], className="mb-4"),
    
    # KPIs
    dbc.Row(id="kpi-cards", className="mb-4"),
    
    # Main Plot
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Reconstruction Error & Anomalies over Time", className="fw-bold"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-main-plot",
                        type="default",
                        children=dcc.Graph(id="main-plot")
                    )
                ])
            ], className="custom-card")
        ])
    ], className="mb-4"),
    
    # Side by side plots
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Reconstruction Error Distribution", className="fw-bold"),
                dbc.CardBody(dcc.Graph(id="dist-plot"))
            ], className="custom-card")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Anomaly Timeline Heatmap (Severity)", className="fw-bold"),
                dbc.CardBody(dcc.Graph(id="heatmap-plot"))
            ], className="custom-card")
        ], width=6)
    ], className="mb-4"),
    
    # Sensor Contribution & Model Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Per-Sensor Contribution to Anomalies", className="fw-bold"),
                dbc.CardBody(dcc.Graph(id="sensor-plot"))
            ], className="custom-card")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Comparison", className="fw-bold"),
                dbc.CardBody(html.Div(id="model-table"))
            ], className="custom-card")
        ], width=6)
    ], className="mb-4")
    
], fluid=True, className="main-container")

# Callbacks
@app.callback(
    Output("api-status-indicator", "children"),
    Input("interval-component", "n_intervals")
)
def update_status(n):
    is_up = get_api_status()
    if is_up:
        return html.Div([
            html.Span(className="status-dot green-dot"),
            html.Span(" API Connected", className="text-success ms-2 fw-bold")
        ])
    else:
        return html.Div([
            html.Span(className="status-dot red-dot"),
            html.Span(" API Disconnected", className="text-danger ms-2 fw-bold")
        ])

@app.callback(
    [Output("kpi-cards", "children"),
     Output("main-plot", "figure"),
     Output("dist-plot", "figure"),
     Output("heatmap-plot", "figure"),
     Output("sensor-plot", "figure"),
     Output("model-table", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    # Only update plots on the very first load to avoid re-running massive predictions
    if n > 0:
        return dash.no_update
        
    is_up = get_api_status()
    if not is_up or X_test is None:
        return html.Div("API is offline or data missing.", className="text-danger"), go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div()
        
    # Fetch data
    threshold_data = requests.get("http://localhost:8000/threshold").json()
    metrics_data = requests.get("http://localhost:8000/metrics").json()
    predictions = fetch_predictions(X_test)
    
    threshold = threshold_data["threshold"]
    f1 = metrics_data["f1_score"]
    
    if not predictions:
        return html.Div("Failed to fetch predictions."), go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div()

    df = pd.DataFrame(predictions)
    total_windows = len(df)
    anom_count = df["is_anomaly"].sum()
    anom_pct = (anom_count / total_windows) * 100 if total_windows > 0 else 0
    
    # KPI Cards
    kpis = [
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Windows Analyzed", className="text-secondary"), html.H3(f"{total_windows:,}", className="text-info")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Anomalies Detected", className="text-secondary"), html.H3(f"{anom_count:,} ({anom_pct:.1f}%)", className="text-danger")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Model F1 Score", className="text-secondary"), html.H3(f"{f1:.4f}", className="text-success")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Current Threshold", className="text-secondary"), html.H3(f"{threshold:.4f}", className="text-warning")])))
    ]
    
    # Main Plot
    fig_main = go.Figure()
    
    # Ground truth shaded regions
    if y_test is not None:
        in_anomaly = False
        start_idx = 0
        for i, val in enumerate(y_test):
            if val == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif val == 0 and in_anomaly:
                in_anomaly = False
                fig_main.add_vrect(x0=start_idx, x1=i-1, fillcolor="red", opacity=0.2, line_width=0, layer="below")
        if in_anomaly:
            fig_main.add_vrect(x0=start_idx, x1=len(y_test)-1, fillcolor="red", opacity=0.2, line_width=0, layer="below")

    # Line plot
    fig_main.add_trace(go.Scatter(x=df["window_id"], y=df["reconstruction_error"], mode='lines', name='Error', line=dict(color='#1f77b4', width=1.5)))
    
    # Anomaly markers
    anoms = df[df["is_anomaly"]]
    fig_main.add_trace(go.Scatter(x=anoms["window_id"], y=anoms["reconstruction_error"], mode='markers', name='Anomaly Prediction', marker=dict(color='red', size=6, symbol='x')))
    
    # Threshold line
    fig_main.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_main.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", margin=dict(l=40, r=40, t=40, b=40))
    
    # Dist Plot
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=df[~df["is_anomaly"]]["reconstruction_error"], name='Normal', marker_color='#1f77b4', opacity=0.7))
    fig_dist.add_trace(go.Histogram(x=df[df["is_anomaly"]]["reconstruction_error"], name='Anomaly', marker_color='red', opacity=0.7))
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig_dist.update_layout(barmode='overlay', template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", margin=dict(l=40, r=40, t=40, b=40))
    
    # Heatmap Plot
    n_segments = 50
    df["segment"] = pd.cut(df["window_id"], bins=n_segments, labels=False)
    heatmap_data = df.groupby(["severity", "segment"]).size().unstack(fill_value=0)
    for sev in ["none", "low", "medium", "high"]:
        if sev not in heatmap_data.index:
            heatmap_data.loc[sev] = 0
    heatmap_data = heatmap_data.loc[["high", "medium", "low", "none"]]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        y=heatmap_data.index,
        x=[f"Seg {i}" for i in range(n_segments)],
        colorscale="Reds"
    ))
    fig_heatmap.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", margin=dict(l=40, r=40, t=40, b=40))
    
    # Sensor Plot
    sensor_errors = compute_sensor_errors(X_test, predictions, model, device)
    fig_sensor = go.Figure()
    if sensor_errors:
        fig_sensor.add_trace(go.Bar(x=list(sensor_errors.keys()), y=list(sensor_errors.values()), marker_color='#e34c26'))
    fig_sensor.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#0d1117", margin=dict(l=40, r=40, t=40, b=40), xaxis_tickangle=-45)
    
    # Table
    table_data = [
        {"Model": "LSTM Autoencoder", "Precision": 0.9317, "Recall": 0.6441, "F1": 0.7616},
        {"Model": "Isolation Forest", "Precision": 0.3698, "Recall": 1.0000, "F1": 0.5399}
    ]
    table = dash_table.DataTable(
        data=table_data,
        style_header={'backgroundColor': '#30363d', 'color': 'white', 'fontWeight': 'bold', 'border': '1px solid #30363d'},
        style_data={'backgroundColor': '#161b22', 'color': 'white', 'border': '1px solid #30363d'},
        style_cell={'textAlign': 'center', 'padding': '10px'}
    )
    
    return kpis, fig_main, fig_dist, fig_heatmap, fig_sensor, table

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

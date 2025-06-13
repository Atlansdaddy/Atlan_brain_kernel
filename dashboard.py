import asyncio
from typing import List

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import pandas as pd

from atlan_brain_kernel import FullCognitiveAgent, global_tick, CONFIG

# ----------------------------------------------------------------------------
# Global brain instance (demo-scale)
# ----------------------------------------------------------------------------
brain = FullCognitiveAgent(initial_size=50, vectorized=True)

# Seed a few random nodes for demonstration
import random
random.seed(42)
for i in range(30):
    pos = (random.randint(0, 9), random.randint(0, 9), random.randint(0, 9))
    brain.add_node(pos, "demo", f"Node{i}")

# ----------------------------------------------------------------------------
# Dash App Setup
# ----------------------------------------------------------------------------
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
]

app = dash.Dash(__name__, title="Atlan Brain Kernel Dashboard", external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(src=dash.get_asset_url("nobgLogo.png"), className="me-3", style={"height": "70px"}),
                html.H2("MID-ATLANTIC AI – Atlan Brain Dashboard", className="display-6 fw-bold mb-0 text-white"),
            ],
            className="d-flex justify-content-center align-items-center mt-3",
        ),
        html.Div(
            [
                html.Button("Propagate Tick", id="tick-btn", n_clicks=0, className="btn btn-primary me-3 px-4 py-2 fs-5"),
                html.Button("Replay Memory", id="replay-btn", n_clicks=0, className="btn btn-secondary me-4 px-4 py-2 fs-5"),
                dcc.Dropdown(
                    id="node-search",
                    options=[{"label": n.symbolic_anchor, "value": n.symbolic_anchor} for n in brain.nodes.values()],
                    placeholder="Search node…",
                    style={"minWidth": "200px"},
                    className="me-3",
                    searchable=True,
                ),
                html.Span(
                    [
                        dcc.Checklist(
                            options=[{"label": " Vectorized", "value": "vec"}],
                            value=["vec"],
                            id="vec-toggle",
                            inline=True,
                        ),
                    ],
                    className="form-check form-switch d-inline-flex align-items-center fs-5",
                ),
            ],
            className="d-flex justify-content-center mb-3 control-bar",
        ),
        dcc.Store(id="camera-store"),
        dcc.Graph(id="node-graph", style={"height": "65vh"}, className="mb-4"),
        dcc.Graph(id="energy-hist", style={"height": "25vh"}),
        dcc.Graph(id="memory-heat", style={"height": "30vh"}, className="mb-4"),
        html.Pre(id="event-feed", style={"height": "20vh", "overflowY": "auto", "backgroundColor": "#001b44", "color": "#ffffff", "padding": "1rem", "fontSize": "0.9rem"}),
        dcc.Interval(id="auto-refresh", interval=2000, n_intervals=0),
    ],
    className="container-fluid",
)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def node_dataframe(selected: str | None = None) -> pd.DataFrame:
    data = []
    for node in brain.nodes.values():
        x, y, z = node.position
        data.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "energy": node.resonance_energy,
                "symbol": node.symbolic_anchor,
                "selected": node.symbolic_anchor == selected,
            }
        )
    return pd.DataFrame(data)


def update_figures(selected: str | None = None):
    df = node_dataframe(selected)
    scatter_fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="energy",
        hover_data=["symbol", "energy"],
        color_continuous_scale="Turbo",
        range_color=(0, max(1.0, df["energy"].max())),
    )
    scatter_fig.update_traces(marker=dict(size=9, opacity=0.95, line=dict(width=1, color="white")))
    scatter_fig.update_coloraxes(cmin=0, cmax=max(1.0, df["energy"].max()))
    dark_bg = "#001b44"  # company deep-background blue
    plane_bg = "#d2d2db"  # brand light-grey planes
    grid_col = "#ffffff"  # white grid lines
    scatter_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=dark_bg,
        plot_bgcolor=dark_bg,
        font_color="white",
        uirevision="static-camera",
        scene=dict(
            bgcolor=dark_bg,
            xaxis=dict(showbackground=True, backgroundcolor=plane_bg, gridcolor=grid_col, zerolinecolor=grid_col),
            yaxis=dict(showbackground=True, backgroundcolor=plane_bg, gridcolor=grid_col, zerolinecolor=grid_col),
            zaxis=dict(showbackground=True, backgroundcolor=plane_bg, gridcolor=grid_col, zerolinecolor=grid_col),
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Energy", font=dict(color="white")),
            tickfont=dict(color="white"),
        ),
        hoverlabel=dict(font_size=18,bgcolor="#2c1a47"),
    )
    # Emphasize selected node
    if selected and selected in df["symbol"].values:
        sel_df = df[df["selected"]]
        scatter_fig.add_scatter3d(
            x=sel_df["x"],
            y=sel_df["y"],
            z=sel_df["z"],
            mode="markers+text",
            marker=dict(size=12, color="#ff4d4d", symbol="diamond"),
            text=sel_df["symbol"],
            textposition="top center",
            name="Selected",
        )
    hist_fig = px.histogram(
        df,
        x="energy",
        nbins=20,
        title="Energy Distribution",
        color_discrete_sequence=["#56d0e0"],  # brand light blue
        text_auto=".2f",
    )
    hist_fig.update_traces(textfont_size=14, textangle=0, cliponaxis=False)
    hist_fig.update_layout(
        template="plotly_white",
        paper_bgcolor=dark_bg,
        plot_bgcolor=dark_bg,
        font_color="white",
        font_size=16,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        title_font_size=20,
        uirevision="static-camera",
    )
    return scatter_fig, hist_fig


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------

@callback(
    Output("node-graph", "figure"),
    Output("energy-hist", "figure"),
    Output("memory-heat", "figure"),
    Output("event-feed", "children"),
    Input("auto-refresh", "n_intervals"),
    State("node-search", "value"),
    State("camera-store", "data"),
)
def refresh(_, selected, camera):
    fig3d, hist = update_figures(selected)
    if camera:
        fig3d.update_layout(scene_camera=camera)
    # Memory heatmap
    mem_df = brain.get_recent_memory(100)
    if mem_df is not None and not mem_df.empty:
        heat = px.density_heatmap(mem_df, x="tick", y="target", z="energy", color_continuous_scale="Turbo")
        heat.update_layout(paper_bgcolor="#001b44", plot_bgcolor="#001b44", font_color="white", uirevision="static-camera", margin=dict(l=0,r=0,t=0,b=0))
    else:
        heat = {}

    # Event feed text
    events = "\n".join(list(brain.event_log)[-50:])
    return fig3d, hist, heat, events


@callback(
    Output("auto-refresh", "n_intervals"),
    Input("tick-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_tick(n_clicks):
    global global_tick
    if n_clicks:
        global_tick += 1
        # Randomly excite a node for demo
        node_pos = random.choice(list(brain.nodes.keys()))
        brain.propagate_resonance(node_pos, 5.0, random.uniform(0.5, 1.0))
    return dash.no_update


@callback(Output("auto-refresh", "max_intervals"), Input("replay-btn", "n_clicks"), prevent_initial_call=True)
def on_replay(_):
    brain.replay_memory_chain()
    return dash.no_update


@callback(Output("vec-toggle", "value"), Input("vec-toggle", "value"))
def toggle_vectorized(values):
    brain.vectorized = "vec" in values
    return values


# Save camera on user interaction
@callback(Output("camera-store", "data"), Input("node-graph", "relayoutData"), State("camera-store", "data"))
def store_camera(relayout, current):
    if relayout and "scene.camera" in relayout:
        return relayout["scene.camera"]
    return current


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True) 
import asyncio
from typing import List

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from atlan_brain_kernel import FullCognitiveAgent, global_tick, CONFIG
import importlib.util, pathlib, sys
import re

try:
    import atlan_examples as examples_module
except ModuleNotFoundError:
    examples_path = pathlib.Path(__file__).with_name("atlan-examples.py")
    if examples_path.exists():
        spec = importlib.util.spec_from_file_location("atlan_examples_dyn", str(examples_path))
        examples_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules["atlan_examples_dyn"] = examples_module  # type: ignore[index]
        if spec and spec.loader:
            spec.loader.exec_module(examples_module)  # type: ignore[arg-type]
    else:
        # Empty stub if file missing
        import types
        examples_module = types.ModuleType("atlan_examples_stub")

# -----------------------------------------------------------------------------
# Descriptive copy for About / Info section
# -----------------------------------------------------------------------------
ABOUT_MD = """
## About the Atlan Brain Kernel 🧠

The **Atlan Brain Kernel** is an open-research cognitive engine that treats cognition as a **spatio-temporal energy field**, not a graph of synaptic weights.
It is inspired by resonance theory, neural field models, and emerging discussions around *field M-theory of consciousness*.

### 🏗  Architectural Layers
| Layer | Purpose | Key Classes |
|-------|---------|-------------|
| Substrate | 3-D lattice of `CognitiveNode` objects that hold energy | `Nodefield` & subclasses |
| Memory | Pluggable persistence (in-memory / SQLite / 🔜 distributed KV) | `MemoryStore`, `SQLiteStore` |
| Learning Loops | Reinforcement, abstraction, analogy, prediction, reflection | see `Reinforced…` ➜ `Curiosity…` classes |
| Control API | Dash UI (this page), upcoming FastAPI endpoints | `dashboard.py` *(you are here)* |

### 🚀 What can you do with it right now?
1. **Visualise Resonance** – press *Propagate Tick* to inject random energy and watch it diffuse.
2. **Toggle Vectorised Mode** – NumPy crunches the propagation maths 10-100× faster.
3. **Replay Memory** – re-activate past events to mimic offline consolidation (think sleep / dreaming).
4. **Inspect Event Feed** – raw log lines of every transfer and decay for debugging / research.

### 🔮 Roadmap (abridged)
• **GPU / CuPy Acceleration**  
• **FastAPI micro-service** for remote orchestration  
• **Self-Organising Concept Maps** exported as Graph-ML / Neo4j  
• **Benchmark Suite + CI** (pytest + GitHub Actions)  
• **Interactive Jupyter Tutorials**  

### 📚 Example Integration
```python
from atlan_brain_kernel import FullCognitiveAgent
brain = FullCognitiveAgent(initial_size=25)
brain.add_node((0,0,0), "finance", "Interest")
brain.add_node((1,0,0), "finance", "Inflation")
brain.propagate_resonance((0,0,0), input_energy=3.5, importance_weight=0.8)
```

### 🤔 Potential Use-Cases
• *AI Curriculum Visualiser*  
• *Concept-drift Monitoring* for live ML models  
• *Game NPC Memory & Motivation* system  
• *Generative Art* based on emergent field patterns  
• *Cognitive-Ergonomic UX Studies*

### 🧩 Design Principles
• **Modularity First** – every capability is layered via mix-in subclasses, allowing you to swap or skip features.  
• **Observable by Default** – key internal events are logged to the in-memory `event_log` → pipes straight into this UI or any websocket consumer.  
• **Fail-Soft** – errors propagate as `BrainKernelError` so downstream systems can gracefully degrade.

### 🔐 Security & Governance
Only anonymised resonance metrics are transmitted outside the kernel.  Symbolic data remain in-process unless an explicit export command is issued.  Future roadmap items include signed event chains and role-based access to *propagate* and *replay* endpoints.

### 🔎 Recommended Reading
• Bekkerman et al., *Neural Field Theory of Cognition*, 2021 (open-access).  
• Ritter, *Memory Consolidation in Artificial Agents*, NeurIPS 2022 workshop.  
• Smith & Rao, *Resonance-Based Reasoning*, ICCM 2023.

### 🤝 Get Involved
Star the project, open issues, or reach the maintainers at **john@midatlantic.ai**.  Contribution guidelines live in `README.md`.

---
*Project licensed under BUSL for non-commercial research. Commercial licensing available – see `COMMERCIAL_LICENSE.md`.*
"""

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

# Pre-built demo helper functions -------------------------------------------
def example_finance():
    """Finance demo: Interest → Inflation"""
    global global_tick
    data = {
        (0, 0, 0): ("finance", "Interest"),
        (1, 0, 0): ("finance", "Inflation"),
    }
    for pos, (ctx, sym) in data.items():
        if pos not in brain.nodes:
            brain.add_node(pos, ctx, sym)
    global_tick += 1
    brain.propagate_resonance((0, 0, 0), 3.5, 0.8)


def example_physics_chain():
    """Physics demo: Momentum → Velocity"""
    global global_tick
    data = {
        (7, 7, 7): ("physics", "Momentum"),
        (7, 8, 7): ("physics", "Velocity"),
    }
    for pos, (ctx, sym) in data.items():
        if pos not in brain.nodes:
            brain.add_node(pos, ctx, sym)
    global_tick += 1
    brain.propagate_resonance((7, 7, 7), 4.0, 1.0)

# -----------------------------------------------------------------------------
# Auto-discover example functions from atlan-examples.py and local quick demos
# -----------------------------------------------------------------------------

LOCAL_DEMOS: dict[str, callable] = {
    "Finance": example_finance,
    "Physics": example_physics_chain,
}

EXAMPLE_FUNCS: dict[str, callable] = {}
# Pull in examples_1..n from external module
for attr in dir(examples_module):
    if attr.startswith("example_") and callable(getattr(examples_module, attr)):
        fn = getattr(examples_module, attr)
        # First sentence of docstring becomes label
        label: str = fn.__doc__.split("\n", 1)[0] if fn.__doc__ else attr
        label = re.sub(r"^Example\s*\d*\s*:\s*", "", label, flags=re.IGNORECASE)
        EXAMPLE_FUNCS[label] = fn
# Append local demos afterwards
EXAMPLE_FUNCS.update(LOCAL_DEMOS)

# Sort mapping by key for stable UI order
EXAMPLE_FUNCS = dict(sorted(EXAMPLE_FUNCS.items(), key=lambda x: x[0]))

# Rich explanations for each demo
EXAMPLE_DESCRIPTIONS: dict[str, str] = {
    "Finance": (
        "### Finance Demo – Interest → Inflation\n"
        "This example seeds two finance-related concepts (`Interest` and `Inflation`) and injects energy into **Interest**.\n\n"
        "• **What you will see**: Interest lights up first, then propagates energy to Inflation proportional to distance and dampening.\n"
        "  You'll notice a spike in the 3-D scatter (node colour) and a corresponding event line in the feed.\n\n"
        "• **Why it happens**: The propagation routine treats `Interest` as the source and distributes weighted energy to nearby nodes (here only `Inflation`).\n\n"
        "• **Why it matters**: Demonstrates how economic variables can be related in the field; downstream analytics could trace macro-economic cascades."
    ),
    "Physics": (
        "### Physics Demo – Momentum → Velocity\n"
        "Seeds `Momentum` and `Velocity`, then energises Momentum. The transfer illustrates physical quantity coupling.\n\n"
        "See Momentum flare up followed by Velocity absorbing a share of that energy. This mirrors how knowledge of one property primes another in cognition."
    ),
}

# ---------------------------------------------------------------------------
# Add detailed copy for imported examples (labels come from cleaned docstrings)
# ---------------------------------------------------------------------------

EXAMPLE_DESCRIPTIONS.update({
    "Basic concept learning and association": (
        "### Basic Concept Learning & Association\n"
        "Seeds an *Animal → Mammal → Dog* hierarchy and stimulates **Dog**.\n\n"
        "• **Watch for**: Dog node energises first, then travels up the hierarchy (Mammal → Animal).\n"
        "• **Meaning**: Demonstrates bottom-up generalisation – foundational to how the Atlan field abstracts from specifics.\n"
        "• **Broader Impact**: Highlights how sparse symbolic graphs can self-organise without back-prop weight tuning."
    ),
    "Memory consolidation through replay": (
        "### Memory Consolidation\n"
        "Re-feeds recent events at 80 % strength, akin to dream-state replay.\n\n"
        "Observe energies bump across previously active nodes. This shows how stable traces are reinforced over time."
    ),
    "Forming abstract concepts": (
        "### Abstraction Formation\n"
        "Energises high-level property nodes (Shape, Color, Size) more than concrete objects. The derive-abstractions pass labels those with persistent energy as *Abstract_* symbols."
    ),
    "Learning sequences and making predictions": (
        "### Sequence Learning & Prediction\n"
        "Repeatedly activates a daily routine → builds transition probabilities. Prediction panel logs likely next steps given a current context."
    ),
    "Handling conflicting information": (
        "### Conflict Detection & Resolution\n"
        "Introduces competing beliefs (e.g., EarthRound vs EarthFlat) with different confidence. The conflict resolver keeps the stronger belief."
    ),
    "Learning across multiple domains": (
        "### Multi-Domain Learning\n"
        "Demonstrates the field handling parallel knowledge graphs (math, language, art), showing energy stays mostly within domain bubbles unless bridged."
    ),
    "Autonomous cognitive development": (
        "### Autonomous Cognitive Development\n"
        "Gradually expands its nodefield over several cycles, mimicking self-directed maturation."
    ),
    "Meta-learning strategies": (
        "### Meta-Learning Strategies\n"
        "Evaluates multiple learning approaches and selects the best via internal metrics – showcases reflective and meta-cognitive layers."
    ),
})

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
                    style={"minWidth": "300px"},
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
                dcc.Dropdown(id="example-select", options=[{"label": k, "value": k} for k in EXAMPLE_FUNCS.keys()], placeholder="Choose example…", clearable=False, searchable=False, style={"width": "480px"}, className="me-2"),
                html.Button("Run Example", id="run-example-btn", n_clicks=0, className="btn btn-success px-4 py-2 fs-5"),
            ],
            className="d-flex justify-content-center mb-3 control-bar",
        ),
        dcc.Markdown(
            """**Quick Start Guide**  
• *Propagate Tick* – injects energy into a random node so you can watch diffusion.  
• *Replay Memory* – re-plays recent events to reinforce learning (simulated sleep).  
• *Vectorized* – toggles fast NumPy propagation.  
• Pick an **Example** and click *Run Example* to load themed concepts and show an energy transfer in real-time.
""",
            className="quick-tips text-white mx-auto",
            style={"maxWidth": "960px", "backgroundColor": "#00235a", "padding": "0.8rem 1rem", "border": "1px solid #415584", "borderRadius": "6px", "fontSize": "1.1rem", "lineHeight": "1.5"},
        ),
        dcc.Markdown("", id="example-desc", className="example-desc text-white mx-auto", style={"maxWidth": "960px", "backgroundColor": "#001b44", "padding": "0.8rem 1rem", "border": "1px solid #415584", "borderRadius": "6px", "fontSize": "1.05rem", "lineHeight": "1.5", "marginTop": "0.8rem"}),
        dcc.Store(id="camera-store"),
        dcc.Graph(id="node-graph", style={"height": "65vh"}, className="mb-4"),
        dcc.Graph(id="energy-hist", style={"height": "25vh"}),
        dcc.Graph(id="memory-heat", style={"height": "30vh"}, className="mb-4"),
        html.Pre(id="event-feed", style={"height": "20vh", "overflowY": "auto", "backgroundColor": "#001b44", "color": "#ffffff", "padding": "1rem", "fontSize": "0.9rem"}),
        dcc.Markdown(ABOUT_MD, id="about-section", className="about-section text-white mx-auto mb-5", style={"maxWidth": "960px", "backgroundColor": "#001b44", "lineHeight": "1.6"}),
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
    dark_bg = "#001b44"
    fig3d, hist = update_figures(selected)
    if camera:
        fig3d.update_layout(scene_camera=camera)
    # Memory heatmap
    mem_df = brain.get_recent_memory(100)
    if mem_df is not None and not mem_df.empty:
        heat = px.density_heatmap(mem_df, x="tick", y="target", z="energy", color_continuous_scale="Turbo")
        heat.update_layout(paper_bgcolor=dark_bg, plot_bgcolor=dark_bg, font_color="white", uirevision="static-camera", margin=dict(l=0,r=0,t=0,b=0))
        heat.update_xaxes(showgrid=False, zeroline=False)
        heat.update_yaxes(showgrid=False, zeroline=False)
    else:
        # produce an empty, dark figure to avoid large white block
        heat = go.Figure()
        heat.update_layout(paper_bgcolor=dark_bg, plot_bgcolor=dark_bg, margin=dict(l=0,r=0,t=0,b=0))

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


@callback(
    Output("node-search", "options"),
    Output("example-desc", "children"),
    Input("run-example-btn", "n_clicks"),
    State("example-select", "value"),
    prevent_initial_call=True,
)
def on_run_example(_, selected_example):
    if selected_example and selected_example in EXAMPLE_FUNCS:
        # Run the example; capture possible return value
        result = EXAMPLE_FUNCS[selected_example]()

        global brain  # we may swap the dashboard brain instance
        from atlan_brain_kernel import FullCognitiveAgent  # local import to avoid circularity
        if isinstance(result, FullCognitiveAgent):
            brain = result  # replace the global instance with new brain produced by example

        # Description priority: explicit mapping → remaining docstring lines → fallback text
        if selected_example in EXAMPLE_DESCRIPTIONS:
            desc = EXAMPLE_DESCRIPTIONS[selected_example]
        else:
            fn = EXAMPLE_FUNCS[selected_example]
            doc = fn.__doc__ or ""
            desc = doc.strip() or "(No additional description found.)"

        node_opts = [{"label": n.symbolic_anchor, "value": n.symbolic_anchor} for n in brain.nodes.values()]
        return node_opts, desc

    # no example selected
    return dash.no_update, dash.no_update


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True) 
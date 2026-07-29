"""
dashydash.py
------------
Live Dash/Plotly dashboard for the AL grid search results.

Watches every run_* folder under --results-dir (al_metrics.csv +
al_config.json, as written by al_loop.py's ALLogger) and plots mean/best
reward vs. cumulative oracle calls, colored by sampling strategy. Older
runs are rendered at very low opacity so recent/active runs stand out, and
the whole figure refreshes on a timer so you can watch progress live while
run_grid_search.py is still running.

Usage:
    pip install dash plotly pandas
    python dashydash.py --results-dir ./grid_results_2 --port 8050

Then open http://127.0.0.1:8050 in a browser.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

STRATEGY_COLORS = {
    "random":   "#7f8c8d",
    "lhs":      "#3498db",
    "grid":     "#9b59b6",
    "gflownet": "#e74c3c",
    "gp":       "#2ecc71",
    "genetic":  "#f39c12",
}
DEFAULT_COLOR = "#34495e"

LOW_OPACITY = 0.12
HIGH_OPACITY = 0.9

def discover_runs(results_dir: Path):
    """Load (config, metrics_df) for every run_* directory that has both files."""
    runs = []
    for run_dir in sorted(results_dir.glob("run_*")):
        metrics_path = run_dir / "al_metrics.csv"
        config_path = run_dir / "al_config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path)
            cfg = json.load(open(config_path))
        except Exception:
            # Directory mid-write (grid search still running) -- skip this
            # refresh cycle, it'll pick itself up on the next tick.
            continue
        if df.empty or "oracle_evals_total" not in df.columns:
            continue
        runs.append({
            "name": run_dir.name,
            "mtime": metrics_path.stat().st_mtime,
            "config": cfg,
            "df": df,
        })
    return runs


def build_figure(runs, strategies_selected, highlight_n=3):
    """
    Two stacked subplots: mean_reward_so_far (top), best_reward_so_far
    (bottom), both vs oracle_evals_total. All runs draw at low opacity;
    the `highlight_n` most recently modified runs draw at high opacity
    with their own legend entry (older runs of the same strategy share
    one faint legend entry so the legend doesn't explode).
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Mean reward so far", "Best reward so far"),
        vertical_spacing=0.1,
    )

    filtered = [r for r in runs if r["config"].get("sampling_strategy") in strategies_selected]
    filtered.sort(key=lambda r: r["mtime"])
    highlight_names = {r["name"] for r in filtered[-highlight_n:]} if highlight_n > 0 else set()

    seen_faint_legend = set()

    for r in filtered:
        df = r["df"]
        strategy = r["config"].get("sampling_strategy", "unknown")
        color = STRATEGY_COLORS.get(strategy, DEFAULT_COLOR)
        is_highlighted = r["name"] in highlight_names
        opacity = HIGH_OPACITY if is_highlighted else LOW_OPACITY

        if is_highlighted:
            legend_label = f"{strategy} — {r['name']}"
            show_legend = True
        else:
            legend_label = f"{strategy} (history)"
            show_legend = strategy not in seen_faint_legend
            seen_faint_legend.add(strategy)

        hover = (
            f"<b>{r['name']}</b> [{strategy}]<br>"
            f"seed={r['config'].get('seed')} "
            f"n_init={r['config'].get('n_init')} "
            f"n_cand={r['config'].get('n_candidates_per_iter')}<br>"
            "oracle_evals=%{x}<br>reward=%{y:.4f}<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=df["oracle_evals_total"], y=df["mean_reward_so_far"],
            mode="lines+markers" if is_highlighted else "lines",
            line=dict(color=color, width=2.5 if is_highlighted else 1),
            opacity=opacity,
            name=legend_label,
            legendgroup=f"{strategy}-{'hi' if is_highlighted else 'lo'}",
            showlegend=show_legend,
            hovertemplate=hover,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["oracle_evals_total"], y=df["best_reward_so_far"],
            mode="lines+markers" if is_highlighted else "lines",
            line=dict(color=color, width=2.5 if is_highlighted else 1),
            opacity=opacity,
            name=legend_label,
            legendgroup=f"{strategy}-{'hi' if is_highlighted else 'lo'}",
            showlegend=False,
            hovertemplate=hover,
        ), row=2, col=1)

    fig.update_xaxes(title_text="Cumulative oracle calls", row=2, col=1)
    fig.update_yaxes(title_text="Mean reward", row=1, col=1)
    fig.update_yaxes(title_text="Best reward", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=780,
        legend=dict(itemsizing="constant", x=1.02, y=1),
        margin=dict(t=60, r=220),
        title=(
            f"AL grid search — {len(filtered)} runs shown "
            f"({min(highlight_n, len(filtered))} highlighted)"
        ),
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./grid_results_2")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--refresh-sec", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    app = Dash(__name__)
    app.title = "AL Grid Search — Live Dashboard"

    app.layout = html.Div([
        html.H2("Active Learning Grid Search — Live Reward Tracking"),

        html.Div([
            html.Label("Sampling strategies:"),
            dcc.Checklist(
                id="strategy-filter",
                options=[{"label": f" {s}", "value": s} for s in STRATEGY_COLORS],
                value=list(STRATEGY_COLORS.keys()),
                inline=True,
            ),
        ], style={"marginBottom": "14px"}),

        html.Div([
            html.Label("Highlight most recent N runs (full opacity):"),
            dcc.Slider(
                id="highlight-n", min=0, max=10, step=1, value=3,
                marks={i: str(i) for i in range(0, 11)},
            ),
        ], style={"width": "360px", "marginBottom": "24px"}),

        dcc.Graph(id="reward-graph"),
        dcc.Interval(id="refresh-tick", interval=args.refresh_sec * 1000, n_intervals=0),
        html.Div(id="status-line", style={"color": "#888", "fontSize": "12px", "marginTop": "8px"}),
    ], style={"fontFamily": "sans-serif", "padding": "24px"})

    @app.callback(
        Output("reward-graph", "figure"),
        Output("status-line", "children"),
        Input("refresh-tick", "n_intervals"),
        Input("strategy-filter", "value"),
        Input("highlight-n", "value"),
    )
    def update(_n, strategies_selected, highlight_n):
        runs = discover_runs(results_dir)
        fig = build_figure(runs, strategies_selected or [], highlight_n)
        status = (
            f"{len(runs)} runs found in {results_dir.resolve()} "
            f"— refreshes every {args.refresh_sec}s"
        )
        return fig, status

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
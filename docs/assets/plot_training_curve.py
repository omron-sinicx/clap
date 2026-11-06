# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=5,<6",  # kaleido 0.2 (bundled Chromium) needs plotly 5
#   "kaleido==0.2.1",
#   "pandas",
# ]
# ///
"""LIBERO success rate per training epoch for CLAP trained with this repository.

Reads docs/assets/training_curve.csv (written by collect_training_curve.py) and saves
docs/assets/training_curve.png. Dashed lines: the paper's 17k-step results (Table 1).
Shaded: epoch 7, the default of configs/. Arrow: training time to epoch 7 on 8 x H200.

    pixi run --script docs/assets/plot_training_curve.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent

# ── Theme: vintage (white background) ────────────────────────────────
THEME = dict(
    colorway=[
        "#D87C7C", "#919E8B", "#D7AB82", "#61A0A8",
        "#724E58", "#4B565B", "#CC7E63", "#EFA18D",
    ],
    highlight=["#F0D0D0", "#F0E4D0", "#DEE3D9"],
    bg="#FFFFFF",
    gridcolor="rgba(0,0,0,0.06)",
    font_color="#333333",
)


def make_template(t: dict) -> go.layout.Template:
    axis = dict(
        showline=True,
        mirror=False,  # left and bottom axes only, no frame
        zeroline=False,
        linecolor="rgba(0,0,0,0.2)",
        linewidth=0.5,
        ticks="outside",
        ticklen=4,
        tickcolor="rgba(0,0,0,0.2)",
        title=dict(standoff=12),
    )
    return go.layout.Template(
        layout=go.Layout(
            colorway=t["colorway"],
            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color=t["font_color"], size=13),
            title=dict(font=dict(size=16), x=0.0, xanchor="left"),
            plot_bgcolor=t["bg"],
            paper_bgcolor=t["bg"],  # opaque, so the PNG stays readable on dark pages
            xaxis=dict(showgrid=False, tickfont=dict(size=11), **axis),
            yaxis=dict(showgrid=True, gridcolor=t["gridcolor"], gridwidth=0.5, tickfont=dict(size=11), **axis),
            legend=dict(
                bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11),
                orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0,
            ),
            margin=dict(l=60, r=20, t=70, b=55),
        ),
    )


TPL = make_template(THEME)

# ── Data ─────────────────────────────────────────────────────────────
SUITES = [
    ("libero_spatial", "LIBERO-Spatial"),
    ("libero_object", "LIBERO-Object"),
    ("libero_goal", "LIBERO-Goal"),
    ("libero_10", "LIBERO-Long"),
    ("average", "Average"),
]
# Paper, Table 1 (17k steps of the VLA-0 codebase = 8 epochs here)
PAPER = {
    "CLAP-0.8B": {"libero_spatial": 92.6, "libero_object": 97.8, "libero_goal": 88.4, "libero_10": 79.6, "average": 89.6},
    "CLAP-2B": {"libero_spatial": 93.0, "libero_object": 97.4, "libero_goal": 90.8, "libero_10": 82.0, "average": 90.8},
    "CLAP-4B": {"libero_spatial": 88.0, "libero_object": 97.4, "libero_goal": 86.6, "libero_10": 67.6, "average": 84.9},
}
# teal, rose and olive: the most distinct colors in the vintage colorway
COLORS = {"CLAP-0.8B": THEME["colorway"][3], "CLAP-2B": THEME["colorway"][0], "CLAP-4B": THEME["colorway"][1]}
MAX_EPOCH = 8  # = the paper's 17k steps; the CSV may hold later epochs

df = pd.read_csv(HERE / "training_curve.csv")
df = df[df["epoch"] <= MAX_EPOCH]
# Average over the 4 suites, only for checkpoints evaluated on all of them
full = df.groupby(["model", "epoch", "steps", "train_hours"]).filter(lambda g: len(g) == 4)
avg = (
    full.groupby(["model", "epoch", "steps", "train_hours"], as_index=False)["success_rate"].mean()
    .assign(suite="average")
)
df = pd.concat([df, avg], ignore_index=True)

# ── Figure ───────────────────────────────────────────────────────────
fig = make_subplots(rows=1, cols=len(SUITES), subplot_titles=[t for _, t in SUITES], horizontal_spacing=0.045)
fig.update_layout(template=TPL)
XRANGE = [0.5, MAX_EPOCH + 0.5]

for col, (suite, _) in enumerate(SUITES, start=1):
    for model, color in COLORS.items():
        d = df[(df["model"] == model) & (df["suite"] == suite)].sort_values("epoch")
        fig.add_trace(go.Scatter(
            x=d["epoch"], y=d["success_rate"], mode="lines+markers",
            name=f"{model[5:]} (this repo)", legendgroup=model, showlegend=(col == 1),
            line=dict(color=color, width=2), marker=dict(size=6, color=color),
            customdata=d["train_hours"], hovertemplate="%{y:.1f}% (%{customdata:.1f} h)<extra></extra>",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=XRANGE, y=[PAPER[model][suite]] * 2, mode="lines",
            name=f"{model[5:]} paper (17k steps)", legendgroup=model + "-paper", showlegend=(col == 1),
            line=dict(color=color, width=1.2, dash="dash"), hoverinfo="skip",
        ), row=1, col=col)
    fig.update_xaxes(range=XRANGE, tickvals=list(range(1, MAX_EPOCH + 1)), title_text="epoch" if col == 3 else None, row=1, col=col)

# Epoch 7 (the default of configs/): shade it and show the training time on 8 x H200
CONFIG_EPOCHS = 7
for col in range(1, len(SUITES) + 1):
    fig.add_vrect(x0=CONFIG_EPOCHS - 0.4, x1=CONFIG_EPOCHS + 0.4, fillcolor=THEME["highlight"][1],
                  opacity=0.6, line_width=0, layer="below", row=1, col=col)
last = df[(df["suite"] == "average") & (df["epoch"] == CONFIG_EPOCHS)].set_index("model")
hours = "<br>".join(
    f"<span style='color:{c}'>{m[5:]}: {last.loc[m, 'train_hours']:.1f} h</span>" for m, c in COLORS.items()
)
fig.add_annotation(
    x=CONFIG_EPOCHS, y=last["success_rate"].min(), xref=f"x{len(SUITES)}", yref=f"y{len(SUITES)}",
    text=hours, align="left", font=dict(size=11),
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor=THEME["font_color"],
    ax=-45, ay=95, standoff=4,
)

fig.update_yaxes(title_text="success rate [%]", row=1, col=1)
fig.update_annotations(font_size=13)
fig.update_layout(width=1300, height=290, margin_r=45)

out = HERE / "training_curve.png"
fig.write_image(out, scale=3)
print(f"Saved: {out}")

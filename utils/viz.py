"""
Plotly helpers.

Plotly is used deliberately here (instead of the usual Vega default) because
the whole point of this app is *animated* explanation: animation frames with a
play button, animated 3D loss surfaces and animated cross-validation diagrams
are things Plotly does natively.

Everything is forced onto one very light template so the figures match the
pastel theme.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

INK = "#17263F"
GRID = "#DCE7F4"
AXIS = "#A9BBD0"
PAPER = "rgba(0,0,0,0)"
PLOT = "rgba(255,255,255,0.62)"

C_BLUE = "#3B7DD8"
C_GREEN = "#2FB58B"
C_AMBER = "#F0A33C"
C_VIOLET = "#9B6BD6"
C_ROSE = "#EB6F92"
C_CYAN = "#2FA8C7"
C_SLATE = "#8A94A6"

SEQ_LIGHT = [
    [0.00, "#F5FAFF"],
    [0.15, "#E0EFFD"],
    [0.30, "#C6E2FB"],
    [0.45, "#A6D0F7"],
    [0.60, "#83BAF1"],
    [0.75, "#5EA0E8"],
    [0.88, "#3B7DD8"],
    [1.00, "#2A62B4"],
]


def style_fig(
    fig: go.Figure,
    *,
    height: int = 420,
    xtitle: str = "",
    ytitle: str = "",
    title: str = "",
    legend: bool = True,
    legend_top: bool = True,
) -> go.Figure:
    """Apply the shared light look to a figure."""
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color=INK), x=0.02, xanchor="left"),
        margin=dict(l=58, r=22, t=52 if title else 30, b=52),
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font=dict(family="Cairo, system-ui, sans-serif", size=12.5, color=INK),
        showlegend=legend,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,.96)",
            bordercolor="#C9DCF2",
            font=dict(color=INK, size=12),
        ),
        hovermode="closest",
    )
    if legend and legend_top:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.015,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,.55)",
                bordercolor="#DCE7F4",
                borderwidth=1,
                font=dict(size=11.5),
            )
        )
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS,
        showline=True,
        mirror=False,
        ticks="outside",
        tickcolor=AXIS,
        ticklen=4,
        tickfont=dict(size=11.5, color="#5B7490"),
        title_font=dict(size=12.5, color="#3C5572"),
    )
    fig.update_xaxes(**axis, title_text=xtitle)
    fig.update_yaxes(**axis, title_text=ytitle)
    return fig


def play_buttons(
    frame_ms: int = 120,
    *,
    label_play: str = "Play",
    label_pause: str = "Pause",
    x: float = 0.0,
    y: float = -0.22,
) -> list[dict]:
    """A compact play/pause control for animated figures."""
    return [
        dict(
            type="buttons",
            direction="right",
            showactive=False,
            x=x,
            y=y,
            xanchor="left",
            yanchor="top",
            pad=dict(t=6, r=6),
            bgcolor="rgba(255,255,255,.92)",
            bordercolor="#C9DCF2",
            borderwidth=1,
            font=dict(size=12, color="#2A62B4"),
            buttons=[
                dict(
                    label="▶  " + label_play,
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_ms, redraw=True),
                            transition=dict(duration=frame_ms // 2),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏸  " + label_pause,
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ]


def frame_slider(labels: list[str], prefix: str = "", y: float = -0.14) -> list[dict]:
    """Slider bound to animation frames."""
    return [
        dict(
            active=0,
            x=0.17,
            y=y,
            len=0.80,
            xanchor="left",
            yanchor="top",
            pad=dict(t=6, b=4),
            currentvalue=dict(
                prefix=prefix,
                visible=True,
                xanchor="right",
                font=dict(size=12, color="#2A62B4"),
            ),
            transition=dict(duration=60),
            bgcolor="#DCE7F4",
            bordercolor="#C9DCF2",
            activebgcolor="#3B7DD8",
            tickcolor="#C9DCF2",
            font=dict(size=10),
            steps=[
                dict(
                    label=lab,
                    method="animate",
                    args=[
                        [lab],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                )
            for lab in labels
            ],
        )
    ]


def soft_band(
    fig: go.Figure, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, colour: str, name: str
) -> go.Figure:
    """Shaded confidence-style band."""
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=_rgba(colour, 0.14),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name=name,
            showlegend=False,
        )
    )
    return fig


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


rgba = _rgba

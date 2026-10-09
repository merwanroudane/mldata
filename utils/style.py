"""
Global styling helpers for the app.

The user explicitly asked for:
  * a right-hand (RTL) sidebar,
  * light / pastel colours (no dark theme),
  * CSS animations and animated explanatory cards.

So custom CSS is intentional here. Everything colour-related that can live
in `.streamlit/config.toml` lives there; this module only adds the RTL flip
and the motion layer.
"""

from __future__ import annotations

import streamlit as st

AUTHOR = "Dr Merwan Roudane"

# ----------------------------------------------------------------------
# Palette (kept in sync with .streamlit/config.toml)
# ----------------------------------------------------------------------
PALETTE = {
    "blue": "#3B7DD8",
    "blue_soft": "#E3F0FE",
    "green": "#2FB58B",
    "green_soft": "#DFF6EE",
    "amber": "#F0A33C",
    "amber_soft": "#FDF0DC",
    "violet": "#9B6BD6",
    "violet_soft": "#EEE6FB",
    "rose": "#EB6F92",
    "rose_soft": "#FDE6EE",
    "cyan": "#2FA8C7",
    "cyan_soft": "#DEF3F8",
    "slate": "#8A94A6",
    "slate_soft": "#EDF1F6",
    "ink": "#17263F",
    "paper": "#FBFDFF",
}

TRAIN_C = PALETTE["blue"]
VAL_C = PALETTE["amber"]
TEST_C = PALETTE["rose"]


_GLOBAL_CSS = """
<style>
/* ==================================================================
   1.  RIGHT-HAND SIDEBAR  +  RTL READING DIRECTION
   ================================================================== */
[data-testid="stAppViewContainer"] { direction: rtl; }
[data-testid="stSidebar"]          { direction: rtl; }
[data-testid="stHeader"]           { direction: rtl; }
[data-testid="stSidebarCollapsedControl"] { direction: rtl; }

/* Arabic body copy reads from the right */
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] blockquote { text-align: right; }

/* but code, maths, charts and tables must stay left-to-right */
[data-testid="stCodeBlock"], pre, code, kbd, samp { direction: ltr; text-align: left; }
.katex, .katex-display, .katex-html { direction: ltr; text-align: center; }
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"],
.js-plotly-plot, [data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stMetric"] { direction: ltr; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { text-align: left; }
.ltr { direction: ltr !important; text-align: left !important; }

/* native markdown tables inherit RTL badly, keep cells right-aligned */
[data-testid="stMain"] table { direction: rtl; }
[data-testid="stMain"] table th,
[data-testid="stMain"] table td { text-align: right; }

/* ==================================================================
   2.  SOFT BACKGROUND WASH  (very light, slowly drifting)
   ================================================================== */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 88% -8%,  #E7F1FD 0%, rgba(231,241,253,0) 62%),
    radial-gradient(900px  480px at 6%  104%, #E6F7F1 0%, rgba(230,247,241,0) 62%),
    radial-gradient(700px  420px at 46% 42%,  #FBF1E4 0%, rgba(251,241,228,0) 70%),
    #FBFDFF;
  background-attachment: fixed;
}

/* ==================================================================
   3.  MOTION LAYER
   ================================================================== */
@keyframes mr-rise    { from { opacity:0; transform: translateY(16px); }
                        to   { opacity:1; transform: translateY(0); } }
@keyframes mr-fade    { from { opacity:0; } to { opacity:1; } }
@keyframes mr-slide-r { from { opacity:0; transform: translateX(26px); }
                        to   { opacity:1; transform: translateX(0); } }
@keyframes mr-pop     { 0%{transform:scale(.86);opacity:0}
                        60%{transform:scale(1.04)}
                        100%{transform:scale(1);opacity:1} }
@keyframes mr-pulse   { 0%,100%{ box-shadow:0 0 0 0 rgba(59,125,216,.30) }
                        50%    { box-shadow:0 0 0 14px rgba(59,125,216,0) } }
@keyframes mr-float   { 0%,100%{ transform: translateY(0) } 50%{ transform: translateY(-7px) } }
@keyframes mr-sheen   { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }
@keyframes mr-grow-x  { from { transform: scaleX(0) } to { transform: scaleX(1) } }
@keyframes mr-grow-y  { from { transform: scaleY(0) } to { transform: scaleY(1) } }
@keyframes mr-drift   { 0%,100%{ background-position:0% 50% } 50%{ background-position:100% 50% } }

/* every block in the main area eases in */
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div { animation: mr-fade .45s ease both; }

/* ==================================================================
   4.  HERO BANNER
   ================================================================== */
.mr-hero {
  position: relative; overflow: hidden;
  border-radius: 26px;
  padding: 34px 38px 30px;
  margin: 2px 0 26px;
  background: linear-gradient(118deg,#EAF3FE 0%,#E6F7F1 38%,#FBF1E4 72%,#F3EAFD 100%);
  background-size: 220% 220%;
  animation: mr-rise .7s cubic-bezier(.2,.8,.2,1) both, mr-drift 22s ease-in-out infinite;
  border: 1px solid #DCE9F8;
  box-shadow: 0 16px 40px -26px rgba(23,38,63,.33);
}
.mr-hero::after {
  content:""; position:absolute; inset:0;
  background: linear-gradient(100deg, transparent 38%, rgba(255,255,255,.60) 50%, transparent 62%);
  background-size: 300% 100%;
  animation: mr-sheen 7s linear infinite;
  pointer-events:none;
}
.mr-hero h1 {
  margin:0 0 8px; font-size: 2.15rem; font-weight: 800; color:#16304F;
  text-align:right; line-height:1.4;
}
.mr-hero .mr-sub { font-size:1.02rem; color:#3C5572; text-align:right; margin:0 0 4px; line-height:1.9; }
.mr-hero .mr-eng { direction:ltr; text-align:right; font-size:.86rem; letter-spacing:.09em;
                   text-transform:uppercase; color:#6C86A6; font-weight:600; margin-bottom:10px; }
.mr-hero .mr-by  {
  display:inline-block; margin-top:16px; padding:7px 18px; border-radius:999px;
  background:rgba(255,255,255,.80); border:1px solid #D2E3F6;
  font-weight:700; color:#2A62B4; font-size:.95rem;
  animation: mr-pulse 3.4s ease-in-out infinite;
}

/* ==================================================================
   5.  CONCEPT CARDS
   ================================================================== */
.mr-card {
  border-radius:18px; padding:18px 20px 16px; margin:10px 0 14px;
  background:rgba(255,255,255,.80);
  border:1px solid #E0EAF6;
  border-right:6px solid var(--mr-accent,#3B7DD8);
  box-shadow:0 10px 26px -22px rgba(23,38,63,.45);
  animation: mr-slide-r .55s cubic-bezier(.2,.8,.2,1) both;
  transition: transform .22s ease, box-shadow .22s ease;
}
.mr-card:hover { transform: translateY(-3px); box-shadow:0 16px 34px -20px rgba(23,38,63,.40); }
.mr-card .mr-ct {
  font-weight:800; color:#16304F; font-size:1.05rem; margin:0 0 8px;
  display:flex; align-items:center; gap:9px; flex-direction:row-reverse; justify-content:flex-end;
}
.mr-card .mr-ct .mr-dot {
  width:11px; height:11px; border-radius:50%; background:var(--mr-accent,#3B7DD8);
  flex:0 0 auto;
}
.mr-card .mr-cb { color:#2E4258; font-size:.97rem; line-height:1.95; text-align:right; }
.mr-card .mr-cb b { color:#16304F; }
.mr-card .mr-en  { direction:ltr; text-align:right; display:block; margin-top:9px;
                   font-size:.84rem; color:#6C86A6; font-weight:600; letter-spacing:.02em; }

/* ==================================================================
   6.  BILINGUAL TERM CHIPS
   ================================================================== */
.mr-terms { display:flex; flex-wrap:wrap; gap:9px; margin:8px 0 14px; justify-content:flex-start; }
.mr-term {
  display:inline-flex; flex-direction:column; align-items:center; gap:1px;
  padding:7px 15px; border-radius:14px;
  background:rgba(255,255,255,.86); border:1px solid #DEE9F6;
  animation: mr-pop .5s cubic-bezier(.2,.9,.2,1) both;
  transition: transform .18s ease, border-color .18s ease;
}
.mr-term:hover { transform: translateY(-3px); border-color:#9FC4EE; }
.mr-term .en { direction:ltr; font-size:.82rem; font-weight:700; color:#2A62B4; letter-spacing:.02em; }
.mr-term .ar { font-size:.88rem; color:#35485F; }

/* ==================================================================
   7.  NUMBERED STEP TIMELINE
   ================================================================== */
.mr-steps { position:relative; margin:6px 0 10px; padding:2px 46px 2px 4px; }
.mr-steps::before {
  content:""; position:absolute; top:10px; bottom:10px; right:19px; width:3px;
  background:linear-gradient(#BBD6F5,#CBEDE0,#F7E3C6); border-radius:3px;
  transform-origin: top; animation: mr-grow-y .9s ease both;
}
.mr-step { position:relative; margin:0 0 16px; animation: mr-slide-r .55s ease both; }
.mr-step .num {
  position:absolute; right:-37px; top:-2px;
  width:36px; height:36px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:.95rem; color:#fff;
  background:var(--mr-accent,#3B7DD8);
  box-shadow:0 6px 16px -8px rgba(23,38,63,.55);
  animation: mr-float 4.5s ease-in-out infinite;
}
.mr-step .sh { font-weight:800; color:#16304F; margin:0 0 3px; text-align:right; }
.mr-step .sb { color:#2E4258; font-size:.95rem; line-height:1.9; text-align:right; }

/* ==================================================================
   8.  SPLIT RIBBON  (train / validation / test)
   ================================================================== */
.mr-ribbon {
  display:flex; direction:ltr; height:66px; border-radius:16px; overflow:hidden;
  border:1px solid #D9E6F5; box-shadow:0 10px 26px -22px rgba(23,38,63,.5);
  margin:10px 0 6px;
}
.mr-ribbon > div {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  color:#fff; font-weight:800; font-size:.95rem; gap:1px;
  transform-origin:left; animation: mr-grow-x .8s cubic-bezier(.2,.8,.2,1) both;
  position:relative; overflow:hidden; transition: filter .2s ease;
}
.mr-ribbon > div:hover { filter:brightness(1.07); }
.mr-ribbon > div span { font-size:.78rem; font-weight:600; opacity:.95; letter-spacing:.03em; }
.mr-ribbon > div::after {
  content:""; position:absolute; inset:0;
  background:linear-gradient(100deg,transparent 40%,rgba(255,255,255,.35) 50%,transparent 60%);
  background-size:300% 100%; animation: mr-sheen 4.5s linear infinite;
}

/* ==================================================================
   9.  FORMULA PANEL  (st.container(key="mreq-N"))
   ================================================================== */
[class*="st-key-mreq-"] {
  border-radius:16px; padding:12px 18px 12px; margin:10px 0;
  background:linear-gradient(135deg,#F4F9FF 0%,#F2FBF7 100%);
  border:1px solid #DCEAF8; border-top:3px solid #8FC0EE;
  animation: mr-rise .5s ease both;
  overflow-x:auto;
}
[class*="st-key-mreq-"] .katex-display { margin:.45em 0; }
.mr-eqcap { font-size:.88rem; color:#5B7490; text-align:right; margin:-2px 2px 8px; line-height:1.85; }

/* ==================================================================
   10.  VERDICT / DIAGNOSIS BOXES
   ================================================================== */
.mr-verdict {
  border-radius:16px; padding:15px 18px; margin:10px 0;
  border:1px solid var(--mr-bd,#DCEAF8); background:var(--mr-bg,#F4F9FF);
  animation: mr-pop .45s ease both;
}
.mr-verdict .vt { font-weight:800; color:#16304F; margin:0 0 5px; text-align:right; }
.mr-verdict .vb { color:#2E4258; font-size:.95rem; line-height:1.9; text-align:right; }

/* ==================================================================
   11.  SIDEBAR POLISH
   ================================================================== */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { text-align:right; }
[data-testid="stSidebarNav"] ul { direction:rtl; }
[data-testid="stSidebarNav"] a  { flex-direction:row-reverse; text-align:right; }
[data-testid="stSidebarNav"] a span { text-align:right; }
[data-testid="stSidebarHeader"] { flex-direction: row-reverse; }

/* ==================================================================
   12.  MISC
   ================================================================== */
[data-testid="stMain"] [data-testid="stExpander"] summary { flex-direction: row-reverse; }
[data-testid="stMain"] [data-testid="stExpander"] summary p { text-align:right; width:100%; }
[data-testid="stMain"] [role="tablist"] { direction: rtl; }
[data-testid="stMain"] [data-testid="stAlertContainer"] { flex-direction: row-reverse; }
[data-testid="stMain"] [data-testid="stAlertContainer"] p { text-align:right; }

/* mermaid renders an SVG: keep it centred and LTR */
[data-testid="stMain"] .mermaid, [data-testid="stMain"] svg[id^="mermaid"],
[data-testid="stMermaidChart"] { direction: ltr; }

/* sliders / selectboxes keep their native LTR numeric feel but RTL labels */
[data-testid="stMain"] [data-testid="stSliderTickBarMin"],
[data-testid="stMain"] [data-testid="stSliderTickBarMax"] { direction: ltr; }

/* tabs: a soft pill look */
[data-testid="stMain"] [role="tab"] { font-weight:600; }

.mr-foot {
  margin-top:34px; padding:16px 20px; border-radius:16px;
  background:rgba(255,255,255,.72); border:1px dashed #D3E2F3;
  text-align:center; color:#5B7490; font-size:.9rem; animation: mr-fade 1s ease both;
}
.mr-foot b { color:#2A62B4; }

/* thin animated strip used as a section separator */
.mr-rule {
  height:4px; border-radius:4px; margin:22px 0 14px;
  background:linear-gradient(90deg,#BBD6F5,#CBEDE0,#F7E3C6,#E6DAFA);
  background-size:300% 100%; animation: mr-sheen 9s linear infinite;
}
</style>
"""


def inject_global_css() -> None:
    """Inject the RTL + motion stylesheet once per rerun."""
    st.html(_GLOBAL_CSS)


# ----------------------------------------------------------------------
# Small HTML builders
# ----------------------------------------------------------------------
def hero(title_ar: str, eng_kicker: str, subtitle_ar: str) -> None:
    st.html(
        f"""
        <div class="mr-hero">
          <div class="mr-eng">{eng_kicker}</div>
          <h1>{title_ar}</h1>
          <p class="mr-sub">{subtitle_ar}</p>
          <div class="mr-by">{AUTHOR}</div>
        </div>
        """
    )


def card(title_ar: str, body_html: str, accent: str = "blue", eng: str | None = None) -> None:
    colour = PALETTE.get(accent, accent)
    eng_html = f'<span class="mr-en">{eng}</span>' if eng else ""
    st.html(
        f"""
        <div class="mr-card" style="--mr-accent:{colour}">
          <div class="mr-ct"><span class="mr-dot"></span>{title_ar}</div>
          <div class="mr-cb">{body_html}{eng_html}</div>
        </div>
        """
    )


def terms(pairs: list[tuple[str, str]]) -> None:
    """Bilingual vocabulary chips: [(english, arabic), ...]."""
    chips = "".join(
        f'<div class="mr-term" style="animation-delay:{i * 0.055:.2f}s">'
        f'<span class="en">{en}</span><span class="ar">{ar}</span></div>'
        for i, (en, ar) in enumerate(pairs)
    )
    st.html(f'<div class="mr-terms">{chips}</div>')


def steps(items: list[tuple[str, str]], accent: str = "blue") -> None:
    """Numbered animated timeline: [(heading, body_html), ...]."""
    colour = PALETTE.get(accent, accent)
    body = "".join(
        f'<div class="mr-step" style="animation-delay:{i * 0.09:.2f}s">'
        f'<div class="num" style="--mr-accent:{colour};animation-delay:{i * 0.25:.2f}s">{i + 1}</div>'
        f'<div class="sh">{h}</div><div class="sb">{b}</div></div>'
        for i, (h, b) in enumerate(items)
    )
    st.html(f'<div class="mr-steps">{body}</div>')


def reset_counters() -> None:
    """Called once per rerun from the entry point so container keys stay stable."""
    st.session_state["_mr_eq_n"] = 0


def equation(latex: str, caption_ar: str = "") -> None:
    """A formula inside a soft, animated panel (with an Arabic plain-words caption)."""
    n = st.session_state.get("_mr_eq_n", 0)
    st.session_state["_mr_eq_n"] = n + 1
    with st.container(key=f"mreq-{n}"):
        st.latex(latex)
        if caption_ar:
            st.html(f'<div class="mr-eqcap">{caption_ar}</div>')


def ribbon(parts: list[tuple[str, str, float, str]]) -> None:
    """Animated proportional ribbon: [(label, sublabel, share, hex_colour), ...]."""
    total = sum(p[2] for p in parts) or 1.0
    html = "".join(
        f'<div style="flex:{p[2] / total:.6f};background:{p[3]};'
        f'animation-delay:{i * 0.16:.2f}s">{p[0]}<span>{p[1]}</span></div>'
        for i, p in enumerate(parts)
        if p[2] > 0
    )
    st.html(f'<div class="mr-ribbon">{html}</div>')


def verdict(title_ar: str, body_html: str, tone: str = "blue") -> None:
    tones = {
        "blue": ("#F4F9FF", "#CBE0F7"),
        "green": ("#F1FBF7", "#C4E9DA"),
        "amber": ("#FEF8EF", "#F2DFC0"),
        "rose": ("#FEF2F6", "#F6CEDD"),
        "violet": ("#F8F4FE", "#E0D2F7"),
    }
    bg, bd = tones.get(tone, tones["blue"])
    st.html(
        f"""
        <div class="mr-verdict" style="--mr-bg:{bg};--mr-bd:{bd}">
          <div class="vt">{title_ar}</div><div class="vb">{body_html}</div>
        </div>
        """
    )


def rule() -> None:
    st.html('<div class="mr-rule"></div>')


def footer(note_ar: str = "") -> None:
    extra = f"<br>{note_ar}" if note_ar else ""
    st.html(
        f'<div class="mr-foot">إعداد وتصميم <b>{AUTHOR}</b> — '
        f"سلسلة «تعلّم الآلة من الصفر» · Machine Learning From Scratch{extra}</div>"
    )

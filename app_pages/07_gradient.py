"""الدرس 6 — النزول التدرّجي: مختبر متحرّك."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import card, equation, footer, hero, rule, steps, terms, verdict

hero(
    "النزول التدرّجي — كيف يجد الحاسوب القاع وهو أعمى؟",
    "Lesson 6 · Gradient descent laboratory",
    "تخيّل أنك في واد كثيف الضباب ولا ترى شيئاً، وتريد الوصول إلى أخفض نقطة. "
    "الحل الوحيد: تحسّس ميل الأرض تحت قدميك، واخطُ خطوة في اتجاه الانحدار. ثم كرّر.",
)

st.mermaid_chart(
    """
graph LR
    S["نقطة بداية عشوائية<br/>θ⁰"] --> G["تحسّس الميل<br/>∇J(θ)"]
    G --> M["اخطُ عكس الميل<br/>θ ← θ − η∇J"]
    M --> C{"هل الميل ≈ 0 ؟"}
    C -->|"لا"| G
    C -->|"نعم"| E["وصلنا القاع<br/>Converged"]
    style S fill:#E3F0FE,stroke:#3B7DD8,stroke-width:2px,color:#17263F
    style G fill:#FDF0DC,stroke:#F0A33C,stroke-width:2px,color:#17263F
    style M fill:#EEE6FB,stroke:#9B6BD6,stroke-width:2px,color:#17263F
    style C fill:#FFFFFF,stroke:#8A94A6,stroke-width:2px,color:#17263F
    style E fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
    """
)

equation(
    r"\boldsymbol{\theta}^{(t+1)}=\boldsymbol{\theta}^{(t)}-\eta\,"
    r"\nabla_{\boldsymbol{\theta}}J\big(\boldsymbol{\theta}^{(t)}\big),"
    r"\qquad \nabla_{\boldsymbol{\theta}}J=\Big(\tfrac{\partial J}{\partial\theta_1},"
    r"\dots,\tfrac{\partial J}{\partial\theta_p}\Big)^{\!\top}",
    "رمز ∇ (نابلا) هو مجرد <b>متجه المشتقات الجزئية</b>: مشتقة واحدة لكل معامل. "
    "وهو يشير دائماً إلى اتجاه <b>أسرع صعود</b>، لذلك نضع إشارة سالبة لننزل.",
)

terms(
    [
        ("Gradient", "التدرّج"),
        ("Learning rate η", "معدّل التعلّم"),
        ("Step size", "حجم الخطوة"),
        ("Convergence", "التقارب"),
        ("Divergence", "التباعد"),
        ("Local minimum", "حد أدنى محلّي"),
        ("Saddle point", "نقطة سرج"),
        ("Convex", "محدّب"),
    ]
)

rule()

# ======================================================================
# 1. المختبر المتحرّك في بُعد واحد
# ======================================================================
st.header(":material/animation: المختبر 1 — شاهد النزول يحدث أمامك")

st.markdown(
    """
اضبط **معدّل التعلّم** و**نقطة البداية**، ثم اضغط زر **▶ Play** أسفل الرسم
لتشاهد الكرة تنزل خطوة خطوة. راقب حجم القفزات وشكل المسار.
    """
)

g1, g2, g3 = st.columns(3, gap="medium")
with g1:
    lr = st.slider("معدّل التعلّم η", 0.01, 1.05, 0.22, 0.01, key="gd_lr")
with g2:
    start = st.slider("نقطة البداية θ⁰", -4.5, 4.5, -4.0, 0.25, key="gd_start")
with g3:
    n_steps = st.slider("عدد الخطوات", 5, 60, 28, 1, key="gd_steps")


def J(w):
    return 0.5 * (w - 1.0) ** 2 + 0.6


def dJ(w):
    return (w - 1.0)


path_w, path_j = [start], [J(start)]
w_cur = start
diverged = False
for _ in range(n_steps):
    w_cur = w_cur - lr * dJ(w_cur)
    if not np.isfinite(w_cur) or abs(w_cur) > 1e4:
        diverged = True
        break
    path_w.append(w_cur)
    path_j.append(J(w_cur))

path_w = np.array(path_w)
path_j = np.array(path_j)

span = max(5.0, float(np.abs(path_w).max()) * 1.15)
wg = np.linspace(-span, span, 320)

base = [
    go.Scatter(
        x=wg, y=J(wg), mode="lines", name="J(θ)",
        line=dict(color=viz.C_BLUE, width=4), hoverinfo="skip",
    ),
    go.Scatter(
        x=[1.0], y=[J(1.0)], mode="markers+text", name="القاع",
        marker=dict(size=15, color=viz.C_CYAN, symbol="star"),
        text=["القاع"], textposition="bottom center",
        textfont=dict(size=11, color=viz.C_CYAN), hoverinfo="skip",
    ),
]

frames = []
for k in range(len(path_w)):
    frames.append(
        go.Frame(
            name=str(k),
            data=[
                go.Scatter(
                    x=path_w[: k + 1], y=path_j[: k + 1], mode="lines+markers",
                    line=dict(color=viz.rgba(viz.C_VIOLET, 0.55), width=2.5),
                    marker=dict(size=7, color=viz.rgba(viz.C_VIOLET, 0.55)),
                    name="المسار", hoverinfo="skip",
                ),
                go.Scatter(
                    x=[path_w[k]], y=[path_j[k]], mode="markers",
                    marker=dict(size=20, color=viz.C_ROSE,
                                line=dict(color="#fff", width=3)),
                    name="الموضع الحالي",
                    hovertemplate="θ=%{x:.3f}<br>J=%{y:.3f}<extra></extra>",
                ),
            ],
            traces=[2, 3],
        )
    )

fig = go.Figure(
    data=base
    + [
        go.Scatter(x=[path_w[0]], y=[path_j[0]], mode="lines+markers",
                   line=dict(color=viz.rgba(viz.C_VIOLET, 0.55), width=2.5),
                   marker=dict(size=7, color=viz.rgba(viz.C_VIOLET, 0.55)),
                   name="المسار", hoverinfo="skip"),
        go.Scatter(x=[path_w[0]], y=[path_j[0]], mode="markers",
                   marker=dict(size=20, color=viz.C_ROSE,
                               line=dict(color="#fff", width=3)),
                   name="الموضع الحالي"),
    ],
    frames=frames,
)
viz.style_fig(
    fig, height=470, xtitle="θ — المعامل", ytitle="J(θ) — التكلفة",
    title=f"η = {lr:.2f} · عدد الخطوات = {len(path_w) - 1}",
)
fig.update_layout(
    updatemenus=viz.play_buttons(170, label_play="شغّل", label_pause="إيقاف", y=-0.20),
    sliders=viz.frame_slider([str(i) for i in range(len(path_w))], prefix="الخطوة "),
    margin=dict(l=58, r=22, t=56, b=96),
)
st.plotly_chart(fig, width="stretch", key="gd_anim")

k1, k2, k3, k4 = st.columns(4)
k1.metric("θ النهائية", "∞" if diverged else f"{path_w[-1]:.4f}")
k2.metric("J النهائية", "∞" if diverged else f"{path_j[-1]:.4f}")
k3.metric("المسافة عن القاع", "∞" if diverged else f"{abs(path_w[-1] - 1.0):.4f}")
k4.metric("η · ميل البداية", f"{lr * abs(dJ(start)):.3f}")

if diverged or lr >= 2.0:
    st.error(
        "**تباعد (Divergence)!** معدّل التعلّم كبير جداً فقفزت الكرة إلى الجانب الآخر "
        "أعلى مما كانت، ثم أعلى، حتى انفجرت الأرقام. في الواقع ستقرأ `NaN` في منحنى الخسارة."
    )
elif lr > 1.0:
    st.warning(
        "**تذبذب (Oscillation).** الخطوة تتجاوز القاع في كل مرة فيتأرجح المسار يميناً "
        "ويساراً. قد يصل في النهاية لكنه يضيّع وقتاً. خفّض η."
    )
elif lr < 0.06:
    st.info(
        "**بطء شديد.** الخطوات صغيرة جداً ولم تصل بعد. هذا آمن لكنه مكلف: "
        "قد تحتاج آلاف الحقب. ارفع η."
    )
else:
    st.success(
        "**معدّل تعلّم جيد.** نزول سريع في البداية ثم خطوات أصغر تلقائياً قرب القاع "
        "(لأن الميل نفسه يصغر هناك). هذا هو السلوك المثالي."
    )

verdict(
    "لماذا تصغر الخطوات تلقائياً قرب القاع؟",
    "لأن حجم الخطوة = η × الميل. وكلما اقتربنا من القاع صغر الميل، فصغرت الخطوة. "
    "<b>النزول التدرّجي يبطّئ من نفسه</b> عند الاقتراب — خاصية جميلة ومجانية.",
    tone="green",
)

rule()

# ======================================================================
# 2. مقارنة ثلاثة معدّلات
# ======================================================================
st.header(":material/speed: ثلاثة معدّلات تعلّم، ثلاثة مصائر")

regimes = [
    ("صغير جداً  η = 0.03", 0.03, viz.C_BLUE, "آمن لكنه بطيء جداً"),
    ("مناسب  η = 0.30", 0.30, viz.C_GREEN, "ينزل بسرعة ويستقر"),
    ("كبير جداً  η = 1.95", 1.95, viz.C_ROSE, "يتذبذب ويتباعد"),
]

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=np.linspace(-6, 8, 300), y=J(np.linspace(-6, 8, 300)),
               mode="lines", name="J(θ)",
               line=dict(color=viz.rgba(viz.C_SLATE, 0.45), width=3), hoverinfo="skip")
)
for label, l, col, _ in regimes:
    w = -4.0
    ws, js = [w], [J(w)]
    for _ in range(22):
        w = w - l * dJ(w)
        if abs(w) > 60:
            break
        ws.append(w)
        js.append(J(w))
    fig2.add_trace(
        go.Scatter(x=ws, y=js, mode="lines+markers", name=label,
                   line=dict(color=col, width=3),
                   marker=dict(size=8, color=col, line=dict(color="#fff", width=1)),
                   hovertemplate="θ=%{x:.2f}<extra></extra>")
    )
fig2.update_yaxes(range=[0, 26])
fig2.update_xaxes(range=[-6.5, 8.5])
viz.style_fig(fig2, height=430, xtitle="θ", ytitle="J(θ)",
              title="المسار نفسه، البداية نفسها، والفرق الوحيد هو η")
st.plotly_chart(fig2, width="stretch", key="gd_three")

t1, t2, t3 = st.columns(3, gap="medium")
with t1:
    card("η صغير جداً",
         "خطوات متناهية الصغر. النموذج يتعلّم لكن ببطء مؤلم، وقد تتوقّف قبل "
         "الوصول فتظنّ أن النموذج ضعيف بينما المشكلة في η فقط.",
         accent="blue", eng="Too small · slow convergence")
with t2:
    card("η مناسب",
         "نزول حادّ في البداية ثم تباطؤ طبيعي قرب القاع. "
         "منحنى الخسارة يشبه حرف L.",
         accent="green", eng="Just right · healthy L-shaped curve")
with t3:
    card("η كبير جداً",
         "يقفز فوق القاع في كل مرة. منحنى الخسارة يتذبذب أو يرتفع، "
         "وقد تظهر قيم NaN. <b>هذا أكثر سبب لفشل التدريب عند المبتدئين</b>.",
         accent="rose", eng="Too large · oscillation / divergence")

st.markdown(
    """
:material/tips_and_updates: **وصفة عملية لضبط η:** ابدأ بـ `0.01`. إذا كان المنحنى يتذبذب
اقسم على 3؛ وإذا كان ينزل ببطء شديد اضرب في 3. جرّب على مقياس لوغاريتمي:
`0.0001, 0.001, 0.01, 0.1, 1`. وفي الممارسة الحديثة نستعمل **جدولة معدّل التعلّم
(Learning rate schedule)**: نبدأ كبيراً ثم نصغّره تدريجياً.
    """
)

rule()

# ======================================================================
# 3. بُعدان: الكفاف والتقييس
# ======================================================================
st.header(":material/blur_on: المختبر 2 — بُعدان، وأثر تقييس الخصائص")

st.markdown(
    """
في الواقع لدينا معاملات كثيرة لا واحد. في بُعدين نرسم **خطوط الكفاف (Contours)**:
كل حلقة تمثّل قيمة تكلفة ثابتة، تماماً مثل خرائط الارتفاع الجغرافية.

**هنا نكتشف شيئاً عملياً بالغ الأهمية:** إذا كانت خصائصك بمقاييس مختلفة
(مساحة بالأمتار من 50 إلى 300، وعدد الغرف من 1 إلى 5)، تصبح خطوط الكفاف
**بيضاوية ممدودة جداً**، فيتعرّج مسار النزول ويبطؤ بشدة. وتقييس الخصائص
يحوّلها إلى **دوائر** فيصبح المسار مستقيماً وسريعاً.
    """
)

sc1, sc2 = st.columns([1, 1], gap="medium")
with sc1:
    scaled = st.toggle("طبّق تقييس الخصائص (Feature scaling)", value=False, key="gd_scale")
with sc2:
    lr2 = st.slider("معدّل التعلّم η", 0.01, 0.45, 0.10, 0.01, key="gd_lr2")

a, b_ = (1.0, 1.0) if scaled else (1.0, 12.0)


def J2(u, v):
    return a * u**2 + b_ * v**2


def dJ2(u, v):
    return 2 * a * u, 2 * b_ * v


u, v = -3.2, 2.4
pu, pv = [u], [v]
for _ in range(40):
    gu, gv = dJ2(u, v)
    u, v = u - lr2 * gu, v - lr2 * gv
    if not np.isfinite(u) or abs(u) > 50 or abs(v) > 50:
        break
    pu.append(u)
    pv.append(v)

ug = np.linspace(-4, 4, 160)
vg = np.linspace(-3, 3, 160)
UU, VV = np.meshgrid(ug, vg)
ZZ = J2(UU, VV)

fig3 = go.Figure()
fig3.add_trace(
    go.Contour(
        x=ug, y=vg, z=ZZ, colorscale=viz.SEQ_LIGHT, showscale=False,
        contours=dict(coloring="heatmap", showlines=True,
                      start=0, end=float(ZZ.max()), size=float(ZZ.max()) / 22),
        line=dict(color="rgba(255,255,255,.55)", width=1),
        hovertemplate="θ₁=%{x:.2f}<br>θ₂=%{y:.2f}<br>J=%{z:.2f}<extra></extra>",
        name="J(θ₁,θ₂)",
    )
)
f3_frames = [
    go.Frame(
        name=str(k),
        data=[
            go.Scatter(x=pu[: k + 1], y=pv[: k + 1], mode="lines+markers",
                       line=dict(color=viz.C_ROSE, width=3),
                       marker=dict(size=7, color="#fff",
                                   line=dict(color=viz.C_ROSE, width=2)),
                       name="مسار النزول", hoverinfo="skip"),
        ],
        traces=[1],
    )
    for k in range(len(pu))
]
fig3.add_trace(
    go.Scatter(x=[pu[0]], y=[pv[0]], mode="lines+markers",
               line=dict(color=viz.C_ROSE, width=3),
               marker=dict(size=7, color="#fff", line=dict(color=viz.C_ROSE, width=2)),
               name="مسار النزول")
)
fig3.add_trace(
    go.Scatter(x=[0], y=[0], mode="markers+text",
               marker=dict(size=16, color=viz.C_GREEN, symbol="star"),
               text=["القاع"], textposition="top center",
               textfont=dict(size=11, color=viz.C_GREEN),
               name="الحد الأدنى", hoverinfo="skip")
)
fig3.frames = f3_frames
viz.style_fig(
    fig3, height=470, xtitle="θ₁", ytitle="θ₂",
    title=("خصائص مُقيَّسة ← كفاف دائري ← مسار مستقيم"
           if scaled else "خصائص غير مُقيَّسة ← كفاف ممدود ← مسار متعرّج"),
)
fig3.update_layout(
    updatemenus=viz.play_buttons(120, label_play="شغّل", label_pause="إيقاف", y=-0.20),
    sliders=viz.frame_slider([str(i) for i in range(len(pu))], prefix="الخطوة "),
    margin=dict(l=58, r=22, t=56, b=96),
)
st.plotly_chart(fig3, width="stretch", key="gd_contour")

st.metric("عدد الخطوات حتى الاستقرار", f"{len(pu) - 1}")

if not scaled:
    st.warning(
        "لاحظ التعرّج (Zig-zag): التدرّج يشير إلى جدار الوادي الضيّق لا إلى القاع. "
        "فعّل **التقييس** في المفتاح أعلاه وقارن عدد الخطوات."
    )
else:
    st.success(
        "بعد التقييس أصبح الكفاف شبه دائري، والتدرّج يشير مباشرة إلى القاع، "
        "فقلّ عدد الخطوات بشكل كبير. **هذا سبب أن StandardScaler إلزامي عملياً**."
    )

equation(
    r"z = \frac{x-\mu}{\sigma}\quad\text{(Standardisation)}\qquad\text{أو}\qquad"
    r"x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}\quad\text{(Min-Max)}",
    "التقييس القياسي يجعل متوسط كل خصيصة صفراً وانحرافها المعياري واحداً. "
    "<b>تحذير حاسم:</b> احسب μ و σ على <b>بيانات التدريب فقط</b>، "
    "ثم طبّقهما على بيانات الاختبار — وإلا وقعت في تسرّب البيانات.",
)

rule()

# ======================================================================
# 4. السطح ثلاثي الأبعاد
# ======================================================================
st.header(":material/landscape: سطح التكلفة في ثلاثة أبعاد")

st.markdown(
    "نفس الدالة السابقة، لكن مرسومة كتضاريس. "
    "**دوّر الشكل بالفأرة** لترى الوادي من زوايا مختلفة."
)

fig4 = go.Figure(
    data=[
        go.Surface(
            x=ug, y=vg, z=ZZ, colorscale=viz.SEQ_LIGHT, opacity=0.93, showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate="θ₁=%{x:.2f}<br>θ₂=%{y:.2f}<br>J=%{z:.2f}<extra></extra>",
        ),
        go.Scatter3d(
            x=pu, y=pv, z=[J2(uu, vv) + 0.4 for uu, vv in zip(pu, pv)],
            mode="lines+markers", name="مسار النزول",
            line=dict(color=viz.C_ROSE, width=6),
            marker=dict(size=4, color=viz.C_ROSE),
        ),
    ]
)
fig4.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Cairo, sans-serif", color=viz.INK, size=12),
    scene=dict(
        xaxis=dict(title="θ₁", backgroundcolor="rgba(250,252,255,1)",
                   gridcolor="#DCE7F4", color="#5B7490"),
        yaxis=dict(title="θ₂", backgroundcolor="rgba(250,252,255,1)",
                   gridcolor="#DCE7F4", color="#5B7490"),
        zaxis=dict(title="J", backgroundcolor="rgba(250,252,255,1)",
                   gridcolor="#DCE7F4", color="#5B7490"),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    ),
    showlegend=False,
)
st.plotly_chart(fig4, width="stretch", key="gd_3d")

rule()

# ======================================================================
# 5. دوال غير محدّبة
# ======================================================================
st.header(":material/terrain: عندما لا يكون الوادي وادياً واحداً")

st.markdown(
    """
الانحدار الخطي محظوظ: دالة تكلفته **محدّبة (Convex)** — أي أنها مثل وعاء واحد،
وأي نزول سيصل حتماً إلى القاع الوحيد. أما الشبكات العصبية فدوالها **غير محدّبة**:
تضاريس فيها قمم وأودية كثيرة. ثلاث عقبات تظهر هنا:
    """
)

xx = np.linspace(-3.4, 3.4, 500)
yy = 0.6 * xx**4 - 2.2 * xx**2 + 0.5 * xx + 3.0

fig5 = go.Figure()
fig5.add_trace(
    go.Scatter(x=xx, y=yy, mode="lines", name="دالة غير محدّبة",
               line=dict(color=viz.C_BLUE, width=4), hoverinfo="skip")
)
fig5.add_trace(
    go.Scatter(x=[-1.39], y=[0.6 * 1.39**4 - 2.2 * 1.39**2 - 0.5 * 1.39 + 3.0],
               mode="markers+text", name="حد أدنى محلّي",
               marker=dict(size=16, color=viz.C_AMBER),
               text=["حد أدنى محلّي"], textposition="top center",
               textfont=dict(size=11, color=viz.C_AMBER), hoverinfo="skip")
)
fig5.add_trace(
    go.Scatter(x=[1.33], y=[0.6 * 1.33**4 - 2.2 * 1.33**2 + 0.5 * 1.33 + 3.0],
               mode="markers+text", name="حد أدنى شامل",
               marker=dict(size=18, color=viz.C_GREEN, symbol="star"),
               text=["حد أدنى شامل"], textposition="bottom center",
               textfont=dict(size=11, color=viz.C_GREEN), hoverinfo="skip")
)
fig5.add_trace(
    go.Scatter(x=[0.06], y=[3.0], mode="markers+text", name="هضبة / سرج",
               marker=dict(size=14, color=viz.C_VIOLET, symbol="square"),
               text=["هضبة (الميل ≈ 0)"], textposition="top center",
               textfont=dict(size=11, color=viz.C_VIOLET), hoverinfo="skip")
)
viz.style_fig(fig5, height=400, xtitle="θ", ytitle="J(θ)",
              title="تضاريس حقيقية: أكثر من قاع واحد")
st.plotly_chart(fig5, width="stretch", key="gd_nonconvex")

n1, n2, n3 = st.columns(3, gap="medium")
with n1:
    card("الحد الأدنى المحلّي",
         "قاع صغير ليس الأعمق. قد يعلق فيه النزول ويتوقّف. "
         "العلاج: تهيئة عشوائية متعدّدة، أو ضجيج SGD، أو الزخم.",
         accent="amber", eng="Local minimum")
with n2:
    card("نقطة السرج",
         "منطقة الميل فيها يكاد يكون صفراً في كل الاتجاهات دون أن تكون قاعاً. "
         "في الأبعاد العالية هذه أخطر من الحدود الدنيا المحلّية.",
         accent="violet", eng="Saddle point / plateau")
with n3:
    card("الوادي الضيّق",
         "منحدر شديد في اتجاه ومستوٍ في آخر. يسبّب تعرّجاً مرهقاً. "
         "العلاج: التقييس، والزخم، و Adam.",
         accent="cyan", eng="Ravine / ill-conditioning")

rule()

# ======================================================================
# 6. أنواع النزول والمحسّنات
# ======================================================================
st.header(":material/fork_right: أنواع النزول والمحسّنات الحديثة")

steps(
    [
        ("Batch Gradient Descent",
         "يحسب التدرّج على <b>كل</b> البيانات قبل كل تحديث. تدرّج دقيق ومسار ناعم، "
         "لكنه بطيء ويحتاج ذاكرة ضخمة."),
        ("Stochastic Gradient Descent (SGD)",
         "تحديث بعد <b>كل مثال</b>. سريع جداً، ومساره المهتزّ يساعده على الهروب من "
         "الحدود الدنيا المحلّية، لكنه لا يستقرّ تماماً."),
        ("Mini-batch GD",
         "دفعات من 32 أو 64 أو 128 مثالاً. <b>المعيار الصناعي</b>: توازن بين "
         "استقرار التدرّج وسرعة التنفيذ على الـ GPU."),
        ("Momentum",
         "يضيف «قصوراً ذاتياً»: v ← βv + ∇J ثم θ ← θ − ηv. "
         "يسرّع في الاتجاهات الثابتة ويخفّف التعرّج، مثل كرة ثقيلة تتدحرج."),
        ("RMSProp / AdaGrad",
         "<b>معدّل تعلّم مختلف لكل معامل</b>، يُصغَّر تلقائياً للمعاملات التي تتحرّك كثيراً. "
         "يحلّ مشكلة الخصائص متفاوتة المقياس."),
        ("Adam",
         "يجمع الزخم + RMSProp. <b>الخيار الافتراضي اليوم</b> في التعلّم العميق؛ "
         "يعمل جيداً بإعدادات شبه ثابتة (η=0.001, β₁=0.9, β₂=0.999)."),
    ],
    accent="cyan",
)

st.markdown(
    """
| المحسّن | الفكرة الأساسية | متى تستعمله |
|---|---|---|
| **SGD** | خطوة ثابتة عكس التدرّج | نماذج بسيطة، أو مع ضبط دقيق + جدولة |
| **SGD + Momentum** | تراكم الاتجاه السابق | رؤية حاسوبية، شبكات التفاف |
| **AdaGrad** | معدّل يتناقص حسب التاريخ | بيانات متفرّقة (نصوص) |
| **RMSProp** | متوسط متحرّك لمربع التدرّج | شبكات متكرّرة (RNN) |
| **Adam** | زخم + تكيّف + تصحيح انحياز | **الافتراضي للبدء دائماً** |
    """
)

verdict(
    "نصيحة عملية مختصرة",
    "ابدأ بـ <b>Adam</b> و <b>η = 0.001</b> و <b>batch = 32</b>. "
    "راقب منحنى الخسارة. إن تذبذب اقسم η على 3؛ إن كان بطيئاً اضرب في 3. "
    "وقبل كل ذلك: <b>قيّس خصائصك</b> — هذه وحدها تحلّ نصف مشاكل التدريب.",
    tone="blue",
)

footer("الدرس التالي: نُشغّل تدريباً حقيقياً أمام عينيك، حقبة بعد حقبة.")

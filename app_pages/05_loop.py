"""الدرس 4 — حلقة التدريب الحيّة: شاهد النموذج يتعلّم فعلاً."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import equation, footer, hero, rule, terms, verdict

hero(
    "حلقة التدريب الحيّة — التعلّم يحدث أمام عينيك",
    "Lesson 4 · A real training loop, epoch by epoch",
    "لا رسوم توضيحية هذه المرة: سنُشغّل نزولاً تدرّجياً حقيقياً بالـ NumPy، "
    "ونرسم بعد كل حقبة الخط المُتعلَّم ومنحنى الخسارة ومسار المعاملات.",
)

terms(
    [
        ("Epoch", "حقبة"),
        ("Batch size", "حجم الدُفعة"),
        ("Learning rate", "معدّل التعلّم"),
        ("Initialization", "التهيئة"),
        ("Train loss", "خسارة التدريب"),
        ("Validation loss", "خسارة التحقق"),
        ("Early stopping", "التوقّف المبكّر"),
    ]
)

# ======================================================================
# الإعدادات
# ======================================================================
st.header(":material/settings: اضبط المعاملات الفائقة ثم شغّل")

with st.form("train_form", border=True):
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        lr = st.select_slider(
            "معدّل التعلّم η",
            options=[0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35],
            value=0.02,
        )
        n_points = st.slider("حجم البيانات n", 30, 400, 120, 10)
    with f2:
        epochs = st.slider("عدد الحقب (Epochs)", 5, 120, 40, 1)
        batch = st.select_slider(
            "حجم الدُفعة (Batch size)", options=[1, 4, 8, 16, 32, 64, 128], value=16
        )
    with f3:
        noise = st.slider("مستوى الضجيج في البيانات σ", 0.2, 6.0, 2.0, 0.2)
        speed = st.select_slider(
            "سرعة العرض", options=["بطيء", "متوسّط", "سريع", "فوري"], value="متوسّط"
        )

    init_mode = st.segmented_control(
        "التهيئة (Initialization)",
        ["أصفار", "عشوائي صغير", "بعيد جداً"],
        default="أصفار",
        selection_mode="single",
    )
    submitted = st.form_submit_button(
        ":material/play_arrow:  شغّل التدريب", type="primary", width="stretch"
    )

DELAY = {"بطيء": 0.22, "متوسّط": 0.09, "سريع": 0.03, "فوري": 0.0}[speed or "متوسّط"]

# ======================================================================
# البيانات الحقيقية (العلاقة التي نريد اكتشافها)
# ======================================================================
TRUE_W, TRUE_B = 2.6, 4.0
rng = np.random.default_rng(2024)
x_raw = rng.uniform(0.0, 10.0, n_points)
y_raw = TRUE_B + TRUE_W * x_raw + rng.normal(0.0, noise, n_points)

# split 80/20 just so we can also watch a validation curve
perm = rng.permutation(n_points)
cut = int(0.8 * n_points)
tr_idx, va_idx = perm[:cut], perm[cut:]
x_tr, y_tr = x_raw[tr_idx], y_raw[tr_idx]
x_va, y_va = x_raw[va_idx], y_raw[va_idx]

st.caption(
    f"العلاقة الحقيقية المخفيّة التي أنشأنا بها البيانات هي "
    f"**y = {TRUE_W} · x + {TRUE_B} + ضجيج**. "
    f"النموذج لا يعرفها، وعليه أن يكتشفها. "
    f"({len(x_tr)} مثال تدريب · {len(x_va)} مثال تحقق)"
)

equation(
    r"\hat{y}=wx+b,\qquad J(w,b)=\frac{1}{m}\sum_{i\in\text{batch}}\big(wx_i+b-y_i\big)^2",
    "النموذج والتكلفة.",
)
equation(
    r"\frac{\partial J}{\partial w}=\frac{2}{m}\sum_i x_i\big(wx_i+b-y_i\big),"
    r"\qquad\frac{\partial J}{\partial b}=\frac{2}{m}\sum_i \big(wx_i+b-y_i\big)",
    "المشتقتان الجزئيتان المستعملتان حرفياً في الكود أدناه — "
    "اشتقّهما بنفسك على ورقة مرة واحدة وستفهم التدريب إلى الأبد.",
)

rule()

# ======================================================================
# منطقة العرض الحيّ
# ======================================================================
st.header(":material/monitor_heart: لوحة التدريب الحيّة")

status_slot = st.empty()
metric_slot = st.container()
col_fit, col_loss = st.columns(2, gap="medium")
fit_slot = col_fit.empty()
loss_slot = col_loss.empty()
path_slot = st.empty()


def cost(w: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((w * x + b - y) ** 2))


def fit_figure(w: float, b: float, epoch: int) -> go.Figure:
    g = np.linspace(-0.4, 10.4, 40)
    f = go.Figure()
    f.add_trace(
        go.Scatter(x=x_tr, y=y_tr, mode="markers", name="تدريب",
                   marker=dict(size=8, color=viz.rgba(viz.C_BLUE, 0.75),
                               line=dict(color="#fff", width=1)),
                   hoverinfo="skip")
    )
    f.add_trace(
        go.Scatter(x=x_va, y=y_va, mode="markers", name="تحقق",
                   marker=dict(size=9, color=viz.rgba(viz.C_AMBER, 0.9),
                               symbol="diamond", line=dict(color="#fff", width=1)),
                   hoverinfo="skip")
    )
    f.add_trace(
        go.Scatter(x=g, y=TRUE_W * g + TRUE_B, mode="lines", name="الحقيقة المخفيّة",
                   line=dict(color=viz.rgba(viz.C_SLATE, 0.6), width=3, dash="dot"),
                   hoverinfo="skip")
    )
    f.add_trace(
        go.Scatter(x=g, y=w * g + b, mode="lines", name="نموذجك الآن",
                   line=dict(color=viz.C_ROSE, width=5),
                   hovertemplate="ŷ=%{y:.2f}<extra></extra>")
    )
    viz.style_fig(f, height=380, xtitle="x", ytitle="y",
                  title=f"الحقبة {epoch} · ŷ = {w:.3f}·x + {b:.3f}")
    f.update_yaxes(range=[float(y_raw.min()) - 3, float(y_raw.max()) + 3])
    return f


def loss_figure(tr_hist: list[float], va_hist: list[float]) -> go.Figure:
    ep = np.arange(1, len(tr_hist) + 1)
    f = go.Figure()
    f.add_trace(
        go.Scatter(x=ep, y=tr_hist, mode="lines+markers", name="خسارة التدريب",
                   line=dict(color=viz.C_BLUE, width=3.5),
                   marker=dict(size=5), hovertemplate="epoch %{x}<br>%{y:.4f}<extra></extra>")
    )
    f.add_trace(
        go.Scatter(x=ep, y=va_hist, mode="lines+markers", name="خسارة التحقق",
                   line=dict(color=viz.C_AMBER, width=3.5, dash="dash"),
                   marker=dict(size=5), hovertemplate="epoch %{x}<br>%{y:.4f}<extra></extra>")
    )
    f.add_hline(y=noise**2, line=dict(color=viz.rgba(viz.C_GREEN, 0.75), width=2, dash="dot"),
                annotation_text="الحد النظري (σ²) — خطأ لا يمكن إزالته",
                annotation_position="top left",
                annotation_font=dict(size=11, color=viz.C_GREEN))
    viz.style_fig(f, height=380, xtitle="الحقبة", ytitle="MSE",
                  title="منحنى الخسارة — لوحة القيادة")
    f.update_yaxes(type="log")
    return f


def path_figure(ws: list[float], bs: list[float]) -> go.Figure:
    wg = np.linspace(min(0.0, min(ws)) - 0.6, max(4.0, max(ws)) + 0.6, 90)
    bg = np.linspace(min(-2.0, min(bs)) - 1.5, max(8.0, max(bs)) + 1.5, 90)
    WW, BB = np.meshgrid(wg, bg)
    ZZ = np.array(
        [[cost(w_, b_, x_tr, y_tr) for w_ in wg] for b_ in bg]
    )
    f = go.Figure()
    f.add_trace(
        go.Contour(x=wg, y=bg, z=np.log10(ZZ + 1e-9), colorscale=viz.SEQ_LIGHT,
                   showscale=False,
                   contours=dict(coloring="heatmap", showlines=True),
                   line=dict(color="rgba(255,255,255,.5)", width=1),
                   hovertemplate="w=%{x:.2f}<br>b=%{y:.2f}<br>log₁₀J=%{z:.2f}<extra></extra>")
    )
    f.add_trace(
        go.Scatter(x=ws, y=bs, mode="lines+markers", name="مسار المعاملات",
                   line=dict(color=viz.C_ROSE, width=3),
                   marker=dict(size=5, color="#fff",
                               line=dict(color=viz.C_ROSE, width=1.6)),
                   hoverinfo="skip")
    )
    f.add_trace(
        go.Scatter(x=[TRUE_W], y=[TRUE_B], mode="markers+text",
                   marker=dict(size=16, color=viz.C_GREEN, symbol="star"),
                   text=["القيم الحقيقية"], textposition="top center",
                   textfont=dict(size=11, color=viz.C_GREEN),
                   name="الحقيقة", hoverinfo="skip")
    )
    viz.style_fig(f, height=430, xtitle="w — الميل", ytitle="b — التقاطع",
                  title="رحلة المعاملات فوق سطح التكلفة")
    return f


# ======================================================================
# التدريب
# ======================================================================
if submitted:
    if init_mode == "أصفار":
        w, b = 0.0, 0.0
    elif init_mode == "عشوائي صغير":
        w, b = float(rng.normal(0, 0.3)), float(rng.normal(0, 0.3))
    else:
        w, b = -6.0, 18.0

    tr_hist: list[float] = []
    va_hist: list[float] = []
    ws: list[float] = [w]
    bs: list[float] = [b]
    rows: list[dict] = []

    prog = status_slot.progress(0.0, text="جارٍ التدريب...")
    # one st.empty() per column so each metric is *replaced*, not appended
    m_slots = [c.empty() for c in metric_slot.columns(5)]

    local_rng = np.random.default_rng(7)
    n_tr = len(x_tr)

    for ep in range(1, epochs + 1):
        order = local_rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            bi = order[s : s + batch]
            xb, yb = x_tr[bi], y_tr[bi]
            m = len(bi)

            err = (w * xb + b) - yb              # forward + residual
            gw = (2.0 / m) * float(np.sum(xb * err))   # ∂J/∂w
            gb = (2.0 / m) * float(np.sum(err))        # ∂J/∂b

            w -= lr * gw                          # update
            b -= lr * gb

            if not np.isfinite(w) or not np.isfinite(b):
                break

        if not np.isfinite(w) or not np.isfinite(b):
            prog.empty()
            st.error(
                "**انفجر التدريب (Divergence).** قيم المعاملات أصبحت `inf`/`NaN`. "
                f"معدّل التعلّم η = {lr} كبير جداً لهذه البيانات. جرّب قيمة أصغر بعشر مرات."
            )
            st.stop()

        j_tr = cost(w, b, x_tr, y_tr)
        j_va = cost(w, b, x_va, y_va)
        tr_hist.append(j_tr)
        va_hist.append(j_va)
        ws.append(w)
        bs.append(b)
        rows.append({"الحقبة": ep, "w": round(w, 4), "b": round(b, 4),
                     "خسارة التدريب": round(j_tr, 4), "خسارة التحقق": round(j_va, 4)})

        prog.progress(ep / epochs, text=f"الحقبة {ep} من {epochs} · MSE = {j_tr:,.4f}")
        m_slots[0].metric("الحقبة", f"{ep}/{epochs}")
        m_slots[1].metric("w", f"{w:.3f}", delta=f"{w - TRUE_W:+.3f} عن الحقيقي",
                          delta_color="off")
        m_slots[2].metric("b", f"{b:.3f}", delta=f"{b - TRUE_B:+.3f} عن الحقيقي",
                          delta_color="off")
        m_slots[3].metric("خسارة التدريب", f"{j_tr:,.4f}")
        m_slots[4].metric("خسارة التحقق", f"{j_va:,.4f}")

        fit_slot.plotly_chart(fit_figure(w, b, ep), width="stretch",
                              key=f"fit_{ep}")
        loss_slot.plotly_chart(loss_figure(tr_hist, va_hist), width="stretch",
                               key=f"loss_{ep}")

        if DELAY:
            time.sleep(DELAY)

    prog.empty()
    path_slot.plotly_chart(path_figure(ws, bs), width="stretch", key="path_final")

    st.session_state["loop_rows"] = rows
    st.session_state["loop_final"] = (w, b, tr_hist[-1], va_hist[-1])

    rule()
    st.header(":material/flag: نتيجة التدريب")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("w المُتعلَّم", f"{w:.4f}", delta=f"الحقيقي {TRUE_W}", delta_color="off")
    r2.metric("b المُتعلَّم", f"{b:.4f}", delta=f"الحقيقي {TRUE_B}", delta_color="off")
    r3.metric("MSE التدريب", f"{tr_hist[-1]:,.4f}")
    r4.metric("MSE التحقق", f"{va_hist[-1]:,.4f}")

    gap = va_hist[-1] - tr_hist[-1]
    if tr_hist[-1] > 3 * noise**2:
        st.warning(
            "خسارة التدريب لا تزال أعلى بكثير من الحد النظري σ². "
            "إمّا أن عدد الحقب غير كافٍ، أو معدّل التعلّم صغير جداً. زِد أحدهما."
        )
    elif abs(gap) < 0.35 * max(tr_hist[-1], 1e-9) + 0.5:
        st.success(
            "خسارة التدريب وخسارة التحقق متقاربتان وقريبتان من الحد النظري σ² — "
            "**تدريب سليم**. النموذج تعلّم الإشارة ولم يحفظ الضجيج."
        )
    else:
        st.info(
            f"الفجوة بين التحقق والتدريب = {gap:+.4f}. "
            "فجوة كبيرة موجبة تشير عادة إلى إفراط في التعلّم "
            "(نراه بوضوح في الدرس 7 مع نماذج أكثر مرونة)."
        )

    with st.expander(":material/table_rows: سجلّ التدريب الكامل (كل حقبة)"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)

    verdict(
        "ماذا رأيت بالضبط؟",
        "رأيت ثلاثة أشياء في وقت واحد: <b>(1)</b> الخط الأحمر يدور ويزحف حتى يطابق "
        "الخط الرمادي المخفي، <b>(2)</b> منحنى الخسارة ينزل ثم يستقر عند حدّ لا ينزل "
        "تحته (وهو ضجيج البيانات σ² — خطأ لا يمكن لأي نموذج إزالته)، "
        "<b>(3)</b> نقطة المعاملات (w,b) تتدحرج على خريطة الكفاف حتى تستقرّ عند النجمة. "
        "هذه الثلاثة هي التدريب.",
        tone="violet",
    )

else:
    status_slot.info(
        ":material/touch_app: اضبط الإعدادات أعلاه ثم اضغط **شغّل التدريب** "
        "لمشاهدة الحلقة تعمل حقبة بعد حقبة."
    )
    fit_slot.plotly_chart(fit_figure(0.0, 0.0, 0), width="stretch", key="fit_idle")
    loss_slot.plotly_chart(loss_figure([noise**2 * 6], [noise**2 * 6]),
                           width="stretch", key="loss_idle")

rule()

# ======================================================================
# تجارب مقترحة
# ======================================================================
st.header(":material/science: تجارب مقترحة — جرّبها بنفسك الآن")

st.markdown(
    """
هذه التجارب تساوي عشر صفحات نظرية. نفّذها واحدة واحدة وراقب الفرق:

1. **η = 0.35 مع دفعة = 1** ← سترى التدريب ينفجر أو يتذبذب بعنف.
   *الدرس: معدّل التعلّم الكبير + دفعة صغيرة = ضجيج شديد في التدرّج.*
2. **η = 0.0005 مع 40 حقبة** ← الخط بالكاد يتحرّك عن الصفر.
   *الدرس: η الصغير ليس خطأ منطقياً، لكنه خطأ عملي مكلف.*
3. **التهيئة «بعيد جداً»** ← شاهد المسار الطويل على خريطة الكفاف.
   *الدرس: التهيئة تحدّد نقطة الانطلاق فقط، لا الوجهة (في الدوال المحدّبة).*
4. **ضجيج σ = 6** ← لاحظ أن الخسارة تستقرّ عند قيمة عالية مهما طال التدريب.
   *الدرس: لا يوجد نموذج يستطيع التنبؤ بالعشوائية. هذا هو **الخطأ غير القابل للإزالة**.*
5. **دفعة = 128 مقابل دفعة = 1** بنفس η ← قارن نعومة منحنى الخسارة.
   *الدرس: الدُفعة الكبيرة = تدرّج أدقّ = مسار أنعم، لكن تحديثات أقل لكل حقبة.*
    """
)

verdict(
    "الحد النظري الذي لا يمكن تجاوزه",
    "الخط الأخضر المنقّط في منحنى الخسارة هو <b>σ²</b>، أي تباين الضجيج الذي أضفناه "
    "للبيانات عمداً. لا يمكن لأي نموذج في الوجود أن ينزل تحته على بيانات جديدة. "
    "إن رأيت خسارة <b>تدريب</b> أقل منه بكثير، فاعلم أن النموذج بدأ "
    "<b>يحفظ الضجيج</b> — وهذا هو الإفراط في التعلّم بعينه.",
    tone="rose",
)

footer("انتهى الجزء الأول. ننتقل الآن إلى الجزء الثاني: كيف نقسّم البيانات؟")

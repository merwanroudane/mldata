"""الدرس 2 — دوال الخسارة: كيف نُعرِّف «الخطأ»؟"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import card, equation, footer, hero, rule, terms, verdict

hero(
    "دالة الخسارة — تعريف «الخطأ» هو أخطر قرار في المشروع",
    "Lesson 2 · Loss functions: defining what wrong means",
    "النموذج سيصغّر ما تطلب منه تصغيره، حرفياً وبلا تفكير. "
    "لذلك اختيار دالة الخسارة ليس تفصيلاً تقنياً، بل هو تعريفك أنت لمعنى «النجاح».",
)

st.markdown(
    """
> :material/format_quote: **القاعدة الذهبية:** النموذج لا يفعل ما تريد، بل يفعل ما **تقيسه**.
> إن قست الشيء الخطأ، حصلت على النموذج الخطأ — ولو كان الكود صحيحاً تماماً.
    """
)

terms(
    [
        ("Loss function", "دالة الخسارة"),
        ("Cost / Objective", "دالة التكلفة / الهدف"),
        ("Residual", "البَاقي"),
        ("Outlier", "قيمة شاذّة"),
        ("Robustness", "المتانة"),
        ("Convexity", "التحدّب"),
        ("Differentiable", "قابلة للاشتقاق"),
    ]
)

rule()

# ======================================================================
# 1. خسائر الانحدار
# ======================================================================
st.header(":material/show_chart: أولاً: دوال خسارة الانحدار (Regression)")

st.markdown(
    "في الانحدار، الخطأ على مثال واحد هو **البَاقي** $e_i = y_i - \\hat{y}_i$. "
    "كل دالة خسارة هي طريقة مختلفة لتحويل هذا الباقي إلى «عقوبة»."
)

r1, r2 = st.columns(2, gap="medium")
with r1:
    card(
        "MSE — متوسط مربع الخطأ",
        "تربّع الباقي، فتعاقب الأخطاء الكبيرة <b>بشدة</b> (خطأ 10 يساوي 100 عقوبة، "
        "وخطأ 1 يساوي 1). قابلة للاشتقاق في كل نقطة ومحدّبة، لذلك هي الأسهل للتحسين.",
        accent="blue",
        eng="Mean Squared Error · sensitive to outliers",
    )
    equation(
        r"\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}\big(y_i-\hat{y}_i\big)^{2}",
        "وحدتها هي <b>مربّع</b> وحدة الهدف (دينار²)، لذلك يصعب تفسيرها مباشرة.",
    )
    card(
        "RMSE — الجذر التربيعي لـ MSE",
        "نفس الترتيب تماماً لكن <b>بوحدة الهدف نفسها</b>، فتصبح قابلة للتفسير: "
        "«متوسط خطئي حوالي 12 ألف دينار».",
        accent="cyan",
        eng="Root Mean Squared Error · same unit as y",
    )
    equation(
        r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(y_i-\hat{y}_i\big)^{2}}",
        "ملاحظة: تصغير RMSE يكافئ تماماً تصغير MSE (الجذر دالة متزايدة).",
    )

with r2:
    card(
        "MAE — متوسط الخطأ المطلق",
        "تأخذ القيمة المطلقة فقط، فتعاقب كل الأخطاء <b>خطّياً</b>. "
        "نتيجة ذلك: <b>متينة أمام القيم الشاذّة</b> — نقطة غريبة واحدة لن تُفسد النموذج كله.",
        accent="green",
        eng="Mean Absolute Error · robust to outliers",
    )
    equation(
        r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}\big|y_i-\hat{y}_i\big|",
        "عيبها الرياضي: غير قابلة للاشتقاق عند الصفر، وتدرّجها ثابت "
        "(±1) فلا «يتباطأ» قرب الحل.",
    )
    card(
        "Huber — الحل الوسط",
        "تتصرّف مثل MSE للأخطاء الصغيرة (ناعمة وسريعة التقارب) ومثل MAE "
        "للأخطاء الكبيرة (متينة). العتبة δ معامل فائق تختاره أنت.",
        accent="violet",
        eng="Huber loss · δ threshold",
    )
    equation(
        r"L_{\delta}(e)=\begin{cases}\tfrac{1}{2}e^{2} & |e|\le\delta\\[4pt]"
        r"\delta\big(|e|-\tfrac{1}{2}\delta\big) & |e|>\delta\end{cases}",
        "قبل العتبة δ: منحنى (تربيعي). بعدها: خط مستقيم (خطّي).",
    )

st.subheader(":material/tune: قارن أشكال الخسائر بنفسك")

hc1, hc2 = st.columns([1, 1], gap="medium")
with hc1:
    delta = st.slider("عتبة Huber  δ", 0.2, 3.0, 1.0, 0.1, key="ls_delta")
with hc2:
    chosen = st.pills(
        "أي دوال تريد مقارنتها؟",
        ["MSE", "MAE", "Huber", "Log-cosh"],
        selection_mode="multi",
        default=["MSE", "MAE", "Huber"],
        key="ls_pick",
    )

e = np.linspace(-4, 4, 400)
series = {
    "MSE": (e**2, viz.C_BLUE, "تربيعية — تنفجر بسرعة"),
    "MAE": (np.abs(e), viz.C_GREEN, "خطّية — نمو ثابت"),
    "Huber": (
        np.where(np.abs(e) <= delta, 0.5 * e**2, delta * (np.abs(e) - 0.5 * delta)),
        viz.C_VIOLET,
        f"تربيعية حتى δ={delta:.1f} ثم خطّية",
    ),
    "Log-cosh": (np.log(np.cosh(e)), viz.C_AMBER, "ناعمة في كل مكان وقريبة من Huber"),
}

fig = go.Figure()
for name, (vals, col, desc) in series.items():
    if name in (chosen or []):
        fig.add_trace(
            go.Scatter(
                x=e,
                y=vals,
                mode="lines",
                name=f"{name} — {desc}",
                line=dict(color=col, width=4),
                hovertemplate="e=%{x:.2f}<br>loss=%{y:.3f}<extra></extra>",
            )
        )
if "Huber" in (chosen or []):
    for s in (-delta, delta):
        fig.add_vline(
            x=s, line=dict(color=viz.rgba(viz.C_VIOLET, 0.5), width=2, dash="dot")
        )
viz.style_fig(
    fig,
    height=420,
    xtitle="e = y − ŷ   (الباقي)",
    ytitle="قيمة الخسارة",
    title="كلما ارتفع المنحنى أسرع، زادت عقوبة الأخطاء الكبيرة",
)
st.plotly_chart(fig, width="stretch", key="ls_shapes")

verdict(
    "كيف تقرأ هذا الرسم؟",
    "انظر إلى <b>مدى انحدار المنحنى عند الأطراف</b>. MSE ترتفع كالصاروخ، "
    "فيصبح النموذج مهووساً بإرضاء النقاط الشاذّة. MAE ترتفع بهدوء، فيتجاهلها. "
    "Huber تجمع بين الاثنين: ناعمة في المنتصف، هادئة عند الأطراف.",
    tone="violet",
)

rule()

# ======================================================================
# 2. أثر القيم الشاذّة
# ======================================================================
st.header(":material/warning: تجربة حاسمة: ماذا تفعل قيمة شاذّة واحدة؟")

st.markdown(
    """
سنأخذ بيانات نظيفة تماماً على خط مستقيم، ثم نضيف **نقطة شاذّة واحدة فقط**
ونراقب كيف ينحرف الخط المُدرَّب بـ MSE بينما يصمد الخط المُدرَّب بـ MAE.
    """
)

oc1, oc2 = st.columns([1, 1], gap="medium")
with oc1:
    out_y = st.slider("ارتفاع القيمة الشاذّة", 0.0, 90.0, 55.0, 2.5, key="ls_out")
with oc2:
    out_x = st.slider("موقعها على المحور الأفقي", 0.5, 9.5, 8.5, 0.5, key="ls_outx")

rng = np.random.default_rng(1)
xc = np.linspace(0.5, 9.5, 24)
yc = 3.0 + 1.8 * xc + rng.normal(0, 0.9, xc.size)

X_all = np.append(xc, out_x)
y_all = np.append(yc, out_y)


def fit_mse(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_mae(x, y, iters=260, lr=0.02):
    """Simple sub-gradient descent on the absolute error."""
    w, b = fit_mse(x, y)
    for _ in range(iters):
        r = y - (w * x + b)
        s = np.sign(r)
        w += lr * np.mean(s * x)
        b += lr * np.mean(s)
    return w, b


w_mse, b_mse = fit_mse(X_all, y_all)
w_mae, b_mae = fit_mae(X_all, y_all)
w_clean, b_clean = fit_mse(xc, yc)

grid = np.linspace(0, 10, 50)
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=xc,
        y=yc,
        mode="markers",
        name="بيانات نظيفة",
        marker=dict(size=10, color=viz.rgba(viz.C_BLUE, 0.8), line=dict(color="#fff", width=1.5)),
        hoverinfo="skip",
    )
)
fig2.add_trace(
    go.Scatter(
        x=[out_x],
        y=[out_y],
        mode="markers+text",
        name="القيمة الشاذّة",
        marker=dict(size=22, color=viz.C_ROSE, symbol="x-thin",
                    line=dict(color=viz.C_ROSE, width=4)),
        text=["شاذّة"],
        textposition="top center",
        textfont=dict(color=viz.C_ROSE, size=12),
        hoverinfo="skip",
    )
)
fig2.add_trace(
    go.Scatter(
        x=grid, y=w_clean * grid + b_clean, mode="lines",
        name="الخط الحقيقي (بدون الشاذّة)",
        line=dict(color=viz.rgba(viz.C_SLATE, 0.75), width=3, dash="dot"),
        hoverinfo="skip",
    )
)
fig2.add_trace(
    go.Scatter(
        x=grid, y=w_mse * grid + b_mse, mode="lines",
        name="مُدرَّب بـ MSE",
        line=dict(color=viz.C_AMBER, width=4.5),
        hovertemplate="MSE fit<extra></extra>",
    )
)
fig2.add_trace(
    go.Scatter(
        x=grid, y=w_mae * grid + b_mae, mode="lines",
        name="مُدرَّب بـ MAE",
        line=dict(color=viz.C_GREEN, width=4.5),
        hovertemplate="MAE fit<extra></extra>",
    )
)
viz.style_fig(
    fig2,
    height=440,
    xtitle="x",
    ytitle="y",
    title="نقطة واحدة تكفي لجرّ خط MSE بعيداً",
)
st.plotly_chart(fig2, width="stretch", key="ls_outlier")

d1, d2, d3 = st.columns(3)
d1.metric("ميل الخط الحقيقي", f"{w_clean:.3f}")
d2.metric("ميل خط MSE", f"{w_mse:.3f}", delta=f"{w_mse - w_clean:+.3f}", delta_color="inverse")
d3.metric("ميل خط MAE", f"{w_mae:.3f}", delta=f"{w_mae - w_clean:+.3f}", delta_color="inverse")

if abs(w_mse - w_clean) > abs(w_mae - w_clean) + 0.02:
    st.warning(
        "لاحظ الأرقام: انحراف خط **MSE** عن الحقيقة أكبر من انحراف خط **MAE**. "
        "هذا هو معنى «المتانة أمام القيم الشاذّة» عملياً."
    )

verdict(
    "متى تختار MAE بدل MSE؟",
    "عندما تحتوي بياناتك على <b>قيم شاذّة حقيقية لا تريد أن يطاردها النموذج</b> "
    "(أخطاء إدخال، أحداث استثنائية، قياسات معطوبة). "
    "وعندما تكون الأخطاء الكبيرة <b>مكلفة فعلاً</b> (تنبؤ طبي، هندسي) استعمل MSE "
    "لأنك <b>تريد</b> أن يعاقبها النموذج بشدة.",
    tone="green",
)

rule()

# ======================================================================
# 3. خسائر التصنيف
# ======================================================================
st.header(":material/category: ثانياً: دوال خسارة التصنيف (Classification)")

st.markdown(
    """
في التصنيف، النموذج لا يُخرج فئة مباشرة بل **احتمالاً** $p \\in [0,1]$.
ولا يمكننا استعمال MSE هنا بكفاءة، لسببين:
نمنح تدرّجات ضعيفة جداً عندما يكون النموذج **واثقاً وخاطئاً**، ودالة التكلفة تفقد تحدّبها.

الحل هو **الإنتروبيا المتقاطعة (Cross-Entropy)**، وتُسمّى أيضاً **Log Loss**:
    """
)

equation(
    r"\ell\big(y,p\big) = -\Big[\,y\log(p) + (1-y)\log(1-p)\,\Big],\qquad y\in\{0,1\}",
    "اقرأها ببساطة: إذا كان الجواب الصحيح y=1 يبقى الحد الأول فقط ← الخسارة = −log(p). "
    "وإذا كان y=0 يبقى الحد الثاني ← الخسارة = −log(1−p). "
    "أي: <b>الخسارة هي سالب لوغاريتم الاحتمال الذي أعطاه النموذج للإجابة الصحيحة</b>.",
)

equation(
    r"J(\boldsymbol{\theta}) = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i\log(p_i)"
    r"+(1-y_i)\log(1-p_i)\,\Big]",
    "وهذا متوسطها على كل البيانات — أي دالة التكلفة التي تُصغَّر فعلياً.",
)

st.markdown("**لماذا اللوغاريتم؟** لأنه يعاقب الثقة الخاطئة عقاباً لا نهائياً. حرّك المؤشر وشاهد:")

p_user = st.slider(
    "الاحتمال الذي أعطاه النموذج للفئة الصحيحة  p", 0.01, 0.99, 0.60, 0.01, key="ls_p"
)

pp = np.linspace(0.001, 0.999, 400)
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=pp, y=-np.log(pp), mode="lines",
        name="−log(p)  ← الخسارة عندما y = 1",
        line=dict(color=viz.C_BLUE, width=4),
        hovertemplate="p=%{x:.2f}<br>loss=%{y:.3f}<extra></extra>",
    )
)
fig3.add_trace(
    go.Scatter(
        x=pp, y=-np.log(1 - pp), mode="lines",
        name="−log(1−p)  ← الخسارة عندما y = 0",
        line=dict(color=viz.C_ROSE, width=4, dash="dash"),
        hovertemplate="p=%{x:.2f}<br>loss=%{y:.3f}<extra></extra>",
    )
)
fig3.add_trace(
    go.Scatter(
        x=[p_user], y=[-np.log(p_user)], mode="markers",
        name="موضعك",
        marker=dict(size=18, color=viz.C_AMBER, line=dict(color="#fff", width=3)),
        hovertemplate="p=%{x:.2f}<extra></extra>",
    )
)
fig3.update_yaxes(range=[0, 5])
viz.style_fig(
    fig3, height=400,
    xtitle="p — الاحتمال المتوقَّع",
    ytitle="الخسارة",
    title="كلما اقترب p من الصفر مع كون الجواب 1، انفجرت الخسارة",
)
st.plotly_chart(fig3, width="stretch", key="ls_ce")

cm1, cm2, cm3 = st.columns(3)
cm1.metric("الخسارة إذا كان y = 1", f"{-np.log(p_user):.3f}")
cm2.metric("الخسارة إذا كان y = 0", f"{-np.log(1 - p_user):.3f}")
cm3.metric("الخسارة عند تخمين عشوائي (p=0.5)", f"{-np.log(0.5):.3f}")

st.info(
    "**مرجع مهم:** خسارة التخمين العشوائي في تصنيف ثنائي متوازن هي "
    "$-\\log(0.5) \\approx 0.693$. إذا كانت خسارة نموذجك أكبر من ذلك، "
    "فهو **أسوأ من رمي عملة معدنية**."
)

st.subheader("دوال خسارة التصنيف الأخرى")

st.markdown(
    """
| الدالة | الصيغة | أين تُستعمل |
|---|---|---|
| **Binary Cross-Entropy** | $-[y\\log p+(1-y)\\log(1-p)]$ | تصنيف ثنائي (الانحدار اللوجستي، الشبكات) |
| **Categorical Cross-Entropy** | $-\\sum_{k=1}^{K} y_k \\log p_k$ | تصنيف متعدّد الفئات مع Softmax |
| **Hinge Loss** | $\\max(0,\\;1-y\\cdot f(x))$، $y\\in\\{-1,1\\}$ | آلات المتجهات الداعمة (SVM) |
| **Focal Loss** | $-(1-p)^{\\gamma}\\log p$ | بيانات شديدة اللاتوازن (كشف الأجسام) |
| **KL Divergence** | $\\sum p\\log\\frac{p}{q}$ | مقارنة توزيعين (التقطير، VAE) |
    """
)

rule()

# ======================================================================
# 4. جدول القرار
# ======================================================================
st.header(":material/fact_check: جدول القرار: أي دالة خسارة لأي مشكلة؟")

st.markdown(
    """
| المشكلة | الخسارة المقترحة | السبب |
|---|---|---|
| انحدار ببيانات نظيفة | **MSE / RMSE** | ناعمة، محدّبة، تقارب سريع |
| انحدار بقيم شاذّة | **MAE** أو **Huber** | متانة أمام الشذوذ |
| انحدار على قيم موجبة متفاوتة الحجم (أسعار، مبيعات) | **MSLE** أو MSE على $\\log y$ | يهتم بالخطأ النسبي لا المطلق |
| تنبؤ بكميات (أريد الكمّية 90٪) | **Pinball / Quantile loss** | يسمح بتقدير الكمّيات لا المتوسط |
| تصنيف ثنائي | **Binary Cross-Entropy** | تدرّجات قوية، احتمالات معايرة |
| تصنيف متعدّد | **Categorical Cross-Entropy** | التعميم الطبيعي للحالة الثنائية |
| تصنيف غير متوازن بشدة | **Weighted CE** أو **Focal** | يمنع طغيان الفئة الأغلبية |
| هوامش قصوى (SVM) | **Hinge** | يُعظّم هامش الفصل |

:material/priority_high: **تحذير يُكلّف كثيرين:** الخسارة التي تُحسِّنها (Loss) قد تختلف عن
المقياس الذي يُحاسَب عليه مشروعك (Metric). مثال: نُحسِّن Cross-Entropy لكن الإدارة تقيسك
بـ F1. اعرف الفرق، واضبط **عتبة القرار** لاحقاً على بيانات التحقق لتحسين المقياس المطلوب.
    """
)

verdict(
    "الخلاصة",
    "دالة الخسارة هي <b>العقد الذي توقّعه مع النموذج</b>. "
    "اكتب العقد بعناية: اسأل نفسك «ما نوع الخطأ الذي يؤلمني فعلاً؟» "
    "ثم اختر الدالة التي تعاقب هذا النوع تحديداً.",
    tone="blue",
)

footer("الدرس التالي: كيف ينزل الحاسوب فعلياً إلى قاع دالة الخسارة؟ (النزول التدرّجي)")

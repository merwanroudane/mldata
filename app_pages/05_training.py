"""الدرس 4 — التشريح الرياضي الكامل لحلقة التدريب."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import card, equation, footer, hero, rule, steps, terms, verdict

hero(
    "التشريح الرياضي للتدريب — القطع الخمس",
    "Lesson 4 · Anatomy of the training process",
    "في الدروس الثلاثة السابقة فهمت الفكرة بلغة عادية. الآن نكتبها بدقّة: "
    "خمس قطع فقط — نموذج، معاملات، دالة خسارة، دالة تكلفة، ومُحسِّن — "
    "كل واحدة بالكلمات وبالصيغة الرياضية معاً.",
)

# ======================================================================
# 1. الصورة الكبرى
# ======================================================================
st.header(":material/loop: الصورة الكبرى: حلقة التدريب")

st.markdown(
    """
كل تدريب في الكون — من انحدار خطي بسيط إلى نموذج لغوي ضخم — هو **نفس الحلقة**
المكرّرة ملايين المرات. احفظ هذه الحلقة وستكون قد فهمت تعلّم الآلة:
    """
)

st.mermaid_chart(
    """
graph LR
    A["1. البيانات<br/>Batch of (X, y)"] --> B["2. التنبؤ<br/>Forward pass<br/>ŷ = f_θ(X)"]
    B --> C["3. قياس الخطأ<br/>Loss  L(y, ŷ)"]
    C --> D["4. حساب الميل<br/>Gradient  ∇J(θ)"]
    D --> E["5. تحديث المعاملات<br/>θ ← θ − η∇J(θ)"]
    E --> A
    C -.->|"الخطأ صغير وثابت"| F["توقّف<br/>Converged"]

    style A fill:#E3F0FE,stroke:#3B7DD8,stroke-width:2px,color:#17263F
    style B fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
    style C fill:#FDE6EE,stroke:#EB6F92,stroke-width:2px,color:#17263F
    style D fill:#FDF0DC,stroke:#F0A33C,stroke-width:2px,color:#17263F
    style E fill:#EEE6FB,stroke:#9B6BD6,stroke-width:2px,color:#17263F
    style F fill:#DEF3F8,stroke:#2FA8C7,stroke-width:2px,color:#17263F
    """
)

verdict(
    "اقرأ الحلقة بالعربية الدارجة",
    "«خُذ بعض الأمثلة ← خمّن الجواب ← قارن بالجواب الصحيح وقِس كم أخطأت ← "
    "اسأل: في أي اتجاه أعدّل أرقامي ليقلّ الخطأ؟ ← عدّل قليلاً ← كرّر». "
    "هذا هو التدريب حرفياً، ولا شيء غيره.",
    tone="violet",
)

terms(
    [
        ("Training", "التدريب"),
        ("Forward pass", "التمرير الأمامي"),
        ("Loss", "الخسارة"),
        ("Gradient", "التدرّج"),
        ("Update rule", "قاعدة التحديث"),
        ("Convergence", "التقارب"),
        ("Iteration", "تكرارة"),
        ("Epoch", "حقبة"),
    ]
)

rule()

# ======================================================================
# 2. القطعة 1 — النموذج والمعاملات
# ======================================================================
st.header(":material/widgets: القطعة 1 — النموذج (Model) والمعاملات (Parameters)")

st.markdown(
    """
**النموذج** هو ببساطة **دالة رياضية** تأخذ المدخلات وتُخرج تنبؤاً. نكتبها:
    """
)

equation(
    r"\hat{y} \;=\; f_{\boldsymbol{\theta}}(\mathbf{x})",
    "اقرأها: «التنبؤ ŷ يساوي الدالة f، المضبوطة بالمعاملات θ، مطبّقة على المثال x». "
    "الحرف <b>θ</b> (ثيتا) هو الاسم العام لكل الأرقام الداخلية التي سيتعلّمها النموذج.",
)

st.markdown(
    """
القوس السفلي $\\boldsymbol{\\theta}$ مهم جداً: هو يقول إن شكل الدالة ثابت (خط مستقيم مثلاً)،
لكن **موضعها قابل للضبط**. التدريب لا يغيّر شكل الدالة، بل يضبط أرقامها فقط.
    """
)

c1, c2 = st.columns(2, gap="medium")
with c1:
    card(
        "النموذج الخطي — أبسط حالة",
        "نفترض أن العلاقة خط مستقيم (أو مستوٍ في أبعاد أعلى). "
        "المعاملات هنا هي الأوزان <b>w</b> والانحياز <b>b</b>.",
        accent="blue",
        eng="Linear model · weights w · bias b",
    )
    equation(
        r"\hat{y} \;=\; w_1x_1 + w_2x_2 + \dots + w_px_p + b"
        r"\;=\; \mathbf{w}^{\top}\mathbf{x} + b",
        "كل خصيصة لها وزن يقول «كم تؤثّر هذه الخصيصة». "
        "و b يرفع أو يخفض الخط كله.",
    )
with c2:
    card(
        "فضاء الفرضيات",
        "مجموعة كل الدوال التي يستطيع نموذجك تمثيلها. النموذج الخطي فضاؤه "
        "كل المستقيمات فقط؛ الشبكة العصبية فضاؤها هائل. "
        "<b>التدريب = اختيار أفضل عنصر من هذا الفضاء</b>.",
        accent="violet",
        eng="Hypothesis space / Model capacity",
    )
    card(
        "لماذا هذا مهم؟",
        "إن كانت العلاقة الحقيقية منحنية وفضاؤك مستقيمات فقط، فلن ينجح التدريب "
        "مهما طال — هذه هي <b>مشكلة التقصير (Underfitting)</b> التي سنراها لاحقاً.",
        accent="rose",
    )

st.subheader("معاملات أم معاملات فائقة؟ الفرق الذي يربك الجميع")

st.markdown(
    """
| | **المعاملات (Parameters)** | **المعاملات الفائقة (Hyperparameters)** |
|---|---|---|
| **من يحدّدها؟** | الخوارزمية أثناء التدريب | أنت، **قبل** التدريب |
| **أمثلة** | $w$، $b$، أوزان الشبكة العصبية | معدّل التعلّم $\\eta$، عدد الحقب، $k$ في KNN، عمق الشجرة، $\\lambda$ |
| **كيف تُختار؟** | بتصغير دالة التكلفة على بيانات التدريب | بالتجريب على **بيانات التحقق (Validation)** |
| **الرمز الشائع** | $\\theta$، $\\mathbf{w}$ | $\\eta$، $\\lambda$، $k$ |
| **عددها** | قد يكون ملايير | عادة من 2 إلى 20 |

:material/lightbulb: **قاعدة ذهبية:** إذا كان الرقم يتغيّر تلقائياً أثناء التدريب فهو **معامل**؛
وإذا كنت أنت من يكتبه في الكود قبل الضغط على «شغّل» فهو **معامل فائق**.
    """
)

rule()

# ======================================================================
# 3. القطعة 2 — الخسارة
# ======================================================================
st.header(":material/error_outline: القطعة 2 — دالة الخسارة (Loss) ودالة التكلفة (Cost)")

st.markdown(
    """
النموذج يحتاج **مُصحِّحاً** يقول له كم أخطأ. هذا المصحِّح هو دالة الخسارة.
انتبه للفرق الدقيق بين كلمتين كثيراً ما تُخلطان:
    """
)

l1, l2 = st.columns(2, gap="medium")
with l1:
    card(
        "دالة الخسارة — Loss",
        "تقيس الخطأ على <b>مثال واحد فقط</b>. مثال: الفرق بين السعر الحقيقي "
        "والسعر المتوقَّع لمنزل واحد، مربّعاً.",
        accent="rose",
        eng="Loss = error on ONE example",
    )
    equation(
        r"\ell\big(y_i,\hat{y}_i\big) \;=\; \big(y_i-\hat{y}_i\big)^2",
        "خسارة المربع لمثال واحد. نربّع لسببين: لنجعل الخطأ موجباً دائماً، "
        "ولنعاقب الأخطاء الكبيرة أكثر.",
    )
with l2:
    card(
        "دالة التكلفة — Cost / Objective",
        "تقيس الخطأ على <b>كل البيانات</b> دفعة واحدة: هي ببساطة <b>متوسط</b> "
        "الخسائر الفردية. هذه هي الدالة التي نُصغّرها فعلياً.",
        accent="amber",
        eng="Cost J(θ) = average loss over the dataset",
    )
    equation(
        r"J(\boldsymbol{\theta}) \;=\; \frac{1}{n}\sum_{i=1}^{n}"
        r"\ell\big(y_i,\;f_{\boldsymbol{\theta}}(\mathbf{x}_i)\big)",
        "الرمز Σ يعني «اجمع من المثال الأول إلى المثال n»، ثم اقسم على n "
        "للحصول على المتوسط. النتيجة: <b>رقم واحد</b> يلخّص أداء النموذج كله.",
    )

verdict(
    "لماذا نحتاج رقماً واحداً؟",
    "لأن التحسين الرياضي يحتاج <b>هدفاً وحيداً</b> يُصغَّر. لو كان لدينا ألف رقم خطأ "
    "لما عرفنا أي نموذج أفضل. دالة التكلفة تضغط كل الأخطاء في رقم واحد "
    "يمكن مقارنته والنزول به.",
    tone="amber",
)

st.markdown(
    """
في الانحدار الخطي، إذا عوّضنا بالنموذج الخطي داخل دالة التكلفة نحصل على
الصيغة الشهيرة **متوسط مربع الخطأ (Mean Squared Error)**:
    """
)

equation(
    r"J(\mathbf{w},b)\;=\;\frac{1}{n}\sum_{i=1}^{n}"
    r"\Big(y_i-\big(\mathbf{w}^{\top}\mathbf{x}_i+b\big)\Big)^{2}",
    "هذه هي الدالة التي تراها في كل كتاب عن الانحدار. "
    "(أحياناً تُكتب مع ½ أمامها لتبسيط المشتقة، وهذا لا يغيّر مكان الحد الأدنى.)",
)

rule()

# ======================================================================
# 4. القطعة 3 — التحسين
# ======================================================================
st.header(":material/tune: القطعة 3 — التحسين (Optimization): البحث عن أقل قيمة")

st.markdown(
    """
الآن صار السؤال رياضياً بحتاً وواضحاً تماماً:
    """
)

equation(
    r"\boldsymbol{\theta}^{*} \;=\; \operatorname*{arg\,min}_{\boldsymbol{\theta}}\; J(\boldsymbol{\theta})",
    "اقرأها: «θ النجمة (الحل الأمثل) هي قيمة θ التي تجعل J أصغر ما يمكن». "
    "رمز <b>arg min</b> يعني: لا نريد قيمة الدالة الصغرى، بل نريد <b>الوسيط</b> الذي يحقّقها.",
)

st.markdown(
    """
كيف نجد هذا الحد الأدنى؟ هناك طريقان:
    """
)

o1, o2 = st.columns(2, gap="medium")
with o1:
    card(
        "الطريق الأول: الحل المغلق",
        "في حالات قليلة جداً (الانحدار الخطي مثلاً) يمكن اشتقاق الدالة، مساواتها "
        "بالصفر، وحلّها جبرياً دفعة واحدة. نتيجة فورية ودقيقة.",
        accent="green",
        eng="Closed-form / Normal equation",
    )
    equation(
        r"\hat{\mathbf{w}}=\big(\mathbf{X}^{\top}\mathbf{X}\big)^{-1}\mathbf{X}^{\top}\mathbf{y}",
        "معادلة المربعات الصغرى العادية (OLS). "
        "عيبها: قلب المصفوفة مكلف جداً إذا كان عدد الخصائص كبيراً، "
        "وغير ممكن أصلاً لمعظم النماذج.",
    )
with o2:
    card(
        "الطريق الثاني: التحسين التكراري",
        "ابدأ من نقطة عشوائية، وانزل خطوة خطوة في اتجاه الانحدار حتى تصل "
        "إلى القاع. <b>هذا هو المستخدم في 99٪ من تعلّم الآلة الحديث</b>.",
        accent="blue",
        eng="Iterative optimization · Gradient Descent",
    )
    equation(
        r"\boldsymbol{\theta}^{(t+1)} \;=\; \boldsymbol{\theta}^{(t)}"
        r"\;-\;\eta\,\nabla_{\boldsymbol{\theta}} J\big(\boldsymbol{\theta}^{(t)}\big)",
        "قاعدة التحديث: المعاملات الجديدة = القديمة − (معدّل التعلّم × التدرّج). "
        "الإشارة السالبة لأننا نريد النزول عكس اتجاه الصعود.",
    )

rule()

# ======================================================================
# 5. فهم المشتقة بصرياً
# ======================================================================
st.header(":material/show_chart: فهم «التدرّج» بصرياً — ما معنى المشتقة؟")

st.markdown(
    """
**المشتقة (Derivative)** عند نقطة هي ببساطة **ميل المماس** عند تلك النقطة.
تخيّلها كبوصلة تقول لك: «هل الأرض تحتك صاعدة أم هابطة، وبأي حدّة؟»

- ميل **موجب** ← الدالة تصعد إلى اليمين ← للنزول يجب أن أتحرّك **يساراً**.
- ميل **سالب** ← الدالة تهبط إلى اليمين ← للنزول يجب أن أتحرّك **يميناً**.
- ميل **صفر** ← أنا في القاع (أو القمة) ← توقّف.

لاحظ أن قاعدة التحديث $\\theta \\leftarrow \\theta - \\eta J'(\\theta)$ تفعل هذا تلقائياً:
الإشارة السالبة تعكس اتجاه الميل دائماً.

**حرّك المؤشر** لترى المماس يتحرّك والخطوة التالية تُحسب أمامك:
    """
)

gc1, gc2 = st.columns([1, 1], gap="medium")
with gc1:
    w_pos = st.slider(
        "موضع المعامل الحالي  w", -4.0, 4.0, -3.0, 0.1, key="tr_wpos"
    )
with gc2:
    eta_vis = st.slider(
        "معدّل التعلّم  η  (Learning rate)", 0.01, 0.9, 0.30, 0.01, key="tr_eta"
    )


def cost_1d(w: np.ndarray) -> np.ndarray:
    return 0.55 * (w - 1.2) ** 2 + 0.9


def grad_1d(w: float) -> float:
    return 1.10 * (w - 1.2)


w_grid = np.linspace(-4.4, 4.4, 300)
j_grid = cost_1d(w_grid)
j_here = float(cost_1d(np.array([w_pos]))[0])
g_here = grad_1d(w_pos)
w_next = w_pos - eta_vis * g_here
j_next = float(cost_1d(np.array([w_next]))[0])

tan_x = np.linspace(w_pos - 1.5, w_pos + 1.5, 20)
tan_y = j_here + g_here * (tan_x - w_pos)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=w_grid,
        y=j_grid,
        mode="lines",
        name="دالة التكلفة J(w)",
        line=dict(color=viz.C_BLUE, width=4),
        hovertemplate="w=%{x:.2f}<br>J=%{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=tan_x,
        y=tan_y,
        mode="lines",
        name=f"المماس · الميل = {g_here:+.2f}",
        line=dict(color=viz.C_AMBER, width=3, dash="dash"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[w_pos],
        y=[j_here],
        mode="markers+text",
        name="الموضع الحالي",
        marker=dict(size=18, color=viz.C_ROSE, line=dict(color="#fff", width=3)),
        text=["الآن"],
        textposition="top center",
        textfont=dict(size=12, color=viz.C_ROSE),
        hovertemplate="w=%{x:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[w_next],
        y=[j_next],
        mode="markers+text",
        name="بعد خطوة واحدة",
        marker=dict(
            size=16,
            color=viz.C_GREEN,
            symbol="diamond",
            line=dict(color="#fff", width=2),
        ),
        text=["الخطوة التالية"],
        textposition="bottom center",
        textfont=dict(size=11, color=viz.C_GREEN),
        hovertemplate="w=%{x:.2f}<extra></extra>",
    )
)
fig.add_annotation(
    x=w_next,
    y=j_next,
    ax=w_pos,
    ay=j_here,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1.3,
    arrowwidth=3,
    arrowcolor=viz.C_VIOLET,
)
fig.add_trace(
    go.Scatter(
        x=[1.2],
        y=[cost_1d(np.array([1.2]))[0]],
        mode="markers+text",
        name="الحد الأدنى الحقيقي",
        marker=dict(size=14, color=viz.C_CYAN, symbol="star"),
        text=["القاع"],
        textposition="bottom center",
        textfont=dict(size=11, color=viz.C_CYAN),
        hoverinfo="skip",
    )
)
viz.style_fig(
    fig,
    height=430,
    xtitle="w — قيمة المعامل",
    ytitle="J(w) — التكلفة",
    title="المماس يقول لك إلى أين تتحرّك",
)
st.plotly_chart(fig, width="stretch", key="tr_tangent")

q1, q2, q3, q4 = st.columns(4)
q1.metric("J(w) الحالية", f"{j_here:.3f}")
q2.metric("الميل J'(w)", f"{g_here:+.3f}")
q3.metric("حجم الخطوة η·J'(w)", f"{eta_vis * g_here:+.3f}")
q4.metric("J بعد الخطوة", f"{j_next:.3f}", delta=f"{j_next - j_here:+.3f}", delta_color="inverse")

if abs(g_here) < 0.05:
    st.success("الميل يكاد يكون صفراً ← أنت في القاع ← **تقارب (Convergence)**. التدريب انتهى.")
elif j_next > j_here:
    st.error(
        "انتبه! التكلفة **ارتفعت** بعد الخطوة. معدّل التعلّم كبير جداً فقفزت فوق القاع. "
        "هذه هي ظاهرة **التباعد (Divergence)**."
    )
else:
    st.info(
        f"التكلفة نزلت من {j_here:.3f} إلى {j_next:.3f}. كرّر هذه الخطوة آلاف المرات "
        "وستصل إلى القاع: **هذا هو التدريب**."
    )

rule()

# ======================================================================
# 6. Epoch / Iteration / Batch
# ======================================================================
st.header(":material/repeat_on: الحقبة، التكرارة، والدُفعة — ثلاث كلمات تُخلط دائماً")

st.markdown(
    """
هذه أكثر ثلاثة مصطلحات تربك المبتدئ. المثال التالي يحسمها نهائياً.
    """
)

e1, e2, e3 = st.columns(3, gap="medium")
with e1:
    n_total = st.number_input("حجم بيانات التدريب n", 100, 100_000, 1_000, step=100, key="tr_n")
with e2:
    bs = st.select_slider(
        "حجم الدُفعة (Batch size)",
        options=[1, 8, 16, 32, 64, 128, 256, 512, 1024],
        value=32,
        key="tr_bs",
    )
with e3:
    n_ep = st.number_input("عدد الحقب (Epochs)", 1, 500, 10, key="tr_ep")

iters_per_epoch = int(np.ceil(n_total / bs))
total_iters = iters_per_epoch * int(n_ep)

mm1, mm2, mm3 = st.columns(3)
mm1.metric("تكرارات في كل حقبة", f"{iters_per_epoch:,}")
mm2.metric("إجمالي التكرارات", f"{total_iters:,}")
mm3.metric("مرات تحديث المعاملات", f"{total_iters:,}")

st.markdown(
    f"""
- **الدُفعة (Batch)** = عدد الأمثلة التي ننظر إليها قبل تحديث واحد للمعاملات. هنا **{bs}**.
- **التكرارة (Iteration / Step)** = **تحديث واحد** للمعاملات. تحتاج دفعة واحدة.
- **الحقبة (Epoch)** = مرور كامل على **كل** بيانات التدريب مرة واحدة.
  هنا تحتاج ⌈{n_total:,} ÷ {bs}⌉ = **{iters_per_epoch:,}** تكرارة.

:material/warning: خطأ شائع: الظنّ أن «حقبة واحدة = تحديث واحد». لا — حقبة واحدة
قد تعني آلاف التحديثات إذا كانت الدُفعة صغيرة.
    """
)

st.subheader("ثلاث عائلات من النزول التدرّجي حسب حجم الدُفعة")

st.markdown(
    """
| النوع | حجم الدُفعة | المزايا | العيوب |
|---|---|---|---|
| **Batch GD** (دفعي كامل) | $n$ (كل البيانات) | مسار ناعم ومستقر، تدرّج دقيق | بطيء جداً، يحتاج ذاكرة هائلة |
| **Stochastic GD (SGD)** | $1$ | سريع جداً، قد يهرب من الحدود الدنيا المحلّية | مسار مهتز وضجيج عالٍ |
| **Mini-batch GD** | $8\\!-\\!512$ عادة | **التوازن الأفضل**، يستفيد من تسريع الـ GPU | يضيف معاملاً فائقاً جديداً |

:material/check_circle: **الممارسة العملية:** تقريباً كل التدريب الحديث يستعمل **Mini-batch**،
وبأحجام من قوى العدد 2 (32، 64، 128) لأسباب تتعلّق بكفاءة العتاد.
    """
)

rule()

# ======================================================================
# 7. الخوارزمية كاملة
# ======================================================================
st.header(":material/terminal: الخوارزمية كاملة — من الورقة إلى الكود")

steps(
    [
        (
            "التهيئة (Initialization)",
            "نبدأ بقيم أولية للمعاملات: أصفار، أو أرقام عشوائية صغيرة. "
            "في الشبكات العصبية التهيئة العشوائية <b>ضرورية</b> (لو بدأنا بأصفار "
            "لبقيت كل العُقد متطابقة ولما تعلّم شيء).",
        ),
        (
            "اخلط البيانات (Shuffle)",
            "في بداية كل حقبة نخلط ترتيب الأمثلة حتى لا يتعلّم النموذج ترتيباً "
            "عرَضياً في الملف. (استثناء مهم: <b>السلاسل الزمنية لا تُخلط أبداً</b>.)",
        ),
        (
            "التمرير الأمامي (Forward pass)",
            "خُذ دفعة، احسب التنبؤات ŷ = f<sub>θ</sub>(X) لكل مثال فيها.",
        ),
        (
            "احسب التكلفة (Compute cost)",
            "قارن ŷ بـ y عبر دالة الخسارة، ثم خُذ المتوسط ← رقم واحد J.",
        ),
        (
            "احسب التدرّج (Backward pass)",
            "اشتق J بالنسبة لكل معامل. في الشبكات العصبية تُسمّى هذه الخطوة "
            "<b>الانتشار العكسي (Backpropagation)</b>، وهي مجرد قاعدة السلسلة في التفاضل.",
        ),
        (
            "حدِّث المعاملات (Update)",
            "θ ← θ − η∇J. هنا فقط «يتعلّم» النموذج فعلياً: أرقامه تتغيّر.",
        ),
        (
            "كرّر وراقب (Repeat & monitor)",
            "ارجع للخطوة 3 مع الدفعة التالية. بعد كل حقبة سجّل خسارة التدريب "
            "<b>وخسارة التحقق</b>، لأن الثانية هي التي تكشف الإفراط في التعلّم.",
        ),
        (
            "توقّف (Stopping criterion)",
            "توقّف عند انتهاء عدد الحقب، أو عندما تتوقّف خسارة التحقق عن التحسّن "
            "لعدد معيّن من الحقب (<b>Early stopping</b>)، أو عندما يصبح التحسّن أصغر من عتبة.",
        ),
    ],
    accent="violet",
)

tab_py, tab_np, tab_sk = st.tabs(
    [":material/description: الخوارزمية بالكلمات", ":material/code: NumPy من الصفر", ":material/bolt: scikit-learn"]
)

with tab_py:
    st.code(
        """
INPUT : X (n×p), y (n), learning_rate η, epochs E, batch_size B
OUTPUT: trained parameters θ

θ ← initialize()                       # أصفار أو عشوائي صغير
for epoch in 1 .. E:
    shuffle(X, y)                      # لا تفعلها مع السلاسل الزمنية
    for each mini-batch (Xb, yb) of size B:
        ŷ  ← f_θ(Xb)                   # 1. forward pass
        J  ← mean( loss(yb, ŷ) )       # 2. cost
        g  ← ∇_θ J                     # 3. gradient
        θ  ← θ − η · g                 # 4. update
    record train_loss, val_loss        # 5. monitor
    if val_loss stopped improving:
        break                          # 6. early stopping
return θ
        """,
        language="text",
    )

with tab_np:
    st.code(
        '''
import numpy as np


def train_linear(X, y, lr=0.05, epochs=200, batch_size=32, seed=0):
    """Mini-batch gradient descent for y ≈ Xw + b, MSE loss."""
    rng = np.random.default_rng(seed)
    n, p = X.shape

    w = np.zeros(p)          # initialisation
    b = 0.0
    history = []

    for epoch in range(epochs):
        idx = rng.permutation(n)                 # shuffle
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            Xb, yb = X[bi], y[bi]

            y_hat = Xb @ w + b                   # forward pass
            err = y_hat - yb                     # (ŷ - y)

            # ∂J/∂w = (2/m) Xᵀ(ŷ - y) ,  ∂J/∂b = (2/m) Σ(ŷ - y)
            m = len(bi)
            grad_w = (2.0 / m) * (Xb.T @ err)
            grad_b = (2.0 / m) * err.sum()

            w -= lr * grad_w                     # update
            b -= lr * grad_b

        mse = float(np.mean((X @ w + b - y) ** 2))
        history.append(mse)

    return w, b, history
        ''',
        language="python",
    )
    st.caption(
        "هذا هو كل تعلّم الآلة في 25 سطراً. كل ما تبقّى في المكتبات الكبرى "
        "هو تحسينات على هذه الفكرة نفسها."
    )

with tab_sk:
    st.code(
        """
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),                 # التقييس داخل الـ pipeline (مهم جداً)
    SGDRegressor(
        loss="squared_error",         # دالة الخسارة
        learning_rate="invscaling",   # جدولة معدّل التعلّم
        eta0=0.01,                    # معدّل التعلّم الابتدائي η
        max_iter=1000,                # الحد الأقصى للحقب
        tol=1e-4,                     # عتبة التوقّف
        early_stopping=True,          # توقّف مبكّر على بيانات تحقق داخلية
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=0,
    ),
)

model.fit(X_train, y_train)           # ← هنا تحدث حلقة التدريب كاملة
y_pred = model.predict(X_test)
        """,
        language="python",
    )
    st.caption(
        "لاحظ أن كل المعاملات الفائقة التي شرحناها موجودة هنا حرفياً: "
        "η (eta0)، الحقب (max_iter)، عتبة التوقّف (tol)، والتوقّف المبكّر."
    )

rule()

# ======================================================================
# 8. مراقبة التدريب
# ======================================================================
st.header(":material/monitor_heart: كيف نعرف أن التدريب يسير جيداً؟ منحنى الخسارة")

st.markdown(
    """
**منحنى الخسارة (Loss curve)** هو لوحة قيادة التدريب: محور أفقي = الحقب،
محور رأسي = قيمة التكلفة. أربعة أشكال نمطية ومعنى كل واحد:
    """
)

ep = np.arange(1, 61)
curves = {
    "سليم — تقارب جيد": (2.4 * np.exp(-0.09 * ep) + 0.18, viz.C_GREEN,
                          "ينزل بسرعة ثم يستقر. هذا ما نريده."),
    "معدّل التعلّم صغير جداً": (2.4 * np.exp(-0.012 * ep) + 0.18, viz.C_BLUE,
                                "ينزل ببطء شديد: ستحتاج آلاف الحقب. ارفع η."),
    "معدّل التعلّم كبير جداً": (0.5 + 1.9 * np.abs(np.sin(0.9 * ep)) * np.exp(0.012 * ep),
                                viz.C_ROSE, "يتذبذب أو يرتفع: خفّض η فوراً."),
    "عالق في هضبة": (np.concatenate([2.4 * np.exp(-0.25 * np.arange(12)) + 0.9,
                                      np.full(48, 0.92)]), viz.C_AMBER,
                      "توقّف عن التحسّن مبكراً: جرّب معدّل تعلّم متغيّراً أو نموذجاً أقوى."),
}

fig2 = go.Figure()
for name, (vals, col, _) in curves.items():
    fig2.add_trace(
        go.Scatter(
            x=ep,
            y=vals,
            mode="lines",
            name=name,
            line=dict(color=col, width=3.5),
            hovertemplate="epoch %{x}<br>J=%{y:.3f}<extra></extra>",
        )
    )
viz.style_fig(
    fig2,
    height=400,
    xtitle="الحقبة (Epoch)",
    ytitle="دالة التكلفة J",
    title="أشكال منحنى الخسارة وما تعنيه",
)
st.plotly_chart(fig2, width="stretch", key="tr_curves")

st.markdown(
    """
| شكل المنحنى | التشخيص | العلاج |
|---|---|---|
| ينزل ثم يستقر | تدريب سليم | لا شيء، تابع |
| ينزل ببطء شديد | $\\eta$ صغير جداً | ارفع $\\eta$ (×3 أو ×10) |
| يتذبذب أو يرتفع | $\\eta$ كبير جداً | اخفض $\\eta$ (÷3 أو ÷10) |
| يستقر عند قيمة عالية | نموذج ضعيف / تقصير | نموذج أقوى، خصائص أفضل |
| خسارة التدريب تنزل وخسارة التحقق ترتفع | **إفراط في التعلّم** | توقّف مبكّر، تنظيم، بيانات أكثر |
| `NaN` أو `inf` | انفجار التدرّج | اخفض $\\eta$، قيّس الخصائص، قصّ التدرّج |
    """
)

verdict(
    "القاعدة الأهم في مراقبة التدريب",
    "ارسم <b>منحنيين دائماً</b>: خسارة التدريب وخسارة التحقق على الرسم نفسه. "
    "المنحنى الأول وحده يخدعك — فنموذج يحفظ البيانات حفظاً سيعطيك خسارة تدريب "
    "قريبة من الصفر وهو في الحقيقة عديم الفائدة. "
    "سنخصّص لهذا درساً كاملاً في «الإفراط والتقصير».",
    tone="rose",
)

rule()

# ======================================================================
# 9. أخطاء شائعة
# ======================================================================
st.header(":material/report: مفاهيم خاطئة شائعة عند المبتدئين")

misc = pd.DataFrame(
    {
        "الاعتقاد الخاطئ": [
            "«النموذج يفهم البيانات»",
            "«كلما زادت الحقب كان أفضل»",
            "«خسارة تدريب صفر = نموذج ممتاز»",
            "«التدريب يغيّر شكل النموذج»",
            "«معدّل التعلّم يُضبط مرة ويُنسى»",
            "«يكفي مقياس واحد للأداء»",
            "«الخوارزمية الأقوى تحل كل شيء»",
        ],
        "الصواب": [
            "النموذج يضبط أرقاماً لتصغير خطأ عددي، لا أكثر.",
            "بعد نقطة معيّنة تبدأ خسارة التحقق في الارتفاع ← إفراط في التعلّم.",
            "غالباً يعني حفظاً للبيانات (Memorisation) لا تعلّماً.",
            "التدريب يضبط المعاملات فقط؛ الشكل تختاره أنت مسبقاً.",
            "هو من أهم المعاملات الفائقة ويُضبط تجريبياً على بيانات التحقق.",
            "اختر المقياس حسب المشكلة؛ الدقة وحدها خادعة في البيانات غير المتوازنة.",
            "بيانات أنظف وخصائص أذكى تتفوّق عادة على خوارزمية أعقد.",
        ],
    }
)
st.dataframe(misc, width="stretch", hide_index=True)

footer("الدرس التالي: دالة الخسارة — كيف نختار «تعريف الخطأ» المناسب لمشكلتنا؟")

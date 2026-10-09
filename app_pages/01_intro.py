"""الدرس 0 — ما هو تعلّم الآلة؟ وما معنى أن «نُدرّب» نموذجاً؟"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.data import toy_table
from utils.style import card, equation, footer, hero, rule, steps, terms, verdict

hero(
    "البداية — ما هو تعلّم الآلة؟ وما معنى «التدريب»؟",
    "Lesson 0 · What is machine learning and what is training?",
    "قبل أي خوارزمية، وقبل أي كود: ما الذي يحدث فعلاً عندما نقول إن الآلة «تتعلّم»؟ "
    "هذا الدرس يبني لك الصورة الكاملة من الصفر، بالكلمات، بالرسم، وبالصيغة الرياضية.",
)

# ======================================================================
# 1. البرمجة التقليدية مقابل تعلّم الآلة
# ======================================================================
st.header(":material/compare_arrows: الفرق الجوهري: البرمجة التقليدية مقابل تعلّم الآلة")

st.markdown(
    """
تخيّل أنك تريد برنامجاً يعرف **سعر منزل** انطلاقاً من مساحته.

في **البرمجة التقليدية (Classical Programming)** أنت من يكتب القاعدة:
«السعر = 0.6 × المساحة + 20». أنت المهندس، وأنت من يعرف القاعدة مسبقاً.

في **تعلّم الآلة (Machine Learning)** الأمر معكوس تماماً: أنت لا تعرف القاعدة.
أنت تملك فقط **أمثلة سابقة** (منازل بيعت فعلاً: مساحتها وسعرها)، وتطلب من
الحاسوب أن **يستنتج القاعدة بنفسه** من هذه الأمثلة. هذا الاستنتاج هو ما نسميه
**التدريب (Training)**.
    """
)

c1, c2 = st.columns(2, gap="medium")

with c1:
    st.markdown("##### :material/code: البرمجة التقليدية")
    st.mermaid_chart(
        """
graph TD
    D["البيانات<br/>Data"] --> P["البرنامج<br/>Rules written by you"]
    R["القاعدة<br/>Rule"] --> P
    P --> O["النتائج<br/>Output"]
    style D fill:#E3F0FE,stroke:#3B7DD8,stroke-width:2px,color:#17263F
    style R fill:#FDF0DC,stroke:#F0A33C,stroke-width:2px,color:#17263F
    style P fill:#FFFFFF,stroke:#8A94A6,stroke-width:2px,color:#17263F
    style O fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
        """
    )
    st.caption("أنت تُدخل القاعدة، والحاسوب يُخرج النتيجة.")

with c2:
    st.markdown("##### :material/psychology: تعلّم الآلة")
    st.mermaid_chart(
        """
graph TD
    D["البيانات<br/>Data (X)"] --> T["التدريب<br/>Training"]
    O["النتائج المعروفة<br/>Labels (y)"] --> T
    T --> R["القاعدة المُكتشفة<br/>Model / Rule"]
    style D fill:#E3F0FE,stroke:#3B7DD8,stroke-width:2px,color:#17263F
    style O fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
    style T fill:#EEE6FB,stroke:#9B6BD6,stroke-width:3px,color:#17263F
    style R fill:#FDE6EE,stroke:#EB6F92,stroke-width:2px,color:#17263F
        """
    )
    st.caption("أنت تُدخل البيانات والنتائج، والحاسوب يُخرج القاعدة.")

verdict(
    "الخلاصة في سطر واحد",
    "تعلّم الآلة = <b>عملية عكسية</b>. بدلاً من أن نكتب القاعدة لنحصل على النتائج، "
    "نُعطي النتائج لنحصل على القاعدة. و«التدريب» هو اسم هذه العملية العكسية.",
    tone="violet",
)

rule()

# ======================================================================
# 2. التعريف الأكاديمي
# ======================================================================
st.header(":material/menu_book: التعريف الأكاديمي الدقيق")

st.markdown(
    """
التعريف الأكثر شيوعاً في الكتب هو تعريف **توم ميتشل (Tom Mitchell, 1997)**،
وهو تعريف مهم لأنه يحوّل كلمة «يتعلّم» الغامضة إلى **ثلاثة أشياء قابلة للقياس**:
    """
)

equation(
    r"\text{A program learns from } \mathbf{E} \text{ w.r.t. task } \mathbf{T} "
    r"\text{ and measure } \mathbf{P} \text{ if its } \mathbf{P} \text{ on } "
    r"\mathbf{T} \text{ improves with } \mathbf{E}",
    "يُقال إن برنامجاً يتعلّم من <b>الخبرة E</b> في أداء <b>مهمة T</b> بالنسبة "
    "لـ<b>مقياس أداء P</b>، إذا كان أداؤه P على المهمة T <b>يتحسّن</b> مع تزايد الخبرة E.",
)

t1, t2, t3 = st.columns(3, gap="medium")
with t1:
    card(
        "E — الخبرة",
        "هي <b>البيانات</b> التي رآها النموذج. كل صف في جدولك هو «تجربة» واحدة. "
        "كلما زادت الأمثلة (مع بقائها نظيفة وممثّلة) زادت الخبرة.",
        accent="blue",
        eng="E = Experience (the training data)",
    )
with t2:
    card(
        "T — المهمة",
        "ما نريد من النموذج أن يفعله: <b>توقّع رقم</b> (سعر، درجة حرارة) أو "
        "<b>تصنيف</b> (مريض/سليم، بريد مزعج/عادي) أو <b>تجميع</b> أو <b>ترتيب</b>.",
        accent="green",
        eng="T = Task (regression, classification, clustering…)",
    )
with t3:
    card(
        "P — مقياس الأداء",
        "رقم واحد يقول لنا: هل نحن نتحسّن أم لا؟ مثل <b>متوسط مربع الخطأ</b> أو "
        "<b>الدقة</b>. بدون P لا يوجد «تعلّم»، فقط تخمين عشوائي.",
        accent="amber",
        eng="P = Performance measure (loss / metric)",
    )

verdict(
    "نقطة يغفل عنها المبتدئون",
    "كلمة «التدريب» لا تعني أن الحاسوب «يفهم». التدريب هو حرفياً: "
    "<b>تعديل أرقام داخلية (المعاملات) خطوة بخطوة حتى يصغر رقم الخطأ</b>. "
    "لا شيء أكثر من ذلك، ولا شيء سحري.",
    tone="amber",
)

rule()

# ======================================================================
# 3. أنواع التعلّم
# ======================================================================
st.header(":material/category: أنواع التعلّم — أين نحن بالضبط؟")

st.markdown(
    "كل ما سنشرحه في هذه الدورة (التدريب، تقسيم البيانات، الإفراط في التعلّم) "
    "ينطبق خصوصاً على **التعلّم المُراقَب**، وهو 90٪ من التطبيقات العملية. "
    "لكن من المفيد أن ترى الخريطة كاملة:"
)

k1, k2, k3 = st.columns(3, gap="medium")
with k1:
    card(
        "التعلّم المُراقَب",
        "عندنا <b>X</b> (المدخلات) و<b>y</b> (الجواب الصحيح). "
        "النموذج يتعلّم العلاقة بينهما.<br>"
        "• <b>انحدار</b>: y رقم مستمر (سعر).<br>"
        "• <b>تصنيف</b>: y فئة (نعم/لا).",
        accent="blue",
        eng="Supervised Learning: Regression & Classification",
    )
with k2:
    card(
        "التعلّم غير المُراقَب",
        "عندنا <b>X</b> فقط، ولا يوجد جواب صحيح. النموذج يبحث عن <b>بنية مخفية</b>: "
        "مجموعات متشابهة (Clustering) أو تقليل الأبعاد (PCA).",
        accent="violet",
        eng="Unsupervised Learning: Clustering, Dimensionality Reduction",
    )
with k3:
    card(
        "التعلّم المُعزَّز",
        "لا يوجد جدول بيانات أصلاً. يوجد <b>وكيل</b> يتفاعل مع <b>بيئة</b>، "
        "يتخذ أفعالاً، ويتلقّى <b>مكافأة أو عقوبة</b>، فيتعلّم السياسة الأفضل.",
        accent="rose",
        eng="Reinforcement Learning: Agent, Environment, Reward",
    )

rule()

# ======================================================================
# 4. ما هو «مجموعة بيانات»؟
# ======================================================================
st.header(":material/table_chart: ما هي «مجموعة البيانات»؟ لغة الرموز")

st.markdown(
    """
قبل أي رياضيات، يجب أن نتفق على أسماء الأشياء. مجموعة البيانات ليست أكثر من
**جدول**: الصفوف أمثلة، والأعمدة صفات.
    """
)

tb = toy_table(10)
d1, d2 = st.columns([1.25, 1], gap="medium")
with d1:
    st.dataframe(
        tb,
        width="stretch",
        hide_index=False,
        column_config={
            "area_m2": st.column_config.NumberColumn("المساحة (م²)", format="%d"),
            "rooms": st.column_config.NumberColumn("الغرف", format="%d"),
            "age_years": st.column_config.NumberColumn("العمر (سنة)", format="%d"),
            "price_k": st.column_config.NumberColumn("السعر (ألف) ← y", format="%.1f"),
        },
    )
    st.caption(
        "ثلاثة أعمدة أولى = الخصائص (Features) = المصفوفة X · "
        "العمود الأخير = الهدف (Target) = المتجه y"
    )

with d2:
    terms(
        [
            ("Sample / Instance", "مثال / مشاهدة (صف)"),
            ("Feature / Predictor", "خصيصة / متغيّر مُفسِّر (عمود)"),
            ("Target / Label", "الهدف / التسمية"),
            ("Design matrix X", "مصفوفة التصميم"),
            ("n", "عدد الأمثلة"),
            ("p", "عدد الخصائص"),
        ]
    )

equation(
    r"\mathbf{X}=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p}\\"
    r" x_{21} & x_{22} & \cdots & x_{2p}\\ \vdots & \vdots & \ddots & \vdots\\"
    r" x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix}_{n\times p},\qquad"
    r"\mathbf{y}=\begin{bmatrix} y_1\\ y_2\\ \vdots\\ y_n\end{bmatrix}_{n\times 1}",
    "المصفوفة <b>X</b> فيها <b>n</b> صفاً (عدد الأمثلة) و<b>p</b> عموداً (عدد الخصائص). "
    "المتجه <b>y</b> فيه الجواب الصحيح لكل مثال. "
    "الرمز <b>x<sub>ij</sub></b> يعني: قيمة الخصيصة رقم j عند المثال رقم i.",
)

st.markdown(
    """
وأي مثال واحد (صف واحد) نكتبه كمتجه:
    """
)
equation(
    r"\mathbf{x}_i=(x_{i1},x_{i2},\dots,x_{ip})^{\top}\in\mathbb{R}^{p}, \qquad y_i\in\mathbb{R}"
    r"\quad\text{(انحدار)}\quad\text{أو}\quad y_i\in\{0,1\}\quad\text{(تصنيف ثنائي)}",
    "لاحظ الفرق: في الانحدار y رقم حقيقي أي قيمة؛ في التصنيف الثنائي y إما 0 أو 1.",
)

rule()

# ======================================================================
# 5. أول تجربة تفاعلية: درّب النموذج بيدك
# ======================================================================
st.header(":material/touch_app: جرّب بنفسك: كن أنت «المُدرِّب»")

st.markdown(
    """
هذه أهم لحظة في الدرس. أمامك نقاط حقيقية (كل نقطة منزل: المساحة ↔ السعر).
نريد خطاً مستقيماً يمرّ قريباً من كل النقاط. معادلة الخط:

$\\hat{y} = w \\cdot x + b$

- $w$ (**Weight / Slope**) = ميل الخط: كم يزيد السعر لكل متر مربع إضافي.
- $b$ (**Bias / Intercept**) = نقطة التقاطع: السعر عند المساحة صفر.

حرّك المؤشرين، وشاهد **الخطوط الرأسية الحمراء** — تلك هي **البواقي (Residuals)**،
أي خطأ النموذج في كل نقطة. مهمتك: اجعل **رقم الخطأ أصغر ما يمكن**.
    """
)

rng = np.random.default_rng(42)
x_demo = np.linspace(1.0, 9.0, 26)
y_demo = 4.5 + 2.2 * x_demo + rng.normal(0, 2.4, x_demo.size)

sc1, sc2, sc3 = st.columns([1, 1, 1.1], gap="medium")
with sc1:
    w = st.slider("الميل w (Weight)", -1.0, 5.0, 0.5, 0.05, key="intro_w")
with sc2:
    b = st.slider("التقاطع b (Bias)", -10.0, 20.0, 0.0, 0.25, key="intro_b")
with sc3:
    show_res = st.toggle("أظهر البواقي (Residuals)", value=True, key="intro_res")

pred = w * x_demo + b
resid = y_demo - pred
mse = float(np.mean(resid**2))
mae = float(np.mean(np.abs(resid)))

# best possible fit (ordinary least squares) for comparison
A = np.vstack([x_demo, np.ones_like(x_demo)]).T
w_best, b_best = np.linalg.lstsq(A, y_demo, rcond=None)[0]
mse_best = float(np.mean((y_demo - (w_best * x_demo + b_best)) ** 2))

fig = go.Figure()

if show_res:
    for xi, yi, pi in zip(x_demo, y_demo, pred):
        fig.add_trace(
            go.Scatter(
                x=[xi, xi],
                y=[yi, pi],
                mode="lines",
                line=dict(color=viz.rgba(viz.C_ROSE, 0.75), width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

fig.add_trace(
    go.Scatter(
        x=x_demo,
        y=y_demo,
        mode="markers",
        name="بيانات حقيقية (y)",
        marker=dict(
            size=11,
            color=viz.rgba(viz.C_BLUE, 0.85),
            line=dict(color="#FFFFFF", width=2),
        ),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_demo,
        y=w_best * x_demo + b_best,
        mode="lines",
        name="أفضل خط ممكن (OLS)",
        line=dict(color=viz.rgba(viz.C_GREEN, 0.55), width=3, dash="dash"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_demo,
        y=pred,
        mode="lines",
        name="خطّك أنت (ŷ = w·x + b)",
        line=dict(color=viz.C_VIOLET, width=4),
        hovertemplate="ŷ=%{y:.2f}<extra></extra>",
    )
)

viz.style_fig(
    fig,
    height=440,
    xtitle="x — المساحة (وحدة مُعيّرة)",
    ytitle="y — السعر",
    title="اضبط الخط يدوياً وراقب الخطأ",
)
st.plotly_chart(fig, width="stretch", key="intro_fit")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MSE — متوسط مربع الخطأ", f"{mse:,.2f}", delta=f"{mse - mse_best:+,.2f} عن الأفضل",
          delta_color="inverse")
m2.metric("MAE — متوسط الخطأ المطلق", f"{mae:,.2f}")
m3.metric("أفضل MSE ممكن", f"{mse_best:,.2f}")
m4.metric("الحل الرياضي الأمثل", f"w={w_best:.2f} · b={b_best:.2f}")

if mse <= mse_best * 1.03:
    st.success(
        "ممتاز! وصلت تقريباً إلى أفضل خط ممكن. "
        "ما فعلته يدوياً الآن هو بالضبط ما يفعله **التدريب** تلقائياً — "
        "لكنه يفعله بالحساب لا بالتخمين."
    )
elif mse <= mse_best * 1.5:
    st.info("قريب جداً. استمر في تحريك المؤشرين لتصغير MSE أكثر.")
else:
    st.warning(
        "الخطأ لا يزال كبيراً. لاحظ أنك تبحث يدوياً في فضاء المعاملات — "
        "والنموذج سيفعل هذا البحث رياضياً وبسرعة هائلة."
    )

verdict(
    "ماذا تعلّمنا من هذه التجربة؟",
    "التدريب هو <b>بحث منظّم عن أفضل قيم للمعاملات</b> (w و b هنا) بحيث يصغر مقياس الخطأ. "
    "أنت بحثت بالتخمين؛ الخوارزمية تبحث بالتدرّج (Gradient). "
    "هذا كل الفرق — والفكرة نفسها تماماً تصلح لنموذج فيه مليار معامل.",
    tone="green",
)

rule()

# ======================================================================
# 6. خريطة الدورة
# ======================================================================
st.header(":material/map: خريطة الدورة — ماذا سنتعلّم بالترتيب؟")

st.markdown(
    """
الدورة مبنيّة على **أربعة أجزاء**، والترتيب مقصود: نبدأ بالفكرة العامة بلغة
بسيطة تماماً، ثم ننزل إلى التفاصيل الرياضية، ثم إلى الممارسة.
    """
)

st.markdown("##### :material/looks_one: الجزء الأول — التدريب بصفة عامة (بلا رياضيات)")
steps(
    [
        (
            "التدريب ببساطة شديدة (الدرس 1)",
            "بلا معادلة واحدة: ما معنى «يتدرّب»، صورة «الآلة ذات المقابض»، "
            "مختبر تُدير فيه المقابض بيدك، ثم آلة تُصحّح نفسها أمامك خطوة بخطوة، "
            "وأسئلة المبتدئين كلها.",
        ),
        (
            "آليات التدريب عبر الخوارزميات (الدرس 2)",
            "هل تتدرّب كل الخوارزميات بالطريقة نفسها؟ <b>لا.</b> "
            "حل مغلق، تحسين تكراري، بناء جشع للأشجار، تخزين فقط في KNN، "
            "عدّ في Naive Bayes، تعزيز تتابعي — وتصنيفات التدريب "
            "(دفعي/متصل، من الصفر/بالنقل، أنواع الإشراف).",
        ),
        (
            "دورة حياة التدريب (الدرس 3)",
            "ما قبل التدريب وما بعده: التهيئة، المراقبة، نقاط الحفظ، معايير التوقّف، "
            "العشوائية والتكرارية، حاسبة الوقت والذاكرة، جدول استكشاف الأخطاء، "
            "وحفظ النموذج وإعادة تدريبه.",
        ),
    ],
    accent="blue",
)

st.markdown("##### :material/looks_two: الجزء الثاني — التدريب بالتفصيل الرياضي")
steps(
    [
        (
            "التشريح الرياضي للتدريب (الدرس 4)",
            "النموذج، المعاملات، دالة الخسارة، دالة التكلفة، التحسين، "
            "Epoch و Iteration و Batch، والخوارزمية كاملة بالكود.",
        ),
        (
            "دالة الخسارة (الدرس 5)",
            "MSE و MAE و RMSE و Huber و Cross-Entropy: متى نستخدم كل واحدة، "
            "ولماذا يتغيّر النموذج بتغيّرها.",
        ),
        (
            "النزول التدرّجي (الدرس 6)",
            "كيف يجد الحاسوب أقل نقطة؟ معدّل التعلّم، التقارب، التباعد، "
            "أثر التقييس، والمحسّنات الحديثة — مع مختبر متحرّك.",
        ),
        (
            "حلقة التدريب الحيّة (الدرس 7)",
            "نُشغّل تدريباً حقيقياً أمام عينيك: Epoch بعد Epoch، "
            "الخط يتحسّن ومنحنى الخسارة ينزل ومسار المعاملات يتحرّك.",
        ),
    ],
    accent="violet",
)

st.markdown("##### :material/looks_3: الجزء الثالث — تقسيم البيانات")
steps(
    [
        (
            "التدريب / التحقق / الاختبار (الدرس 8)",
            "قاعدة 80-20 وكل القواعد الأخرى، الحساب الإحصائي لحجم مجموعة الاختبار، "
            "التقسيم الطبقي والزمني وبالمجموعات، وتسرّب البيانات.",
        ),
        (
            "التحقق المتقاطع (الدرس 9)",
            "K-Fold و Stratified و ShuffleSplit و Group و TimeSeries و LOOCV "
            "و Nested CV — بالحركة.",
        ),
    ],
    accent="amber",
)

st.markdown("##### :material/looks_4: الجزء الرابع — التشخيص والتقييم")
steps(
    [
        (
            "الإفراط والتقصير (الدرس 10)",
            "Overfitting و Underfitting، المقايضة بين التحيّز والتباين، "
            "منحنيات التعلّم والتحقق، التنظيم، والتوقّف المبكّر.",
        ),
        (
            "معايير التقييم (الدرس 11)",
            "مقاييس الانحدار والتصنيف، مصفوفة الالتباس التفاعلية، ROC-AUC، "
            "وفخّ الدقة في البيانات غير المتوازنة.",
        ),
    ],
    accent="green",
)

verdict(
    "نصيحة في الترتيب",
    "إن كنت مبتدئاً تماماً، <b>لا تقفز إلى الجزء الثاني</b>. "
    "الدروس الثلاثة الأولى بلا رياضيات تقريباً، وهي التي ستجعل المعادلات "
    "لاحقاً تبدو بديهية بدل أن تبدو طلاسم.",
    tone="blue",
)

footer("ابدأ الآن بالدرس الأول: «التدريب ببساطة» من القائمة على اليمين :material/arrow_forward:")

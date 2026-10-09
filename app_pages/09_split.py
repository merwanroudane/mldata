"""الدرس 8 — تقسيم البيانات: التدريب والتحقق والاختبار."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import (
    TEST_C,
    TRAIN_C,
    VAL_C,
    card,
    equation,
    footer,
    hero,
    ribbon,
    rule,
    steps,
    terms,
    verdict,
)

hero(
    "تقسيم البيانات — التدريب والتحقق والاختبار",
    "Lesson 8 · Train / Validation / Test splitting",
    "لماذا نقسّم أصلاً؟ ما القواعد؟ هل 80-20 قانون مقدّس؟ وما هي الأخطاء التي "
    "تجعل نتائجك ممتازة على الورق وكارثية في الواقع؟ كل شيء هنا.",
)

terms(
    [
        ("Training set", "مجموعة التدريب"),
        ("Validation set", "مجموعة التحقق"),
        ("Test set", "مجموعة الاختبار"),
        ("Hold-out", "عيّنة محجوزة"),
        ("Generalization", "التعميم"),
        ("Stratified split", "تقسيم طبقي"),
        ("Data leakage", "تسرّب البيانات"),
        ("Shuffle", "خلط"),
    ]
)

# ======================================================================
# 1. لماذا نقسّم؟
# ======================================================================
st.header(":material/help_center: أولاً: لماذا نقسّم البيانات أصلاً؟")

st.markdown(
    """
تخيّل أستاذاً أعطى طلابه **عشرة تمارين مع حلولها**، ثم جاء الامتحان
وكانت أسئلته **نفس التمارين العشرة حرفياً**. الطالب الذي حفظ الحلول
سيأخذ 20/20 — لكن هل نقول إنه **فهم** المادة؟ بالطبع لا.

النموذج يفعل الشيء نفسه بالضبط. إن قيّمناه على **البيانات نفسها** التي
تدرّب عليها، فنحن نمتحنه في الأسئلة التي رآها مع الحلول. النتيجة ستكون
**ممتازة ومُضلِّلة تماماً**.
    """
)

st.markdown(
    """
> :material/star: **الهدف الحقيقي لتعلّم الآلة ليس الأداء على البيانات الموجودة،
> بل الأداء على بيانات لم يرها النموذج قط.** هذا اسمه **التعميم (Generalization)**،
> وهو السبب الوحيد لوجود التقسيم.
    """
)

a1, a2, a3 = st.columns(3, gap="medium")
with a1:
    card(
        "التدريب = التمارين المحلولة",
        "النموذج يرى المدخلات <b>والإجابات</b>، ويضبط معاملاته عليها. "
        "هذه هي البيانات الوحيدة التي تؤثّر مباشرة في قيم w و b.",
        accent="blue",
        eng="Training set — the model fits its parameters here",
    )
with a2:
    card(
        "التحقق = الامتحان التجريبي",
        "نقيس عليها الأداء <b>أثناء</b> التطوير لنختار المعاملات الفائقة "
        "(η، عمق الشجرة، λ…) ونقرّر متى نتوقّف. النموذج لا يتدرّب عليها، "
        "لكنك <b>تتخذ قرارات</b> بناءً عليها، فتتلوّث تدريجياً.",
        accent="amber",
        eng="Validation set — model selection & hyperparameter tuning",
    )
with a3:
    card(
        "الاختبار = الامتحان الرسمي",
        "تُفتح <b>مرة واحدة فقط</b>، في النهاية، بعد تجميد كل القرارات. "
        "هي تقديرك النزيه لأداء النموذج في الواقع.",
        accent="rose",
        eng="Test set — final unbiased estimate, used ONCE",
    )

st.mermaid_chart(
    """
graph TD
    ALL["كل البيانات<br/>Full dataset (n)"] --> TR["تدريب · Train<br/>ضبط المعاملات θ"]
    ALL --> VA["تحقق · Validation<br/>اختيار المعاملات الفائقة"]
    ALL --> TE["اختبار · Test<br/>التقدير النهائي"]
    TR --> M["نموذج مُدرَّب"]
    VA --> M2["أفضل إعدادات"]
    M --> F["النموذج النهائي"]
    M2 --> F
    F --> TE
    TE --> R["الرقم الذي تكتبه في التقرير"]

    style ALL fill:#EDF1F6,stroke:#8A94A6,stroke-width:2px,color:#17263F
    style TR fill:#E3F0FE,stroke:#3B7DD8,stroke-width:3px,color:#17263F
    style VA fill:#FDF0DC,stroke:#F0A33C,stroke-width:3px,color:#17263F
    style TE fill:#FDE6EE,stroke:#EB6F92,stroke-width:3px,color:#17263F
    style M fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style M2 fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style F fill:#EEE6FB,stroke:#9B6BD6,stroke-width:2px,color:#17263F
    style R fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
    """
)

verdict(
    "القاعدة الذهبية التي لا تُكسر أبداً",
    "<b>مجموعة الاختبار تُلمس مرة واحدة فقط، في النهاية.</b> "
    "إن نظرت إليها، ثم عدّلت النموذج، ثم نظرت إليها ثانية — فقد تحوّلت عملياً "
    "إلى مجموعة تحقق، وفقدت قيمتها كتقدير نزيه. "
    "كل قرار تتخذه بعد رؤية نتيجة الاختبار هو <b>تسرّب للمعلومة</b>.",
    tone="rose",
)

rule()

# ======================================================================
# 2. أدوار المجموعات الثلاث
# ======================================================================
st.header(":material/table: جدول الأدوار الثلاثة — احفظه")

st.markdown(
    """
| | **التدريب (Train)** | **التحقق (Validation)** | **الاختبار (Test)** |
|---|---|---|---|
| **هل يرى النموذج الإجابات؟** | نعم | لا (لكن نحن نراها) | لا |
| **هل تتغيّر المعاملات θ بسببها؟** | نعم مباشرة | لا | لا |
| **هل تتغيّر المعاملات الفائقة بسببها؟** | لا | **نعم** | لا |
| **كم مرة تُستعمل؟** | آلاف المرات | عشرات/مئات المرات | **مرة واحدة** |
| **ماذا تُستعمل له؟** | ضبط $\\theta$ | ضبط $\\eta,\\lambda,k$، التوقّف المبكّر، اختيار النموذج | التقرير النهائي |
| **هل تقديرها نزيه؟** | لا (متفائل جداً) | لا (متفائل قليلاً) | **نعم** |
| **النسبة النموذجية** | 60-80٪ | 10-20٪ | 10-20٪ |

:material/priority_high: **لماذا تكون نتيجة التحقق متفائلة قليلاً؟** لأنك جرّبت عليها
عشرات الإعدادات واخترت الأفضل. اختيار الأفضل من بين كثير على عيّنة واحدة يُنتج
**انحيازاً تفاؤلياً (Selection bias)** — ولهذا نحتفظ بمجموعة اختبار منفصلة.
    """
)

rule()

# ======================================================================
# 3. مُقسِّم تفاعلي
# ======================================================================
st.header(":material/call_split: مُقسِّم تفاعلي — جرّب النسب وشاهد النتيجة")

p1, p2 = st.columns([1.1, 1], gap="medium")
with p1:
    n_total = st.number_input(
        "حجم بياناتك الكلي n", min_value=20, max_value=10_000_000, value=1_000, step=50
    )
with p2:
    preset = st.selectbox(
        "قوالب جاهزة",
        [
            "مخصّص (حرّك المؤشرات)",
            "80 / 20  — تدريب واختبار فقط",
            "70 / 30  — بيانات صغيرة",
            "60 / 20 / 20  — كلاسيكي ثلاثي",
            "70 / 15 / 15  — شائع جداً",
            "80 / 10 / 10  — بيانات متوسّطة",
            "98 / 1 / 1  — بيانات ضخمة (> مليون)",
            "50 / 25 / 25  — مقارنة نماذج كثيرة",
        ],
        index=4,
    )

PRESETS = {
    "80 / 20  — تدريب واختبار فقط": (80, 0, 20),
    "70 / 30  — بيانات صغيرة": (70, 0, 30),
    "60 / 20 / 20  — كلاسيكي ثلاثي": (60, 20, 20),
    "70 / 15 / 15  — شائع جداً": (70, 15, 15),
    "80 / 10 / 10  — بيانات متوسّطة": (80, 10, 10),
    "98 / 1 / 1  — بيانات ضخمة (> مليون)": (98, 1, 1),
    "50 / 25 / 25  — مقارنة نماذج كثيرة": (50, 25, 25),
}

if preset in PRESETS:
    tr_pct, va_pct, te_pct = PRESETS[preset]
    st.caption(
        f"القالب المختار: **{tr_pct}٪ تدريب · {va_pct}٪ تحقق · {te_pct}٪ اختبار**. "
        "اختر «مخصّص» من القائمة لتحريك النسب بنفسك."
    )
else:
    cc1, cc2 = st.columns(2, gap="medium")
    with cc1:
        tr_pct = st.slider("نسبة التدريب ٪", 10, 98, 70, 1, key="sp_tr2")
    with cc2:
        va_pct = st.slider("نسبة التحقق ٪", 0, 45, 15, 1, key="sp_va2")
    te_pct = 100 - tr_pct - va_pct
    if te_pct < 0:
        st.error(
            f"المجموع تجاوز 100٪ ({tr_pct} + {va_pct} = {tr_pct + va_pct}). "
            "خفّض إحدى النسبتين."
        )
        te_pct = 0

n_tr = int(round(n_total * tr_pct / 100))
n_va = int(round(n_total * va_pct / 100))
n_te = max(0, n_total - n_tr - n_va)

ribbon(
    [
        (f"{tr_pct}٪", f"تدريب · {n_tr:,}", tr_pct, TRAIN_C),
        (f"{va_pct}٪", f"تحقق · {n_va:,}", va_pct, VAL_C),
        (f"{te_pct}٪", f"اختبار · {n_te:,}", te_pct, TEST_C),
    ]
)

s1, s2, s3, s4 = st.columns(4)
s1.metric("أمثلة التدريب", f"{n_tr:,}")
s2.metric("أمثلة التحقق", f"{n_va:,}")
s3.metric("أمثلة الاختبار", f"{n_te:,}")
s4.metric("المجموع", f"{n_tr + n_va + n_te:,}")

warn = []
if n_te and n_te < 30:
    warn.append(
        f"مجموعة الاختبار فيها **{n_te}** مثالاً فقط — تقديرك سيكون شديد الاهتزاز. "
        "استعمل **التحقق المتقاطع** بدل عيّنة محجوزة."
    )
if n_va and n_va < 30:
    warn.append(
        f"مجموعة التحقق فيها **{n_va}** مثالاً فقط — اختيار المعاملات الفائقة عليها "
        "سيكون عشوائياً تقريباً."
    )
if n_tr < 50:
    warn.append(
        f"مجموعة التدريب فيها **{n_tr}** مثالاً فقط — لن يتعلّم النموذج شيئاً ذا قيمة."
    )
if va_pct == 0 and te_pct > 0:
    warn.append(
        "لا توجد مجموعة تحقق. هذا مقبول **فقط** إن كنت لن تضبط أي معامل فائق، "
        "أو إن كنت ستستعمل التحقق المتقاطع داخل بيانات التدريب."
    )
if n_total >= 500_000 and te_pct >= 20:
    warn.append(
        f"بياناتك ضخمة ({n_total:,}). حجز {te_pct}٪ للاختبار يعني "
        f"{n_te:,} مثالاً — هذا إهدار. {max(1, int(50_000 / n_total * 100))}٪ أو أقل تكفي تماماً."
    )

for wtxt in warn:
    st.warning(wtxt)
if not warn:
    st.success("نسب معقولة لحجم بياناتك. :material/check_circle:")

rule()

# ======================================================================
# 4. القواعد: ما الذي يحدّد النسبة؟
# ======================================================================
st.header(":material/rule: هل 80-20 قاعدة مقدّسة؟ لا. إليك القواعد الحقيقية")

st.markdown(
    """
**80-20 ليست قانوناً**، بل عادة قديمة ورثناها من زمن كانت فيه البيانات بالمئات.
النسبة الصحيحة تُشتقّ من سؤال واحد:

> **كم مثالاً أحتاج في مجموعة الاختبار حتى يكون تقديري دقيقاً بما يكفي؟**

وهذا سؤال **إحصائي** له جواب رقمي، لا سؤال ذوقي.
    """
)

st.subheader("الجدول العملي حسب حجم البيانات")

rules = pd.DataFrame(
    {
        "حجم البيانات n": [
            "أقل من 100",
            "100 – 1,000",
            "1,000 – 10,000",
            "10,000 – 100,000",
            "100,000 – 1,000,000",
            "أكثر من 1,000,000",
        ],
        "التقسيم المقترح": [
            "لا تقسّم: استعمل LOOCV أو 5-Fold CV متكرّر",
            "70 / 15 / 15  أو  5-Fold CV + اختبار 20٪",
            "70 / 15 / 15  أو  60 / 20 / 20",
            "80 / 10 / 10",
            "90 / 5 / 5",
            "98 / 1 / 1  (أو حتى 99.8 / 0.1 / 0.1)",
        ],
        "السبب": [
            "كل مثال ثمين؛ العيّنة المحجوزة ستكون عشوائية تماماً",
            "نحتاج توازناً بين تدريب كافٍ وتقدير مقبول",
            "النسب الكلاسيكية تعمل جيداً هنا",
            "10٪ = آلاف الأمثلة، وهذا يكفي لتقدير دقيق",
            "5٪ من 500 ألف = 25 ألف مثال، أكثر من كافٍ",
            "1٪ من مليون = 10 آلاف مثال؛ الباقي يذهب للتدريب",
        ],
    }
)
st.dataframe(rules, width="stretch", hide_index=True)

verdict(
    "المبدأ الذي يولّد كل هذه الأرقام",
    "مجموعة الاختبار لا تحتاج <b>نسبة</b> معيّنة، بل <b>عدداً مطلقاً</b> كافياً. "
    "إذا كانت 10,000 مثال تكفي لتقدير دقيق، فلا فرق إن كانت تمثّل 20٪ من 50 ألف "
    "أو 1٪ من مليون. <b>الباقي كله يجب أن يذهب للتدريب</b>، لأن التدريب هو ما يحسّن النموذج.",
    tone="green",
)

st.subheader(":material/analytics: لماذا؟ الحساب الإحصائي وراء القاعدة")

st.markdown(
    """
افترض أنك تقيس **الدقة (Accuracy)**. نتيجتك على مجموعة الاختبار هي تقدير
لاحتمال مجهول $p$. الخطأ المعياري لهذا التقدير هو:
    """
)

equation(
    r"\mathrm{SE}(\hat{p})=\sqrt{\frac{p\,(1-p)}{n_{\text{test}}}}"
    r"\qquad\Longrightarrow\qquad"
    r"\text{CI}_{95\%}\;=\;\hat{p}\;\pm\;1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{n_{\text{test}}}}",
    "لاحظ وجود <b>n في المقام تحت جذر</b>: لتقليل عدم اليقين إلى النصف تحتاج "
    "<b>أربعة أضعاف</b> حجم مجموعة الاختبار. وهذا يفسّر لماذا تصبح الزيادة "
    "بعد بضعة آلاف بلا فائدة تُذكر.",
)

acc_hat = st.slider(
    "الدقة الملاحَظة على الاختبار  p̂", 0.50, 0.99, 0.90, 0.01, key="sp_acc"
)

n_grid = np.unique(np.round(np.logspace(1.3, 5.3, 160)).astype(int))
half = 1.96 * np.sqrt(acc_hat * (1 - acc_hat) / n_grid)

figc = go.Figure()
viz.soft_band(figc, n_grid, np.zeros_like(half), half * 100, viz.C_BLUE, "band")
figc.add_trace(
    go.Scatter(
        x=n_grid, y=half * 100, mode="lines", name="نصف عرض فترة الثقة 95٪ (نقاط مئوية)",
        line=dict(color=viz.C_BLUE, width=4),
        hovertemplate="n_test=%{x:,}<br>±%{y:.2f} نقطة<extra></extra>",
    )
)
for marker, col, lab in ((1.0, viz.C_GREEN, "±1 نقطة"), (2.0, viz.C_AMBER, "±2 نقطة"),
                         (5.0, viz.C_ROSE, "±5 نقاط")):
    need = int(np.ceil(acc_hat * (1 - acc_hat) * (1.96 / (marker / 100)) ** 2))
    figc.add_hline(
        y=marker, line=dict(color=viz.rgba(col, 0.8), width=2, dash="dot"),
        annotation_text=f"{lab}  ←  تحتاج ≈ {need:,} مثالاً",
        annotation_position="top right",
        annotation_font=dict(size=11, color=col),
    )
figc.update_xaxes(type="log")
figc.update_yaxes(range=[0, 12])
viz.style_fig(
    figc, height=430,
    xtitle="حجم مجموعة الاختبار n_test (مقياس لوغاريتمي)",
    ytitle="عدم اليقين ± (نقاط مئوية)",
    title="كم مثالاً تحتاج فعلاً في مجموعة الاختبار؟",
)
st.plotly_chart(figc, width="stretch", key="sp_ci")

need_1 = int(np.ceil(acc_hat * (1 - acc_hat) * (1.96 / 0.01) ** 2))
need_2 = int(np.ceil(acc_hat * (1 - acc_hat) * (1.96 / 0.02) ** 2))
need_5 = int(np.ceil(acc_hat * (1 - acc_hat) * (1.96 / 0.05) ** 2))
nn1, nn2, nn3 = st.columns(3)
nn1.metric("لدقّة ±5 نقاط", f"{need_5:,} مثال")
nn2.metric("لدقّة ±2 نقطة", f"{need_2:,} مثال")
nn3.metric("لدقّة ±1 نقطة", f"{need_1:,} مثال")

st.info(
    f"**اقرأ الأرقام:** بدقة ملاحَظة {acc_hat:.0%}، إذا أردت أن تقول «دقّتي "
    f"{acc_hat:.0%} ± نقطتين» فأنت بحاجة إلى **{need_2:,}** مثال اختبار على الأقل. "
    "وإذا كانت مجموعة اختبارك 100 مثال فقط، فالفاصل سيكون ±6 نقاط تقريباً — "
    "أي أن الفرق بين نموذجين بـ 3 نقاط **لا معنى له إحصائياً**."
)

rule()

# ======================================================================
# 5. أنواع التقسيم
# ======================================================================
st.header(":material/shuffle: أنواع التقسيم — الاختيار الخاطئ يُفسد كل شيء")

st.markdown(
    "التقسيم العشوائي ليس صحيحاً دائماً. إليك الأنواع الأربعة الأساسية "
    "ومتى يكون كل واحد **إلزامياً**:"
)

tabs = st.tabs(
    [
        ":material/casino: عشوائي",
        ":material/layers: طبقي",
        ":material/group: حسب المجموعات",
        ":material/schedule: زمني",
    ]
)

# ---------------------------------------------------------------- random
with tabs[0]:
    st.markdown(
        """
**التقسيم العشوائي (Random split)** هو الافتراضي: نخلط الصفوف ثم نقطع.

- **متى يصلح؟** عندما تكون الأمثلة **مستقلة ومتماثلة التوزيع (i.i.d.)**،
  أي لا ترتيب زمني ولا تكرار لنفس الشخص/الجهاز.
- **خطر خفي:** لو اختلفت نسب الفئات بين القطع صدفة، صار التقييم مضلّلاً.
  (الحل: التقسيم الطبقي في التبويب التالي.)
        """
    )
    st.code(
        """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20٪ للاختبار
    shuffle=True,        # اخلط قبل القطع
    random_state=42,     # ← حتمية النتائج (ضرورية للتكرارية)
)
        """,
        language="python",
    )
    st.markdown(
        """
**وللحصول على ثلاث مجموعات** نستدعي الدالة **مرّتين**:
        """
    )
    st.code(
        """
# 1) اقتطع الاختبار أولاً وضعه جانباً ولا تلمسه
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# 2) قسّم الباقي إلى تدريب وتحقق
#    0.1765 من 85٪ ≈ 15٪ من الأصل
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.1765, random_state=42
)

print(len(X_train), len(X_val), len(X_test))   # ≈ 70% / 15% / 15%
        """,
        language="python",
    )
    st.caption(
        "انتبه للحساب: النسبة في الاستدعاء الثاني تُحسب من **الباقي** لا من الأصل. "
        "القاعدة: `test_size₂ = val_frac / (1 − test_frac)`."
    )

# ------------------------------------------------------------ stratified
with tabs[1]:
    st.markdown(
        """
**التقسيم الطبقي (Stratified split)** يضمن أن **نسبة كل فئة** في كل قطعة
تساوي نسبتها في البيانات الأصلية.

- **متى يكون إلزامياً؟** في **كل تصنيف**، وخصوصاً عند **اللاتوازن**
  (مثلاً 2٪ احتيال و98٪ عادي).
- **لماذا؟** لأن التقسيم العشوائي قد يضع 0 أو 1 حالة احتيال في مجموعة الاختبار،
  فيصبح أي تقييم بلا معنى.
        """
    )

    imb = st.slider("نسبة الفئة النادرة ٪", 1, 50, 5, 1, key="sp_imb")
    n_sim, n_test_sim = 1000, 100
    rs = np.random.default_rng(0)
    rare = int(n_sim * imb / 100)
    labels = np.array([1] * rare + [0] * (n_sim - rare))

    draws = [int(rs.choice(labels, n_test_sim, replace=False).sum()) for _ in range(400)]
    expected = n_test_sim * imb / 100

    figs = go.Figure()
    figs.add_trace(
        go.Histogram(
            x=draws, nbinsx=max(8, int(max(draws) - min(draws)) + 1),
            marker=dict(color=viz.rgba(viz.C_BLUE, 0.65),
                        line=dict(color="#fff", width=1)),
            name="تقسيم عشوائي (400 محاولة)",
            hovertemplate="عدد الحالات النادرة=%{x}<br>تكرار=%{y}<extra></extra>",
        )
    )
    figs.add_vline(
        x=expected, line=dict(color=viz.C_GREEN, width=4),
        annotation_text=f"التقسيم الطبقي يعطي دائماً ≈ {expected:.0f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color=viz.C_GREEN),
    )
    viz.style_fig(
        figs, height=380,
        xtitle=f"عدد أمثلة الفئة النادرة داخل مجموعة اختبار من {n_test_sim}",
        ytitle="عدد المرات",
        title="التقسيم العشوائي يتأرجح — والطبقي يثبّت النسبة",
        legend=False,
    )
    st.plotly_chart(figs, width="stretch", key="sp_strat")

    st.error(
        f"مع فئة نادرة بنسبة **{imb}٪**، التقسيم العشوائي قد يعطيك ما بين "
        f"**{min(draws)}** و **{max(draws)}** حالة في مجموعة الاختبار. "
        "تخيّل أن تقيس «الاستدعاء (Recall)» على 1 أو 2 حالة فقط!"
    )

    st.code(
        """
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,          # ← هذا السطر هو كل الفرق
    random_state=42,
)

# للانحدار: يمكن التقسيم الطبقي على y مُقسَّمة إلى شرائح
import pandas as pd
bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=bins, random_state=42
)
        """,
        language="python",
    )

# ---------------------------------------------------------------- groups
with tabs[2]:
    st.markdown(
        """
**التقسيم حسب المجموعات (Group split)**: عندما تكون عدّة صفوف **تابعة لنفس
الكيان** (مريض له 10 زيارات، مستخدم له 50 عملية، صورة لنفس الشخص من زوايا).

:material/dangerous: **الخطر:** التقسيم العشوائي قد يضع زيارات المريض نفسه
في التدريب **وفي** الاختبار. النموذج سيتعرّف على «المريض» لا على «المرض»،
وستحصل على دقة وهمية عالية جداً تنهار في المستشفى.

**القاعدة:** كل مجموعة تذهب كاملة إلى جهة واحدة فقط.
        """
    )
    st.code(
        """
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=patient_id))

# أو للتحقق المتقاطع
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=patient_id):
    ...
        """,
        language="python",
    )
    st.warning(
        "سؤال اسأله دائماً قبل التقسيم: **«هل يمكن أن يظهر الكيان نفسه في أكثر من صف؟»** "
        "إن كان الجواب نعم، فالتقسيم العشوائي خطأ."
    )

# ------------------------------------------------------------- time based
with tabs[3]:
    st.markdown(
        """
**التقسيم الزمني (Time-based split)**: في السلاسل الزمنية والتنبؤ،
التقسيم العشوائي **خطأ فادح** — لأنه يسمح للنموذج بالتدرّب على **المستقبل**
والتنبؤ بـ **الماضي**. هذا يُسمّى **النظر إلى الأمام (Look-ahead bias)**.

**القاعدة الحديدية:** كل بيانات التدريب يجب أن تكون **أقدم** من كل بيانات الاختبار.
`shuffle=False` دائماً.
        """
    )

    n_t = 120
    cut1, cut2 = int(0.7 * n_t), int(0.85 * n_t)
    t = np.arange(n_t)
    series = np.cumsum(np.random.default_rng(3).normal(0.12, 1.0, n_t)) + 20

    figt = go.Figure()
    figt.add_vrect(x0=-1, x1=cut1, fillcolor=viz.rgba(TRAIN_C, 0.13), line_width=0,
                   annotation_text="تدريب (الماضي)", annotation_position="top left",
                   annotation_font=dict(size=12, color=TRAIN_C))
    figt.add_vrect(x0=cut1, x1=cut2, fillcolor=viz.rgba(VAL_C, 0.16), line_width=0,
                   annotation_text="تحقق", annotation_position="top left",
                   annotation_font=dict(size=12, color=VAL_C))
    figt.add_vrect(x0=cut2, x1=n_t, fillcolor=viz.rgba(TEST_C, 0.16), line_width=0,
                   annotation_text="اختبار (المستقبل)", annotation_position="top left",
                   annotation_font=dict(size=12, color=TEST_C))
    figt.add_trace(
        go.Scatter(x=t, y=series, mode="lines", name="السلسلة الزمنية",
                   line=dict(color=viz.C_BLUE, width=3),
                   hovertemplate="t=%{x}<br>y=%{y:.2f}<extra></extra>")
    )
    viz.style_fig(figt, height=360, xtitle="الزمن →", ytitle="القيمة",
                  title="القطع يكون بالزمن، لا بالعشوائية", legend=False)
    st.plotly_chart(figt, width="stretch", key="sp_time")

    st.code(
        """
# ❌ خطأ فادح في السلاسل الزمنية
train_test_split(X, y, test_size=0.2, shuffle=True)

# ✅ الصحيح: قطع زمني بسيط
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

# ✅ أو نافذة متوسّعة للتحقق المتقاطع الزمني
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=0)
for tr, va in tscv.split(X):
    ...   # tr دائماً قبل va زمنياً
        """,
        language="python",
    )
    st.info(
        "**نصيحة إضافية:** إن كان هدفك التنبؤ بـ 7 أيام مقدّماً، اترك **فجوة (gap)** "
        "من 7 أيام بين نهاية التدريب وبداية الاختبار، وإلا تسرّبت معلومات الحدود."
    )

rule()

# ======================================================================
# 6. تسرّب البيانات
# ======================================================================
st.header(":material/water_damage: تسرّب البيانات — العدو رقم واحد")

st.markdown(
    """
**تسرّب البيانات (Data Leakage)** هو أن تصل إلى النموذج — بشكل مباشر أو غير
مباشر — **معلومة من مجموعة الاختبار** أو **معلومة من المستقبل**. النتيجة دائماً:
أداء رائع في التجربة، وانهيار في الإنتاج. وهو **أكثر سبب لفشل مشاريع تعلّم الآلة**.
    """
)

leaks = pd.DataFrame(
    {
        "الخطأ": [
            "تقييس/تطبيع قبل التقسيم",
            "ملء القيم الناقصة قبل التقسيم",
            "اختيار الخصائص على كل البيانات",
            "صفوف مكرّرة في التدريب والاختبار",
            "خصيصة تحوي الجواب ضمنياً",
            "خصائص مبنية على المستقبل",
            "ضبط العتبة على مجموعة الاختبار",
            "تكرار التجريب على الاختبار",
        ],
        "لماذا هو تسرّب؟": [
            "μ و σ حُسبا من بيانات الاختبار أيضاً",
            "المتوسط/الوسيط تأثّر بقيم الاختبار",
            "الارتباطات حُسبت وأنت ترى الاختبار",
            "النموذج «يتذكّر» الصف لا يتنبّأ به",
            "مثل عمود «تاريخ إلغاء الاشتراك» للتنبؤ بالإلغاء",
            "متوسط متحرّك يشمل أياماً لاحقة",
            "العتبة صارت مضبوطة على الامتحان",
            "كل محاولة تسرّب قليلاً من المعلومة",
        ],
        "العلاج": [
            "fit على التدريب فقط، ثم transform على الباقي (Pipeline)",
            "SimpleImputer داخل Pipeline بعد التقسيم",
            "ضع اختيار الخصائص داخل Pipeline داخل CV",
            "احذف المكرّرات قبل التقسيم",
            "راجع كل عمود: هل كان متاحاً وقت التنبؤ؟",
            "احسب النوافذ بالماضي فقط (shift)",
            "اضبط العتبة على التحقق",
            "استعمل التحقق المتقاطع، والاختبار مرة واحدة",
        ],
    }
)
st.dataframe(leaks, width="stretch", hide_index=True)

st.subheader("الترتيب الصحيح للعمليات — احفظ هذا المخطّط")

d1, d2 = st.columns(2, gap="medium")
with d1:
    st.markdown("##### :material/cancel: الترتيب الخاطئ")
    st.mermaid_chart(
        """
graph TD
    A["كل البيانات"] --> B["تقييس / ملء ناقص<br/>على كل البيانات"]
    B --> C["تقسيم"]
    C --> D["تدريب"]
    D --> E["نتيجة ممتازة<br/>❌ وهمية"]
    style A fill:#EDF1F6,stroke:#8A94A6,color:#17263F
    style B fill:#FDE6EE,stroke:#EB6F92,stroke-width:3px,color:#17263F
    style C fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style D fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style E fill:#FDE6EE,stroke:#EB6F92,stroke-width:2px,color:#17263F
        """
    )
with d2:
    st.markdown("##### :material/check_circle: الترتيب الصحيح")
    st.mermaid_chart(
        """
graph TD
    A["كل البيانات"] --> C["تقسيم أولاً"]
    C --> T["تدريب"]
    C --> S["اختبار (مُجمَّد)"]
    T --> F["fit للتقييس على التدريب فقط"]
    F --> M["تدريب النموذج"]
    M --> P["transform ثم predict على الاختبار"]
    S --> P
    P --> R["نتيجة نزيهة ✅"]
    style A fill:#EDF1F6,stroke:#8A94A6,color:#17263F
    style C fill:#E3F0FE,stroke:#3B7DD8,stroke-width:3px,color:#17263F
    style T fill:#E3F0FE,stroke:#3B7DD8,color:#17263F
    style S fill:#FDE6EE,stroke:#EB6F92,color:#17263F
    style F fill:#DFF6EE,stroke:#2FB58B,color:#17263F
    style M fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style P fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style R fill:#DFF6EE,stroke:#2FB58B,stroke-width:2px,color:#17263F
        """
    )

st.code(
    """
# ❌ تسرّب كلاسيكي
scaler = StandardScaler().fit(X)              # رأى الاختبار!
X_scaled = scaler.transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)

# ✅ الحل الصحيح: Pipeline يضمن الترتيب تلقائياً
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000)),
])

pipe.fit(X_tr, y_tr)        # fit يحدث على التدريب فقط — داخلياً وتلقائياً
print(pipe.score(X_te, y_te))
    """,
    language="python",
)

verdict(
    "لماذا الـ Pipeline ليس رفاهية؟",
    "لأنه <b>يجعل التسرّب مستحيلاً هيكلياً</b>. كل خطوة تُدرَّب على ما يصلها من "
    "بيانات التدريب فقط، وعند التحقق المتقاطع يُعاد ضبط كل خطوة داخل كل طيّة. "
    "استعمل Pipeline دائماً، حتى في التجارب السريعة.",
    tone="blue",
)

rule()

# ======================================================================
# 7. تفاصيل عملية
# ======================================================================
st.header(":material/checklist: تفاصيل عملية يسأل عنها الجميع")

with st.expander(":material/help: هل أخلط البيانات (shuffle) أم لا؟", expanded=True):
    st.markdown(
        """
- **اخلط** في الحالة العامة — كثير من الملفات مرتّبة حسب الفئة أو التاريخ،
  والقطع بدون خلط يعطيك مجموعة اختبار من فئة واحدة.
- **لا تخلط أبداً** في السلاسل الزمنية أو أي بيانات لها ترتيب معنوي.
- عند الخلط **ثبّت `random_state`** حتى تكون نتائجك قابلة للتكرار.
        """
    )

with st.expander(":material/help: ما فائدة random_state؟"):
    st.markdown(
        """
هو **بذرة المولّد العشوائي**. تثبيته يجعل التقسيم نفسه يتكرّر في كل تشغيل، وهذا:

- يسمح لك بمقارنة نموذجين على **نفس** القطع بالضبط (مقارنة عادلة)،
- يجعل نتائجك قابلة للتكرار من طرف غيرك،
- **لكن** لا تقع في الفخّ: إن جرّبت عدّة قيم لـ `random_state` واخترت الأفضل،
  فقد ضبطت نموذجك على التقسيم لا على المشكلة. القيمة تُثبَّت مرة ولا تُبحث.
        """
    )

with st.expander(":material/help: هل أعيد التدريب على كل البيانات في النهاية؟"):
    st.markdown(
        """
**نعم، وهذه ممارسة قياسية** بعد انتهاء كل شيء:

1. استعمل التدريب + التحقق لاختيار النموذج والمعاملات الفائقة.
2. قِس الأداء النهائي على الاختبار **مرة واحدة** ← هذا هو الرقم الذي تُعلنه.
3. ثم **أعد تدريب النموذج بالإعدادات نفسها على 100٪ من البيانات** قبل النشر،
   لأن بيانات أكثر = نموذج أفضل.

:material/warning: لكن **لا تُعِد قياس الأداء** بعد ذلك — الرقم المُعلن يبقى هو رقم الخطوة 2.
        """
    )

with st.expander(":material/help: ماذا لو تغيّر توزيع البيانات مع الزمن؟"):
    st.markdown(
        """
هذا اسمه **انزياح التوزيع (Distribution shift / Concept drift)**. إن كان متوقّعاً:

- اجعل مجموعة الاختبار **من أحدث فترة زمنية**، لا عشوائية — فهي تحاكي الإنتاج.
- أنشئ مجموعة اختبار ثانية «خارج التوزيع» عمداً لقياس المتانة.
- راقب الأداء بعد النشر وأعد التدريب دورياً.
        """
    )

with st.expander(":material/help: بياناتي صغيرة جداً (أقل من 100 صف)، ماذا أفعل؟"):
    st.markdown(
        """
لا تستعمل عيّنة محجوزة — ستكون النتيجة عشوائية. بدلاً من ذلك:

- **Repeated Stratified K-Fold** (مثلاً 5 طيّات × 10 تكرارات) وتقرير المتوسط ± الانحراف.
- أو **LOOCV** إن كان النموذج سريع التدريب.
- و**لا تُبالغ** في تعقيد النموذج: مع 100 صف، نموذج خطي منظَّم غالباً أفضل من غابة عشوائية.
- اذكر دائماً **تباين النتيجة**، لا المتوسط وحده.

(سنفصّل كل هذا في الدرس التالي.)
        """
    )

rule()

# ======================================================================
# 8. قائمة المراجعة
# ======================================================================
st.header(":material/task_alt: قائمة مراجعة التقسيم — مرّ عليها قبل كل مشروع")

steps(
    [
        ("احذف المكرّرات أولاً",
         "صفوف مكرّرة بين التدريب والاختبار = تسرّب مباشر. "
         "<code>df.drop_duplicates()</code> قبل أي شيء."),
        ("حدّد وحدة التقسيم",
         "هل الصف مستقل؟ أم ينتمي إلى مريض/مستخدم/جهاز؟ "
         "إن كان ينتمي لمجموعة، فالتقسيم <b>بالمجموعة</b>."),
        ("هل للبيانات ترتيب زمني؟",
         "إن نعم: <b>قطع زمني</b> و<code>shuffle=False</code> وفجوة مناسبة."),
        ("هل المشكلة تصنيف؟",
         "إن نعم: <b>stratify=y</b> دائماً، خصوصاً مع اللاتوازن."),
        ("اقتطع الاختبار وجمّده",
         "ضعه جانباً حرفياً ولا تفتحه حتى النهاية. "
         "بعضهم يحفظه في ملف منفصل لتجنّب الإغراء."),
        ("ثبّت random_state",
         "مرة واحدة، ولا تبحث عن أفضل بذرة."),
        ("ضع كل المعالجة داخل Pipeline",
         "التقييس، ملء الناقص، الترميز، اختيار الخصائص — كلها داخل الـ Pipeline."),
        ("تحقّق من التوزيعات بعد التقسيم",
         "قارن نسب الفئات ومتوسطات الخصائص بين القطع. "
         "فرق كبير = إشارة إلى مشكلة في التقسيم."),
        ("استعمل الاختبار مرة واحدة",
         "واكتب الرقم في التقرير. إن احتجت تعديلاً بعدها، "
         "فاعلم أن رقمك لم يعد نزيهاً تماماً وصرّح بذلك."),
    ],
    accent="green",
)

footer("الدرس التالي: التحقق المتقاطع — ماذا نفعل حين لا تكفي عيّنة محجوزة واحدة؟")

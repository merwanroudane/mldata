"""الدرس 6 — التحقق المتقاطع: كل طريقة بالحركة."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import TEST_C, TRAIN_C, VAL_C, card, equation, footer, hero, rule, terms, verdict

hero(
    "التحقق المتقاطع — حين لا تكفي عيّنة محجوزة واحدة",
    "Lesson 6 · Cross-validation, fold by fold",
    "عيّنة تحقق واحدة تعطيك رقماً واحداً، وقد يكون هذا الرقم محض حظّ. "
    "التحقق المتقاطع يعطيك عدّة أرقام فتعرف المتوسط <b>والتشتّت</b> معاً.",
)

terms(
    [
        ("Cross-validation", "التحقق المتقاطع"),
        ("Fold", "طيّة"),
        ("K-Fold", "ك-طيّات"),
        ("LOOCV", "ترك واحد خارجاً"),
        ("Nested CV", "تحقق متداخل"),
        ("Variance of estimate", "تباين التقدير"),
    ]
)

# ======================================================================
# 1. المشكلة
# ======================================================================
st.header(":material/casino: المشكلة: رقم واحد لا يكفي")

st.markdown(
    """
لنفترض أن عندك 200 مثالاً. حجزت 40 منها للتحقق فحصلت على دقة **87٪**.
لو حجزت 40 **أخرى** لربما حصلت على **81٪** أو **92٪**.

أي رقم هو الصحيح؟ **كلها**. المشكلة أن تقديرك مبني على عيّنة واحدة صغيرة،
فهو **عالي التباين (High variance)**. والحل البسيط والذكي:

> **بدل أن نحجز قطعة واحدة، نحجز كل قطعة بدورها — ثم نأخذ المتوسط.**
    """
)

n_dem = 200
reps = st.slider("كم مرة نعيد تقسيم 80/20 عشوائياً؟", 5, 300, 80, 5, key="cv_reps")
rs = np.random.default_rng(12)
true_acc = 0.86
sims = rs.binomial(40, true_acc, reps) / 40

figv = go.Figure()
figv.add_trace(
    go.Histogram(
        x=sims * 100, nbinsx=18,
        marker=dict(color=viz.rgba(VAL_C, 0.6), line=dict(color="#fff", width=1)),
        name="دقة عيّنة محجوزة واحدة",
        hovertemplate="الدقة=%{x:.1f}٪<br>التكرار=%{y}<extra></extra>",
    )
)
figv.add_vline(
    x=true_acc * 100, line=dict(color=viz.C_GREEN, width=4),
    annotation_text=f"الدقة الحقيقية = {true_acc:.0%}",
    annotation_position="top right",
    annotation_font=dict(size=12, color=viz.C_GREEN),
)
viz.style_fig(
    figv, height=360, xtitle="الدقة المُقاسة ٪", ytitle="عدد المرات",
    title=f"نفس النموذج، نفس البيانات — فقط التقسيم تغيّر "
          f"(المدى: {sims.min():.0%} إلى {sims.max():.0%})",
    legend=False,
)
st.plotly_chart(figv, width="stretch", key="cv_var")

st.error(
    f"المدى كله من **{sims.min():.0%}** إلى **{sims.max():.0%}** جاء من "
    "**نفس النموذج ونفس البيانات** — الفرق كله من حظّ التقسيم. "
    "لهذا لا يجوز أن تقول «نموذجي أفضل بـ 2٪» بناءً على عيّنة محجوزة واحدة."
)

rule()

# ======================================================================
# 2. K-Fold بالحركة
# ======================================================================
st.header(":material/repeat: K-Fold — الفكرة الأساسية، متحرّكة")

st.markdown(
    """
**الخطوات:** قسّم البيانات إلى $k$ أجزاء متساوية (طيّات). ثم كرّر $k$ مرّات:
في كل مرة اجعل طيّة واحدة هي **التحقق** والباقي **تدريب**. النتيجة النهائية
هي متوسط الـ $k$ نتائج.
    """
)

equation(
    r"\mathrm{CV}_{k}=\frac{1}{k}\sum_{j=1}^{k}\mathcal{M}\big(f^{(-j)},\,D_j\big)"
    r"\qquad\text{و}\qquad"
    r"s=\sqrt{\frac{1}{k-1}\sum_{j=1}^{k}\big(\mathcal{M}_j-\mathrm{CV}_k\big)^2}",
    "الرمز f<sup>(−j)</sup> يعني: النموذج المُدرَّب على كل الطيّات <b>ما عدا</b> الطيّة j، "
    "ثم يُقيَّم على الطيّة j. والانحراف s يقيس <b>مدى استقرار</b> نموذجك — "
    "وهو مهم بقدر أهمية المتوسط نفسه.",
)

k = st.slider("عدد الطيّات k", 2, 10, 5, 1, key="cv_k")
n_cv = 30
fold_of = np.arange(n_cv) % k

frames = []
for j in range(k):
    colors = [TRAIN_C if f != j else VAL_C for f in fold_of]
    frames.append(
        go.Frame(
            name=f"الطيّة {j + 1}",
            data=[
                go.Bar(
                    x=list(range(n_cv)),
                    y=[1] * n_cv,
                    marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
                    hovertext=[
                        f"مثال {i + 1} — " + ("تحقق" if fold_of[i] == j else "تدريب")
                        for i in range(n_cv)
                    ],
                    hoverinfo="text",
                    showlegend=False,
                )
            ],
            traces=[0],
            layout=go.Layout(
                title=dict(
                    text=f"الجولة {j + 1} من {k} — الطيّة {j + 1} للتحقق (برتقالي) "
                         f"والباقي للتدريب (أزرق)"
                )
            ),
        )
    )

figk = go.Figure(
    data=[
        go.Bar(
            x=list(range(n_cv)),
            y=[1] * n_cv,
            marker=dict(
                color=[TRAIN_C if f != 0 else VAL_C for f in fold_of],
                line=dict(color="#FFFFFF", width=1.5),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    ],
    frames=frames,
)
figk.update_yaxes(visible=False, range=[0, 1.25])
viz.style_fig(
    figk, height=260, xtitle="أمثلة البيانات (مرتّبة)", ytitle="",
    title=f"الجولة 1 من {k}", legend=False,
)
figk.update_layout(
    bargap=0.12,
    updatemenus=viz.play_buttons(900, label_play="شغّل الجولات", label_pause="إيقاف", y=-0.45),
    sliders=viz.frame_slider([f"الطيّة {j + 1}" for j in range(k)], prefix="", y=-0.30),
    margin=dict(l=40, r=20, t=52, b=110),
)
st.plotly_chart(figk, width="stretch", key="cv_kfold")

kk1, kk2, kk3 = st.columns(3)
kk1.metric("عدد مرات التدريب", f"{k}")
kk2.metric("نسبة التدريب في كل جولة", f"{(k - 1) / k:.0%}")
kk3.metric("نسبة التحقق في كل جولة", f"{1 / k:.0%}")

st.info(
    f"**لاحظ الميزة الكبرى:** كل مثال من الـ {n_cv} استُعمل للتحقق **مرة واحدة بالضبط** "
    f"وللتدريب **{k - 1} مرات**. أي أننا استفدنا من كل البيانات في التقييم "
    "وفي التدريب معاً — وهذا مستحيل بعيّنة محجوزة واحدة."
)

st.subheader("كيف تختار k؟ المقايضة")

st.markdown(
    """
| k | حجم التدريب في كل جولة | انحياز التقدير | تباين التقدير | التكلفة الحسابية |
|---|---|---|---|---|
| **2** | 50٪ | عالٍ (تدريب قليل ← تقدير متشائم) | منخفض | رخيص جداً |
| **5** | 80٪ | منخفض | متوسّط | 5 تدريبات |
| **10** | 90٪ | منخفض جداً | متوسّط | 10 تدريبات |
| **n (LOOCV)** | 99.9٪ | شبه معدوم | **عالٍ جداً** | n تدريباً — مكلف |

:material/check_circle: **التوصية العملية:** **k = 5** هو الخيار الافتراضي الممتاز
(وk = 10 إن كانت البيانات صغيرة والتدريب سريع). الفرق بين 5 و10 ضئيل عملياً،
بينما التكلفة تتضاعف.
    """
)

verdict(
    "لماذا LOOCV تباينه عالٍ رغم أنه يستعمل أكبر قدر من البيانات؟",
    "لأن النماذج الـ n المُدرَّبة متشابهة جداً (تختلف بمثال واحد فقط)، "
    "فنتائجها <b>مترابطة بشدة</b>، والمتوسط لا يقلّل التباين كما نأمل. "
    "إضافة إلى ذلك، كل تقييم يتم على <b>مثال واحد</b> فقط، وهو إمّا 0 أو 1.",
    tone="amber",
)

rule()

# ======================================================================
# 3. العائلة الكاملة
# ======================================================================
st.header(":material/dashboard: عائلة طرق التحقق المتقاطع كاملة")


def strip_figure(matrix: np.ndarray, title: str, row_labels: list[str]) -> go.Figure:
    """matrix: rows = splits, cols = samples, values 0=unused 1=train 2=val/test."""
    cmap = {0: "#E7ECF3", 1: TRAIN_C, 2: VAL_C}
    f = go.Figure()
    n_rows, n_cols = matrix.shape
    for r in range(n_rows):
        f.add_trace(
            go.Bar(
                x=list(range(n_cols)),
                y=[0.78] * n_cols,
                base=[n_rows - r - 1] * n_cols,
                marker=dict(
                    color=[cmap[int(v)] for v in matrix[r]],
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                hovertext=[
                    f"{row_labels[r]} · مثال {c + 1}: "
                    + {0: "غير مستعمل", 1: "تدريب", 2: "تحقق"}[int(matrix[r, c])]
                    for c in range(n_cols)
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )
    f.update_yaxes(
        tickmode="array",
        tickvals=[n_rows - r - 0.6 for r in range(n_rows)],
        ticktext=row_labels,
        range=[-0.2, n_rows + 0.1],
        showgrid=False,
    )
    viz.style_fig(f, height=70 + 46 * n_rows, xtitle="الأمثلة بالترتيب", ytitle="",
                  title=title, legend=False)
    f.update_layout(barmode="overlay", bargap=0.1)
    return f


cv_tabs = st.tabs(
    [
        ":material/grid_view: K-Fold",
        ":material/layers: Stratified",
        ":material/shuffle: ShuffleSplit",
        ":material/person: Group",
        ":material/schedule: TimeSeries",
        ":material/all_inclusive: LOOCV",
    ]
)

NS = 24

with cv_tabs[0]:
    kk = 4
    M = np.ones((kk, NS))
    for j in range(kk):
        M[j, np.arange(NS) % kk == j] = 2
    st.plotly_chart(strip_figure(M, "K-Fold (k = 4)", [f"جولة {i+1}" for i in range(kk)]),
                    width="stretch", key="cv_s_kfold")
    card(
        "K-Fold",
        "التقسيم الأساسي. كل مثال يُستعمل للتحقق مرة واحدة. "
        "<b>الاستعمال:</b> الحالة العامة، الانحدار، وأي بيانات مستقلة.",
        accent="blue",
        eng="sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)",
    )

with cv_tabs[1]:
    kk = 4
    cls = np.array([1 if i % 5 == 0 else 0 for i in range(NS)])
    M = np.ones((kk, NS))
    for c in (0, 1):
        idx = np.where(cls == c)[0]
        for j in range(kk):
            M[j, idx[np.arange(len(idx)) % kk == j]] = 2
    st.plotly_chart(
        strip_figure(M, "Stratified K-Fold — نسبة الفئات محفوظة في كل طيّة",
                     [f"جولة {i+1}" for i in range(kk)]),
        width="stretch", key="cv_s_strat",
    )
    card(
        "Stratified K-Fold",
        "مثل K-Fold لكنه يحافظ على <b>نسبة كل فئة</b> داخل كل طيّة. "
        "<b>الاستعمال: إلزامي في كل تصنيف</b>، وخصوصاً مع اللاتوازن.",
        accent="green",
        eng="StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
    )

with cv_tabs[2]:
    kk, rs2 = 4, np.random.default_rng(5)
    M = np.ones((kk, NS))
    for j in range(kk):
        M[j, rs2.choice(NS, int(0.25 * NS), replace=False)] = 2
    st.plotly_chart(
        strip_figure(M, "ShuffleSplit — عيّنات تحقق عشوائية مستقلة (قد تتداخل)",
                     [f"جولة {i+1}" for i in range(kk)]),
        width="stretch", key="cv_s_shuffle",
    )
    card(
        "ShuffleSplit / Monte-Carlo CV",
        "تقسيم عشوائي يتكرّر عدّة مرات، بحجم تحقق تختاره أنت وبعدد جولات مستقل عن k. "
        "<b>الاستعمال:</b> بيانات ضخمة (تريد 10 جولات بـ 10٪ فقط)، "
        "أو حين تريد عدد جولات كثيراً دون تصغير مجموعة التحقق.",
        accent="violet",
        eng="ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)",
    )

with cv_tabs[3]:
    groups = np.repeat(np.arange(6), 4)[:NS]
    kk = 3
    M = np.ones((kk, NS))
    for j in range(kk):
        gsel = [g for g in range(6) if g % kk == j]
        M[j, np.isin(groups, gsel)] = 2
    st.plotly_chart(
        strip_figure(M, "GroupKFold — كل مجموعة (4 أمثلة) تبقى كاملة في جهة واحدة",
                     [f"جولة {i+1}" for i in range(kk)]),
        width="stretch", key="cv_s_group",
    )
    card(
        "GroupKFold",
        "يضمن أن أمثلة <b>الكيان نفسه</b> (مريض، مستخدم، جهاز) لا تتوزّع بين "
        "التدريب والتحقق. <b>الاستعمال: إلزامي</b> كلما كان للصفوف انتماء مشترك.",
        accent="rose",
        eng="GroupKFold(n_splits=5).split(X, y, groups=patient_id)",
    )

with cv_tabs[4]:
    kk = 4
    M = np.zeros((kk, NS))
    for j in range(kk):
        end_tr = int(NS * (j + 1) / (kk + 1))
        M[j, :end_tr] = 1
        M[j, end_tr : end_tr + int(NS / (kk + 1))] = 2
    st.plotly_chart(
        strip_figure(M, "TimeSeriesSplit — نافذة تدريب متوسّعة، والتحقق دائماً في المستقبل",
                     [f"جولة {i+1}" for i in range(kk)]),
        width="stretch", key="cv_s_time",
    )
    card(
        "TimeSeriesSplit",
        "نافذة التدريب تكبر مع كل جولة، والتحقق يأتي <b>بعدها زمنياً دائماً</b>. "
        "الرمادي = بيانات مستقبلية لم تُستعمل بعد. "
        "<b>الاستعمال: إلزامي</b> في كل سلسلة زمنية.",
        accent="amber",
        eng="TimeSeriesSplit(n_splits=5, gap=0, max_train_size=None)",
    )
    st.caption(
        "بديل: **النافذة المنزلقة (Rolling window)** حيث يبقى حجم التدريب ثابتاً "
        "وينزلق للأمام — أنسب حين يتغيّر سلوك السلسلة مع الزمن."
    )

with cv_tabs[5]:
    small = 10
    M = np.ones((small, small))
    for j in range(small):
        M[j, j] = 2
    st.plotly_chart(
        strip_figure(M, "LOOCV — كل مثال بدوره هو مجموعة التحقق كلها",
                     [f"جولة {i+1}" for i in range(small)]),
        width="stretch", key="cv_s_loo",
    )
    card(
        "Leave-One-Out CV",
        "الحالة القصوى: k = n. أقصى استفادة من البيانات لكن <b>n تدريباً</b> "
        "وتباين تقدير عالٍ. <b>الاستعمال:</b> بيانات صغيرة جداً ونماذج سريعة فقط.",
        accent="cyan",
        eng="LeaveOneOut()  ·  LeavePOut(p=2)  ·  LeaveOneGroupOut()",
    )

rule()

# ======================================================================
# 4. التحقق المتداخل
# ======================================================================
st.header(":material/account_tree: التحقق المتداخل (Nested CV) — المستوى الاحترافي")

st.markdown(
    """
**المشكلة الدقيقة:** إن استعملت التحقق المتقاطع **لاختيار المعاملات الفائقة**
ثم أعلنت نتيجته نفسها كأداء نهائي، فأنت متفائل — لأنك اخترت الأفضل من بين
عشرات المحاولات على **نفس** الطيّات.

**الحل:** حلقتان متداخلتان:
- **الحلقة الداخلية** تختار المعاملات الفائقة.
- **الحلقة الخارجية** تقيس أداء **العملية كاملة** (بما فيها الاختيار).
    """
)

st.mermaid_chart(
    """
graph TD
    D["كل البيانات"] --> O["حلقة خارجية · 5 طيّات<br/>Outer loop"]
    O --> OT["تدريب خارجي (4 طيّات)"]
    O --> OV["اختبار خارجي (طيّة 1)"]
    OT --> I["حلقة داخلية · 5 طيّات<br/>Inner loop"]
    I --> G["GridSearch على المعاملات الفائقة"]
    G --> B["أفضل إعدادات لهذه الطيّة"]
    B --> RT["أعد التدريب على التدريب الخارجي كله"]
    RT --> SC["قيّم على الاختبار الخارجي"]
    OV --> SC
    SC --> AVG["متوسط الـ 5 نتائج<br/>= تقدير نزيه للأداء"]

    style D fill:#EDF1F6,stroke:#8A94A6,color:#17263F
    style O fill:#FDE6EE,stroke:#EB6F92,stroke-width:3px,color:#17263F
    style I fill:#FDF0DC,stroke:#F0A33C,stroke-width:3px,color:#17263F
    style OT fill:#E3F0FE,stroke:#3B7DD8,color:#17263F
    style OV fill:#FDE6EE,stroke:#EB6F92,color:#17263F
    style G fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style B fill:#EEE6FB,stroke:#9B6BD6,color:#17263F
    style RT fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style SC fill:#FFFFFF,stroke:#8A94A6,color:#17263F
    style AVG fill:#DFF6EE,stroke:#2FB58B,stroke-width:3px,color:#17263F
    """
)

st.code(
    """
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

grid = GridSearchCV(
    estimator=pipe,                  # Pipeline كامل (تقييس + نموذج)
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=inner,                        # ← الحلقة الداخلية: اختيار المعاملات
    scoring="f1",
    n_jobs=-1,
)

scores = cross_val_score(grid, X, y, cv=outer, scoring="f1")   # ← الحلقة الخارجية
print(f"F1 = {scores.mean():.3f} ± {scores.std():.3f}")
    """,
    language="python",
)

st.warning(
    "**التكلفة:** 5 × 5 × (عدد التركيبات). مع 5 قيم لـ C نحصل على 125 تدريباً. "
    "استعمل التحقق المتداخل في الأبحاث والنشر العلمي؛ وفي العمل اليومي "
    "يكفي غالباً: تحقق متقاطع للاختيار + مجموعة اختبار محجوزة منفصلة."
)

rule()

# ======================================================================
# 5. قراءة النتائج
# ======================================================================
st.header(":material/insights: كيف تقرأ نتيجة التحقق المتقاطع؟")

st.markdown(
    "لا تكتب أبداً «الدقة 0.87» فقط. اكتب دائماً **المتوسط ± الانحراف المعياري**. "
    "التشتّت يقول لك هل نتيجتك موثوقة أم هشّة:"
)

demo = pd.DataFrame(
    {
        "الطيّة": [1, 2, 3, 4, 5],
        "النموذج أ (مستقر)": [0.861, 0.874, 0.868, 0.859, 0.872],
        "النموذج ب (هشّ)": [0.930, 0.742, 0.901, 0.690, 0.915],
    }
)
A = demo["النموذج أ (مستقر)"].to_numpy()
B = demo["النموذج ب (هشّ)"].to_numpy()

cvt1, cvt2 = st.columns([1.1, 1], gap="medium")
with cvt1:
    st.dataframe(demo, width="stretch", hide_index=True)
with cvt2:
    figb = go.Figure()
    figb.add_trace(go.Box(y=A, name="النموذج أ", marker_color=viz.C_BLUE,
                          boxpoints="all", jitter=0.4, pointpos=0))
    figb.add_trace(go.Box(y=B, name="النموذج ب", marker_color=viz.C_ROSE,
                          boxpoints="all", jitter=0.4, pointpos=0))
    viz.style_fig(figb, height=330, ytitle="الدقة", title="التشتّت يحكي القصة", legend=False)
    st.plotly_chart(figb, width="stretch", key="cv_box")

b1, b2 = st.columns(2)
b1.metric("النموذج أ", f"{A.mean():.3f} ± {A.std(ddof=1):.3f}")
b2.metric("النموذج ب", f"{B.mean():.3f} ± {B.std(ddof=1):.3f}")

verdict(
    "أيّهما تختار؟",
    f"متوسطهما متقارب ({A.mean():.3f} مقابل {B.mean():.3f})، لكن انحراف النموذج ب "
    f"أكبر بـ {B.std(ddof=1) / A.std(ddof=1):.1f} مرة. "
    "<b>اختر النموذج أ.</b> النموذج ب يعتمد بشدّة على أي بيانات صادفها — "
    "وهذا سلوك نموذج سيفاجئك بشكل سيئ في الإنتاج. "
    "في الهندسة العملية، <b>الاستقرار يساوي الدقة في الأهمية</b>.",
    tone="blue",
)

st.code(
    """
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

res = cross_validate(
    pipe, X, y,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"],   # عدّة مقاييس دفعة واحدة
    return_train_score=True,                 # ← مهم جداً لكشف الإفراط
    n_jobs=-1,
)

for m in ["accuracy", "f1", "roc_auc"]:
    tr = res[f"train_{m}"]
    te = res[f"test_{m}"]
    print(f"{m}: train {tr.mean():.3f} | valid {te.mean():.3f} ± {te.std():.3f}")
    """,
    language="python",
)

st.success(
    "**`return_train_score=True` هو كنز:** مقارنة درجة التدريب بدرجة التحقق "
    "في كل طيّة تكشف الإفراط في التعلّم فوراً — وهو موضوع الدرس التالي."
)

footer("الدرس التالي: الإفراط والتقصير في التعلّم — تشخيص المرض من الأرقام.")

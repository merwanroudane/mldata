"""الدرس 7 — الإفراط والتقصير في التعلّم، والمقايضة بين التحيّز والتباين."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.data import true_curve, wave_sample
from utils.style import card, equation, footer, hero, rule, steps, terms, verdict

hero(
    "الإفراط والتقصير في التعلّم — تشخيص المرض من الأرقام",
    "Lesson 7 · Overfitting, underfitting and the bias–variance trade-off",
    "لماذا يعطيك نموذج دقّة 99٪ في التدريب و55٪ في الواقع؟ ولماذا يفشل نموذج آخر "
    "في كل مكان؟ هنا نتعلّم قراءة الأعراض ووصف الدواء.",
)

terms(
    [
        ("Overfitting", "إفراط في التعلّم"),
        ("Underfitting", "تقصير في التعلّم"),
        ("Bias", "التحيّز"),
        ("Variance", "التباين"),
        ("Generalization gap", "فجوة التعميم"),
        ("Model capacity", "سعة النموذج"),
        ("Regularization", "التنظيم"),
        ("Irreducible error", "خطأ غير قابل للإزالة"),
    ]
)

# ======================================================================
# 1. التشبيه
# ======================================================================
st.header(":material/school: التشبيه الذي يشرح كل شيء: ثلاثة طلاب")

t1, t2, t3 = st.columns(3, gap="medium")
with t1:
    card(
        "الطالب الكسول — تقصير",
        "لم يدرس إلا صفحة واحدة. يجيب إجابة واحدة على كل سؤال. "
        "<b>ضعيف في التمارين وضعيف في الامتحان</b>. نموذجه بسيط جداً "
        "ليستوعب تعقيد المادة.",
        accent="amber",
        eng="Underfitting · high bias · low variance",
    )
with t2:
    card(
        "الطالب الحافظ — إفراط",
        "حفظ كل التمارين وحلولها حرفاً بحرف، حتى الأخطاء المطبعية. "
        "<b>ممتاز في التمارين، كارثي في الامتحان</b> لأن الأسئلة صيغت بشكل مختلف.",
        accent="rose",
        eng="Overfitting · low bias · high variance",
    )
with t3:
    card(
        "الطالب الفاهم — التوازن",
        "فهم <b>القاعدة</b> وراء التمارين وتجاهل التفاصيل العرَضية. "
        "<b>جيد في التمارين وجيد في الامتحان</b>. هذا هدفنا دائماً.",
        accent="green",
        eng="Good fit · balanced bias and variance",
    )

verdict(
    "الفكرة الأساسية في جملة",
    "البيانات = <b>إشارة + ضجيج</b>. التقصير يعني أن النموذج لم يلتقط حتى الإشارة. "
    "الإفراط يعني أنه التقط الإشارة <b>وحفظ الضجيج معها</b>. "
    "والنموذج الجيد يلتقط الإشارة ويتجاهل الضجيج.",
    tone="violet",
)

rule()

# ======================================================================
# 2. المختبر الرئيسي
# ======================================================================
st.header(":material/science: المختبر — حرّك مرونة النموذج وشاهد الظاهرتين")

st.markdown(
    """
سنُولِّد بيانات من منحنى حقيقي (رمادي منقّط) ثم نضيف ضجيجاً. سنُدرّب كثير حدود
(Polynomial) بدرجات مختلفة. **الدرجة = مرونة النموذج**:

- **درجة 1** = خط مستقيم فقط ← نموذج صلب جداً.
- **درجة 15** = منحنى يستطيع أن يلتوي كما يشاء ← نموذج مرن جداً.

حرّك المؤشر وراقب **الخطين في الرسم الثاني**: خطأ التدريب وخطأ الاختبار.
    """
)

ov1, ov2, ov3 = st.columns(3, gap="medium")
with ov1:
    degree = st.slider("درجة كثير الحدود (المرونة)", 1, 16, 3, 1, key="of_deg")
with ov2:
    n_train = st.slider("حجم بيانات التدريب", 10, 120, 24, 2, key="of_n")
with ov3:
    noise_lv = st.slider("مستوى الضجيج σ", 0.05, 0.9, 0.28, 0.05, key="of_noise")

df_tr = wave_sample(n_train, noise_lv, seed=11)
df_te = wave_sample(200, noise_lv, seed=99)

x_tr, y_tr = df_tr["x"].to_numpy(), df_tr["y"].to_numpy()
x_te, y_te = df_te["x"].to_numpy(), df_te["y"].to_numpy()


def poly_fit(x, y, d):
    """Least-squares polynomial fit (degree capped below the sample size)."""
    d = int(min(d, max(1, len(x) - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.polynomial.polynomial.Polynomial.fit(x, y, deg=d)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


model = poly_fit(x_tr, y_tr, degree)
grid = np.linspace(0, 6.4, 400)

err_tr = mse(model(x_tr), y_tr)
err_te = mse(model(x_te), y_te)

cfit, cerr = st.columns([1.15, 1], gap="medium")

with cfit:
    figf = go.Figure()
    figf.add_trace(
        go.Scatter(x=grid, y=true_curve(grid), mode="lines", name="العلاقة الحقيقية",
                   line=dict(color=viz.rgba(viz.C_SLATE, 0.7), width=3, dash="dot"),
                   hoverinfo="skip")
    )
    figf.add_trace(
        go.Scatter(x=x_te, y=y_te, mode="markers", name="بيانات اختبار",
                   marker=dict(size=5, color=viz.rgba(viz.C_ROSE, 0.30)),
                   hoverinfo="skip")
    )
    figf.add_trace(
        go.Scatter(x=x_tr, y=y_tr, mode="markers", name="بيانات تدريب",
                   marker=dict(size=11, color=viz.C_BLUE,
                               line=dict(color="#fff", width=2)),
                   hoverinfo="skip")
    )
    figf.add_trace(
        go.Scatter(x=grid, y=model(grid), mode="lines", name=f"النموذج (درجة {degree})",
                   line=dict(color=viz.C_VIOLET, width=4.5),
                   hovertemplate="ŷ=%{y:.2f}<extra></extra>")
    )
    figf.update_yaxes(range=[float(y_te.min()) - 1.2, float(y_te.max()) + 1.2])
    viz.style_fig(figf, height=420, xtitle="x", ytitle="y",
                  title=f"ملاءمة النموذج · درجة {degree}")
    st.plotly_chart(figf, width="stretch", key="of_fit")

with cerr:
    degs = np.arange(1, 17)
    e_tr, e_te = [], []
    for d in degs:
        m = poly_fit(x_tr, y_tr, int(d))
        e_tr.append(mse(m(x_tr), y_tr))
        e_te.append(mse(np.clip(m(x_te), -50, 50), y_te))
    e_tr, e_te = np.array(e_tr), np.array(e_te)
    best_d = int(degs[int(np.argmin(e_te))])

    fige = go.Figure()
    fige.add_trace(
        go.Scatter(x=degs, y=e_tr, mode="lines+markers", name="خطأ التدريب",
                   line=dict(color=viz.C_BLUE, width=3.5), marker=dict(size=7),
                   hovertemplate="درجة %{x}<br>MSE=%{y:.4f}<extra></extra>")
    )
    fige.add_trace(
        go.Scatter(x=degs, y=e_te, mode="lines+markers", name="خطأ الاختبار",
                   line=dict(color=viz.C_ROSE, width=3.5, dash="dash"),
                   marker=dict(size=7),
                   hovertemplate="درجة %{x}<br>MSE=%{y:.4f}<extra></extra>")
    )
    fige.add_vline(x=best_d, line=dict(color=viz.C_GREEN, width=3, dash="dot"),
                   annotation_text=f"الأفضل: درجة {best_d}",
                   annotation_position="top left",
                   annotation_font=dict(size=12, color=viz.C_GREEN))
    fige.add_vline(x=degree, line=dict(color=viz.rgba(viz.C_VIOLET, 0.65), width=3),
                   annotation_text="أنت هنا", annotation_position="bottom right",
                   annotation_font=dict(size=11, color=viz.C_VIOLET))
    fige.add_vrect(x0=0.5, x1=best_d - 0.5, fillcolor=viz.rgba(viz.C_AMBER, 0.10),
                   line_width=0, annotation_text="منطقة التقصير",
                   annotation_position="top left",
                   annotation_font=dict(size=10, color=viz.C_AMBER))
    fige.add_vrect(x0=best_d + 0.5, x1=16.5, fillcolor=viz.rgba(viz.C_ROSE, 0.10),
                   line_width=0, annotation_text="منطقة الإفراط",
                   annotation_position="top right",
                   annotation_font=dict(size=10, color=viz.C_ROSE))
    fige.update_yaxes(type="log")
    viz.style_fig(fige, height=420, xtitle="درجة كثير الحدود (المرونة) →",
                  ytitle="MSE (مقياس لوغاريتمي)",
                  title="منحنى التحقق — شكل حرف U الشهير")
    st.plotly_chart(fige, width="stretch", key="of_valcurve")

mm1, mm2, mm3, mm4 = st.columns(4)
mm1.metric("خطأ التدريب", f"{err_tr:.4f}")
mm2.metric("خطأ الاختبار", f"{err_te:.4f}")
mm3.metric("فجوة التعميم", f"{err_te - err_tr:+.4f}")
mm4.metric("الدرجة المثلى هنا", f"{best_d}")

ratio = err_te / max(err_tr, 1e-9)
if degree < best_d and err_tr > 1.6 * noise_lv**2:
    st.warning(
        "**تقصير في التعلّم (Underfitting).** خطأ التدريب نفسه مرتفع — "
        "النموذج لا يستطيع حتى ملاءمة البيانات التي رآها. "
        "لاحظ أن المنحنى البنفسجي لا يتابع شكل المنحنى الرمادي. "
        ":material/build: **العلاج:** زِد المرونة (درجة أعلى، نموذج أقوى، خصائص أكثر)."
    )
elif ratio > 3.0 and degree > best_d:
    st.error(
        f"**إفراط في التعلّم (Overfitting).** خطأ الاختبار أكبر بـ **{ratio:.1f}×** "
        "من خطأ التدريب. المنحنى يلتوي ليمرّ بكل نقطة زرقاء — إنه **يحفظ الضجيج**. "
        ":material/build: **العلاج:** خفّض الدرجة، أضف تنظيماً، أو زِد بيانات التدريب."
    )
else:
    st.success(
        "**ملاءمة جيدة.** الخطآن متقاربان وقريبان من الحد النظري σ² = "
        f"{noise_lv**2:.4f}. النموذج التقط الإشارة ولم يطارد الضجيج."
    )

st.info(
    ":material/lightbulb: **جرّب هذا الآن:** ثبّت الدرجة على **15**، ثم حرّك "
    "«حجم بيانات التدريب» من 10 إلى 120 تدريجياً. ستشاهد الإفراط **يتلاشى من تلقاء نفسه**. "
    "هذا يثبت القاعدة: **بيانات أكثر هي أقوى علاج للإفراط في التعلّم**."
)

rule()

# ======================================================================
# 3. جدول التشخيص
# ======================================================================
st.header(":material/medical_services: جدول التشخيص — اقرأ المرض من رقمين")

st.markdown(
    """
كل ما تحتاجه رقمان: **خطأ التدريب** و**خطأ التحقق/الاختبار**. وأربع حالات لا خامس لها:
    """
)

diag = pd.DataFrame(
    {
        "خطأ التدريب": ["مرتفع", "منخفض", "مرتفع", "منخفض"],
        "خطأ الاختبار": ["مرتفع", "مرتفع", "منخفض", "منخفض"],
        "التشخيص": [
            "تقصير (Underfitting) — تحيّز عالٍ",
            "إفراط (Overfitting) — تباين عالٍ",
            "مستحيل عادة ← ابحث عن خطأ في الكود أو تسرّب",
            "ملاءمة جيدة (Good fit) ✅",
        ],
        "العلاج": [
            "نموذج أقوى · خصائص أفضل · تدريب أطول · تنظيم أقل",
            "بيانات أكثر · تنظيم أقوى · نموذج أبسط · توقّف مبكّر",
            "راجع التقسيم: هل مجموعة الاختبار أسهل؟ هل هناك تسرّب؟",
            "لا شيء — جمّد النموذج وانشره",
        ],
    }
)
st.dataframe(diag, width="stretch", hide_index=True)

st.markdown(
    """
:material/priority_high: **تنبيه مهم:** «مرتفع» و«منخفض» نسبيان، ويُقاسان مقابل
**مستوى الأداء المقبول (Bayes error / Human level)**. خطأ 10٪ قد يكون ممتازاً
في مشكلة صعبة وكارثياً في مشكلة سهلة. حدّد دائماً **المرجع** قبل الحكم.
    """
)

rule()

# ======================================================================
# 4. منحنيات التعلّم
# ======================================================================
st.header(":material/stacked_line_chart: منحنيات التعلّم — هل تنفع بيانات أكثر؟")

st.markdown(
    """
**منحنى التعلّم (Learning curve)** يرسم الخطأ مقابل **حجم بيانات التدريب**.
إنه أهم أداة تشخيصية لأنه يجيب عن سؤال عملي بملايين الدولارات:
**هل أستثمر في جمع بيانات أكثر، أم في نموذج أفضل؟**
    """
)

scenario = st.segmented_control(
    "اختر الحالة",
    ["تقصير (تحيّز عالٍ)", "إفراط (تباين عالٍ)", "ملاءمة جيدة"],
    default="إفراط (تباين عالٍ)",
    selection_mode="single",
    key="of_scen",
)

m_sizes = np.arange(10, 420, 10)
if scenario == "تقصير (تحيّز عالٍ)":
    lc_tr = 0.44 + 0.05 * np.exp(-m_sizes / 60)
    lc_va = 0.50 + 0.16 * np.exp(-m_sizes / 60)
    note = (
        "**الخطآن يلتقيان بسرعة عند مستوى مرتفع.** زيادة البيانات لن تساعد إطلاقاً — "
        "المنحنيان مسطّحان. المشكلة في **النموذج نفسه**: هو بسيط جداً."
    )
    tone = "amber"
elif scenario == "إفراط (تباين عالٍ)":
    lc_tr = 0.035 + 0.045 * np.exp(-m_sizes / 120)
    lc_va = 0.085 + 0.58 * np.exp(-m_sizes / 110)
    note = (
        "**فجوة واسعة بين المنحنيين، لكنها تضيق باستمرار.** "
        "هذه أوضح إشارة على أن **جمع بيانات أكثر سيساعد فعلاً**."
    )
    tone = "rose"
else:
    lc_tr = 0.075 + 0.035 * np.exp(-m_sizes / 80)
    lc_va = 0.098 + 0.10 * np.exp(-m_sizes / 80)
    note = (
        "**المنحنيان متقاربان ومستقرّان عند مستوى منخفض.** "
        "وصلت إلى حدود ما تسمح به بياناتك. "
        "التحسّن الإضافي يحتاج خصائص جديدة لا كميات أكبر."
    )
    tone = "green"

figl = go.Figure()
viz.soft_band(figl, m_sizes, lc_tr, lc_va, viz.C_VIOLET, "الفجوة")
figl.add_trace(
    go.Scatter(x=m_sizes, y=lc_tr, mode="lines", name="خطأ التدريب",
               line=dict(color=viz.C_BLUE, width=4),
               hovertemplate="n=%{x}<br>خطأ=%{y:.3f}<extra></extra>")
)
figl.add_trace(
    go.Scatter(x=m_sizes, y=lc_va, mode="lines", name="خطأ التحقق",
               line=dict(color=viz.C_ROSE, width=4, dash="dash"),
               hovertemplate="n=%{x}<br>خطأ=%{y:.3f}<extra></extra>")
)
figl.add_hline(y=0.03, line=dict(color=viz.rgba(viz.C_GREEN, 0.8), width=2, dash="dot"),
               annotation_text="المستوى المرجعي المنشود",
               annotation_position="bottom right",
               annotation_font=dict(size=11, color=viz.C_GREEN))
figl.update_yaxes(range=[0, 0.72])
viz.style_fig(figl, height=410, xtitle="حجم بيانات التدريب n →", ytitle="الخطأ",
              title=f"منحنى التعلّم — {scenario}")
st.plotly_chart(figl, width="stretch", key="of_lc")

verdict("كيف تقرأ المنحنى؟", note, tone=tone)

st.markdown(
    """
| ما تراه | المعنى | القرار |
|---|---|---|
| المنحنيان متقاربان **ومرتفعان** | تحيّز عالٍ | **لا تجمع بيانات**؛ قوِّ النموذج |
| فجوة **واسعة** تضيق مع n | تباين عالٍ | **اجمع بيانات أكثر** — سيساعد |
| فجوة واسعة **ثابتة** لا تضيق | تسرّب أو اختلاف توزيع | راجع التقسيم والبيانات |
| المنحنيان متقاربان ومنخفضان | ملاءمة جيدة | انشر النموذج |
    """
)

st.code(
    """
from sklearn.model_selection import learning_curve, validation_curve
import numpy as np

# 1) منحنى التعلّم: الخطأ مقابل حجم البيانات
sizes, train_sc, val_sc = learning_curve(
    pipe, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
)

# 2) منحنى التحقق: الخطأ مقابل تعقيد النموذج
train_sc, val_sc = validation_curve(
    pipe, X, y,
    param_name="model__max_depth",
    param_range=[1, 2, 3, 5, 8, 12, 20, None],
    cv=5, scoring="f1", n_jobs=-1,
)
    """,
    language="python",
)

rule()

# ======================================================================
# 5. التحيّز والتباين
# ======================================================================
st.header(":material/balance: المقايضة بين التحيّز والتباين — الأساس النظري")

st.markdown(
    """
هذه أهم معادلة نظرية في تعلّم الآلة. إن فهمتها فهمت لماذا لا يوجد «نموذج أفضل مطلقاً».
الخطأ المتوقَّع عند أي نقطة $x$ يتفكّك إلى **ثلاثة** أجزاء:
    """
)

equation(
    r"\mathbb{E}\Big[\big(y-\hat{f}(x)\big)^{2}\Big]="
    r"\underbrace{\Big(\mathbb{E}[\hat{f}(x)]-f(x)\Big)^{2}}_{\textbf{Bias}^2}"
    r"\;+\;\underbrace{\mathbb{E}\Big[\big(\hat{f}(x)-\mathbb{E}[\hat{f}(x)]\big)^{2}\Big]}_{\textbf{Variance}}"
    r"\;+\;\underbrace{\sigma^{2}}_{\textbf{Irreducible}}",
    "الخطأ الكلي = مربع التحيّز + التباين + الخطأ غير القابل للإزالة.",
)

bv1, bv2, bv3 = st.columns(3, gap="medium")
with bv1:
    card(
        "التحيّز² (Bias²)",
        "كم يخطئ نموذجك <b>في المتوسط</b> لو أعدنا التدريب على عيّنات كثيرة مختلفة. "
        "تحيّز عالٍ = افتراضات صلبة جداً (مثل فرض أن العلاقة خطية وهي ليست كذلك). "
        "<b>↑ مع بساطة النموذج</b>.",
        accent="amber",
        eng="Systematic error from wrong assumptions",
    )
with bv2:
    card(
        "التباين (Variance)",
        "كم تتغيّر تنبؤات نموذجك إذا غيّرنا بيانات التدريب قليلاً. "
        "تباين عالٍ = النموذج يلاحق تفاصيل العيّنة. "
        "<b>↑ مع مرونة النموذج</b>.",
        accent="rose",
        eng="Sensitivity to the particular training sample",
    )
with bv3:
    card(
        "σ² — غير قابل للإزالة",
        "ضجيج أصيل في البيانات: أخطاء قياس، عشوائية حقيقية، متغيّرات مفقودة. "
        "<b>لا يمكن لأي نموذج في الوجود تخفيضه</b>. هو الحدّ الأدنى المطلق لخطئك.",
        accent="slate",
        eng="Noise floor · Bayes error",
    )

cap = np.linspace(0.2, 10, 220)
bias2 = 4.2 / (cap**1.35)
var = 0.055 * cap**1.55
irr = np.full_like(cap, 0.42)
total = bias2 + var + irr
opt = float(cap[int(np.argmin(total))])

figbv = go.Figure()
figbv.add_trace(go.Scatter(x=cap, y=bias2, mode="lines", name="التحيّز² (Bias²)",
                           line=dict(color=viz.C_AMBER, width=3.5),
                           hovertemplate="%{y:.3f}<extra></extra>"))
figbv.add_trace(go.Scatter(x=cap, y=var, mode="lines", name="التباين (Variance)",
                           line=dict(color=viz.C_ROSE, width=3.5),
                           hovertemplate="%{y:.3f}<extra></extra>"))
figbv.add_trace(go.Scatter(x=cap, y=irr, mode="lines", name="σ² — خطأ غير قابل للإزالة",
                           line=dict(color=viz.C_SLATE, width=2.5, dash="dot"),
                           hovertemplate="%{y:.3f}<extra></extra>"))
figbv.add_trace(go.Scatter(x=cap, y=total, mode="lines", name="الخطأ الكلي",
                           line=dict(color=viz.C_BLUE, width=5),
                           hovertemplate="%{y:.3f}<extra></extra>"))
figbv.add_vline(x=opt, line=dict(color=viz.C_GREEN, width=3, dash="dash"),
                annotation_text="السعة المثلى",
                annotation_position="top right",
                annotation_font=dict(size=12, color=viz.C_GREEN))
figbv.update_yaxes(range=[0, 6])
viz.style_fig(figbv, height=430, xtitle="سعة النموذج / مرونته →", ytitle="الخطأ المتوقَّع",
              title="التفكيك الكلاسيكي: المجموع يرسم حرف U")
st.plotly_chart(figbv, width="stretch", key="of_bv")

verdict(
    "لماذا سُمّيت «مقايضة»؟",
    "لأن تقليل أحدهما يرفع الآخر عادة. نموذج أبسط ← تحيّز أعلى وتباين أقل. "
    "نموذج أمرن ← تحيّز أقل وتباين أعلى. "
    "هدفك ليس تصفير أيّهما، بل <b>تصغير مجموعهما</b> — وهذه هي النقطة الخضراء. "
    "<br><br>(ملاحظة متقدّمة: في الشبكات العصبية الضخمة جداً لُوحظت ظاهرة "
    "<b>Double Descent</b> حيث يعود الخطأ للنزول بعد الارتفاع — لكن منحنى U "
    "يبقى الإطار الصحيح للنماذج الكلاسيكية التي تبدأ بها.)",
    tone="blue",
)

rule()

# ======================================================================
# 6. التنظيم
# ======================================================================
st.header(":material/compress: التنظيم (Regularization) — كبح جماح النموذج")

st.markdown(
    """
بدلاً من تبسيط النموذج يدوياً، نترك له مرونته لكن **نعاقبه** على استعمالها.
نضيف حدّاً جزائياً إلى دالة التكلفة:
    """
)

equation(
    r"J_{\text{reg}}(\boldsymbol{\theta})=\underbrace{J(\boldsymbol{\theta})}_{\text{الخطأ}}"
    r"\;+\;\lambda\,\underbrace{\Omega(\boldsymbol{\theta})}_{\text{عقوبة التعقيد}}",
    "λ (لامبدا) هو <b>معامل فائق</b> يحدّد شدّة العقوبة: "
    "λ=0 يعني لا تنظيم إطلاقاً، وλ كبيرة جداً تسحق كل المعاملات إلى الصفر.",
)

rg1, rg2 = st.columns(2, gap="medium")
with rg1:
    equation(
        r"\text{Ridge (L2)}:\quad \Omega=\sum_{j=1}^{p}w_j^{2}",
        "يُصغّر كل الأوزان نحو الصفر <b>دون أن يصل أيّها إلى الصفر تماماً</b>. "
        "ممتاز عندما تكون كل الخصائص مفيدة قليلاً، وعند وجود ارتباط بينها.",
    )
with rg2:
    equation(
        r"\text{Lasso (L1)}:\quad \Omega=\sum_{j=1}^{p}\big|w_j\big|",
        "يدفع أوزاناً كاملة إلى <b>الصفر بالضبط</b>، فيقوم عملياً "
        "<b>باختيار الخصائص</b> نيابة عنك. ممتاز حين تشكّ أن معظم الخصائص عديمة الفائدة.",
    )

st.markdown("**Elastic Net** يجمع الاثنين: $\\Omega = \\alpha\\sum|w_j| + (1-\\alpha)\\sum w_j^2$.")

lam = st.select_slider(
    "شدّة التنظيم λ",
    options=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    value=1e-4,
    key="of_lam",
)

d_fix = 14
Xp = np.vander(x_tr, d_fix + 1, increasing=True)
Xp_te = np.vander(x_te, d_fix + 1, increasing=True)
sc = np.maximum(np.abs(Xp).max(axis=0), 1e-12)
Xs, Xs_te = Xp / sc, Xp_te / sc
I = np.eye(d_fix + 1)
I[0, 0] = 0.0
coef = np.linalg.pinv(Xs.T @ Xs + lam * I * len(x_tr)) @ (Xs.T @ y_tr)

gs = np.vander(grid, d_fix + 1, increasing=True) / sc
y_hat_grid = gs @ coef

figr = go.Figure()
figr.add_trace(go.Scatter(x=grid, y=true_curve(grid), mode="lines", name="الحقيقة",
                          line=dict(color=viz.rgba(viz.C_SLATE, 0.7), width=3, dash="dot"),
                          hoverinfo="skip"))
figr.add_trace(go.Scatter(x=x_tr, y=y_tr, mode="markers", name="تدريب",
                          marker=dict(size=10, color=viz.C_BLUE,
                                      line=dict(color="#fff", width=2)),
                          hoverinfo="skip"))
figr.add_trace(go.Scatter(x=grid, y=y_hat_grid, mode="lines",
                          name=f"كثير حدود درجة {d_fix} · λ = {lam:g}",
                          line=dict(color=viz.C_GREEN, width=4.5),
                          hovertemplate="ŷ=%{y:.2f}<extra></extra>"))
figr.update_yaxes(range=[float(y_te.min()) - 1.2, float(y_te.max()) + 1.2])
viz.style_fig(figr, height=400, xtitle="x", ytitle="y",
              title="نموذج شديد المرونة (درجة 14) يُروَّض بالتنظيم فقط")
st.plotly_chart(figr, width="stretch", key="of_reg")

rr1, rr2, rr3 = st.columns(3)
rr1.metric("خطأ التدريب", f"{mse(Xs @ coef, y_tr):.4f}")
rr2.metric("خطأ الاختبار", f"{mse(np.clip(Xs_te @ coef, -50, 50), y_te):.4f}")
rr3.metric("مجموع مربعات الأوزان ‖w‖²", f"{float(np.sum(coef[1:] ** 2)):,.2f}")

if lam == 0.0:
    st.error(
        "λ = 0 ← لا تنظيم. لاحظ التموّجات العنيفة والأوزان الضخمة. "
        "**هذا هو الإفراط في التعلّم مرسوماً.**"
    )
elif lam >= 1.0:
    st.warning(
        "λ كبيرة جداً ← سُحقت الأوزان وصار المنحنى شبه مستقيم. "
        "**عالجنا الإفراط لكننا وقعنا في التقصير.** هذه هي المقايضة بعينها."
    )
else:
    st.success(
        "λ معتدلة ← المنحنى يتابع الشكل الحقيقي دون مطاردة كل نقطة. "
        "لاحظ: **الدرجة ما زالت 14**، لكن التنظيم منعها من الانفلات."
    )

st.markdown(
    """
:material/key: **كيف تختار λ؟** لا بالتخمين ولا بالقراءة — بل **بالتحقق المتقاطع**:
جرّب شبكة لوغاريتمية `[1e-5, 1e-4, ..., 10, 100]` واختر التي تُعطي أفضل درجة تحقق.
    """
)

st.code(
    """
from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np

alphas = np.logspace(-5, 2, 30)

ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train, y_train)

print("أفضل λ لـ Ridge:", ridge.alpha_)
print("أفضل λ لـ Lasso:", lasso.alpha_)
print("خصائص ألغاها Lasso:", int(np.sum(lasso.coef_ == 0)), "من", len(lasso.coef_))
    """,
    language="python",
)

rule()

# ======================================================================
# 7. صندوق الأدوات
# ======================================================================
st.header(":material/handyman: صندوق الأدوات — ماذا أفعل لكل حالة؟")

o1, o2 = st.columns(2, gap="medium")
with o1:
    st.markdown("##### :material/trending_down: علاج الإفراط (Overfitting)")
    steps(
        [
            ("اجمع بيانات أكثر", "العلاج الأقوى والأكثر ضماناً على الإطلاق."),
            ("توليد بيانات اصطناعية", "Data augmentation: قلب، قصّ، تدوير للصور؛ "
                                      "إعادة صياغة للنصوص."),
            ("بسّط النموذج", "درجة أقل، عمق شجرة أقل، طبقات أقل، خصائص أقل."),
            ("أضف تنظيماً", "L1 / L2 / Elastic Net، وزيادة λ."),
            ("توقّف مبكّر", "راقب خسارة التحقق وتوقّف عند أول ارتفاع مستمر."),
            ("Dropout / Batch Norm", "خاص بالشبكات العصبية."),
            ("جمّع نماذج (Ensembling)", "Bagging و Random Forest يقلّلان التباين بالتوسيط."),
            ("قلّل الخصائص", "اختيار الخصائص أو تقليل الأبعاد (PCA)."),
        ],
        accent="rose",
    )
with o2:
    st.markdown("##### :material/trending_up: علاج التقصير (Underfitting)")
    steps(
        [
            ("زِد مرونة النموذج", "درجة أعلى، شجرة أعمق، طبقات أكثر."),
            ("هندسة خصائص أفضل", "تفاعلات، تحويلات لوغاريتمية، حدود تربيعية، "
                                  "خصائص مجالية من خبرتك."),
            ("درّب لمدة أطول", "أحياناً المشكلة ببساطة أن الحقب لم تكفِ."),
            ("قلّل التنظيم", "خفّض λ، أوقف Dropout."),
            ("غيّر عائلة النموذج", "من خطي إلى تعزيز تدرّجي أو شبكة عصبية."),
            ("راجع جودة البيانات", "خصائص غير مفيدة أصلاً؟ ضجيج في الـ labels؟"),
        ],
        accent="amber",
    )

st.subheader(":material/timer: التوقّف المبكّر (Early Stopping) — مجاني وفعّال")

ep2 = np.arange(1, 81)
tr_l = 1.5 * np.exp(-0.075 * ep2) + 0.07
va_l = 1.55 * np.exp(-0.085 * ep2) + 0.16 + 0.0052 * np.maximum(0, ep2 - 32) ** 1.55
best_ep = int(ep2[int(np.argmin(va_l))])

fige2 = go.Figure()
fige2.add_trace(go.Scatter(x=ep2, y=tr_l, mode="lines", name="خسارة التدريب",
                           line=dict(color=viz.C_BLUE, width=4),
                           hovertemplate="epoch %{x}<br>%{y:.3f}<extra></extra>"))
fige2.add_trace(go.Scatter(x=ep2, y=va_l, mode="lines", name="خسارة التحقق",
                           line=dict(color=viz.C_ROSE, width=4, dash="dash"),
                           hovertemplate="epoch %{x}<br>%{y:.3f}<extra></extra>"))
fige2.add_vline(x=best_ep, line=dict(color=viz.C_GREEN, width=3),
                annotation_text=f"هنا توقّف · الحقبة {best_ep}",
                annotation_position="top left",
                annotation_font=dict(size=12, color=viz.C_GREEN))
fige2.add_vrect(x0=best_ep, x1=80, fillcolor=viz.rgba(viz.C_ROSE, 0.09), line_width=0,
                annotation_text="منطقة الإفراط", annotation_position="top right",
                annotation_font=dict(size=11, color=viz.C_ROSE))
viz.style_fig(fige2, height=400, xtitle="الحقبة", ytitle="الخسارة",
              title="النقطة التي تبدأ عندها خسارة التحقق في الارتفاع")
st.plotly_chart(fige2, width="stretch", key="of_early")

st.markdown(
    f"""
الفكرة: احفظ نسخة من المعاملات عند **أفضل خسارة تحقق** (هنا الحقبة {best_ep})،
واستمر قليلاً للتأكّد (**patience**)، فإن لم يتحسّن شيء ارجع إلى تلك النسخة.

```python
best, patience, wait = float("inf"), 10, 0
for epoch in range(max_epochs):
    train_one_epoch()
    v = validation_loss()
    if v < best - 1e-4:
        best, wait = v, 0
        save_checkpoint()          # احفظ أفضل نسخة
    else:
        wait += 1
        if wait >= patience:
            load_checkpoint()      # ارجع إليها
            break
```
    """
)

rule()

# ======================================================================
# 8. ملخص
# ======================================================================
st.header(":material/summarize: ملخّص الدرس في سبع جمل")

st.markdown(
    """
1. **لا يوجد نموذج جيد بذاته** — يوجد نموذج مناسب لحجم بياناتك وتعقيد مشكلتك.
2. **الفجوة بين خطأ التدريب وخطأ التحقق هي مقياس الإفراط.** راقبها دائماً.
3. **خطأ التدريب المرتفع نفسه هو مقياس التقصير.** لا تعالجه بتنظيم.
4. **منحنى التحقق** (الخطأ مقابل التعقيد) يحدّد المرونة المثلى.
5. **منحنى التعلّم** (الخطأ مقابل حجم البيانات) يحدّد هل تنفع بيانات أكثر.
6. **التنظيم والتوقّف المبكّر** يسمحان بنموذج مرن دون إفراط.
7. **الخطأ غير القابل للإزالة موجود دائماً.** من يَعِدك بدقة 100٪ إمّا يخدعك
   أو عنده تسرّب بيانات.
    """
)

footer("الدرس التالي: معايير التقييم — بأي رقم نحكم على النموذج أصلاً؟")

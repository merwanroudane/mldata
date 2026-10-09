"""الدرس 2 — آليات التدريب عبر عائلات الخوارزميات (التدريب ليس شيئاً واحداً)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.style import card, equation, footer, hero, rule, steps, terms, verdict

hero(
    "هل تتدرّب كل الخوارزميات بالطريقة نفسها؟",
    "Lesson 2 · How different algorithm families actually train",
    "كثير من الدورات تشرح النزول التدرّجي ثم توحي بأنه «هو التدريب». "
    "هذا غير صحيح. هناك <b>عدّة آليات مختلفة تماماً</b> للتدريب، "
    "وبعضها لا يستعمل مشتقّة واحدة. هنا الصورة العامة الكاملة.",
)

st.warning(
    ":material/priority_high: **صحّح هذا المفهوم الآن:** التدريب **ليس** مرادفاً "
    "للنزول التدرّجي. النزول التدرّجي واحد من عدّة طرق فقط — "
    "وهو لا يُستعمل أصلاً في أشجار القرار ولا في KNN ولا في Naive Bayes، "
    "وهي من أكثر الخوارزميات استعمالاً."
)

# ======================================================================
# 1. الوصفة الكونية
# ======================================================================
st.header(":material/all_inclusive: الوصفة التي تجمع كل أنواع التدريب")

st.markdown(
    """
مهما اختلفت الخوارزمية، فإن أي تدريب في الدنيا يتكوّن من **ثلاثة مكوّنات**
فقط. إن عرفت الثلاثة لأي خوارزمية، فقد فهمت تدريبها:
    """
)

u1, u2, u3 = st.columns(3, gap="medium")
with u1:
    card(
        "1 · التمثيل — ما شكل الحل؟",
        "ما الذي نبحث عنه أصلاً؟ أوزان؟ شجرة؟ احتمالات؟ نقاط محفوظة؟ "
        "هذا يحدّده <b>اختيارك للخوارزمية</b>.",
        accent="blue",
        eng="Representation · the hypothesis space",
    )
with u2:
    card(
        "2 · المعيار — ما معنى «حل جيد»؟",
        "دالة خسارة؟ نقاء الفئات؟ أرجحية عظمى؟ أكبر هامش؟ "
        "هذا هو <b>المقياس الداخلي</b> الذي تُحسّنه الخوارزمية.",
        accent="amber",
        eng="Evaluation criterion · objective",
    )
with u3:
    card(
        "3 · البحث — كيف نجد الحل؟",
        "حل جبري مباشر؟ نزول تدرّجي؟ بحث جشع؟ عدّ وإحصاء؟ تخزين فقط؟ "
        "هذه هي <b>آلية التدريب</b> الفعلية.",
        accent="green",
        eng="Search / optimisation procedure",
    )

verdict(
    "لماذا هذا التقسيم مفيد لك عملياً؟",
    "لأنه يجعلك تقرأ أي خوارزمية جديدة في دقيقتين: اسأل فقط "
    "«ما الذي تبحث عنه؟ وبأي معيار تحكم؟ وكيف تبحث؟» "
    "— وستفهم تدريبها فوراً دون حفظ.",
    tone="violet",
)

rule()

# ======================================================================
# 2. العائلات
# ======================================================================
st.header(":material/account_tree: ست آليات تدريب مختلفة — مع ما يحدث داخل fit()")

fam_tabs = st.tabs(
    [
        ":material/function: حل مباشر",
        ":material/trending_down: تحسين تكراري",
        ":material/park: بناء جشع",
        ":material/pin_drop: تخزين فقط",
        ":material/percent: عدّ وإحصاء",
        ":material/layers: تجميع تتابعي",
    ]
)

# ------------------------------------------------------- 1. closed form
with fam_tabs[0]:
    st.subheader("1 · الحل المغلق (Closed-form / Analytical solution)")
    st.markdown(
        """
**الفكرة:** في حالات محظوظة يمكن حلّ مسألة التحسين **جبرياً دفعة واحدة**:
نشتقّ دالة التكلفة، نساويها بالصفر، ونحلّ المعادلة. **لا تكرار، ولا حقب، ولا معدّل تعلّم.**

**مَن يستعملها:** الانحدار الخطي (OLS)، تحليل المكوّنات الرئيسية (PCA)،
التحليل التمييزي الخطي (LDA).
        """
    )
    equation(
        r"\hat{\mathbf{w}}=\big(\mathbf{X}^{\top}\mathbf{X}\big)^{-1}\mathbf{X}^{\top}\mathbf{y}",
        "كل «التدريب» هنا هو تنفيذ هذا السطر. عمليات مصفوفات فقط، "
        "وتنتهي في أجزاء من الثانية على بيانات متوسّطة.",
    )

    st.markdown(
        """
| الميزة | العيب |
|---|---|
| حل **دقيق** لا تقريبي | يحتاج قلب مصفوفة: تكلفته تكبر مكعّبياً مع عدد الخصائص |
| لا معاملات فائقة للتحسين (لا η ولا حقب) | يفشل إذا كانت الخصائص مترابطة تماماً |
| سريع جداً على بيانات صغيرة ومتوسّطة | **غير متاح** لمعظم النماذج (الشبكات مثلاً) |
| نتيجة حتمية قابلة للتكرار 100٪ | يحتاج كل البيانات في الذاكرة دفعة واحدة |
        """
    )
    st.info(
        ":material/lightbulb: **متى تلاحظه؟** عندما تُشغّل `LinearRegression().fit()` "
        "فتنتهي فوراً بلا أي شريط تقدّم — لأنها لا تُكرّر شيئاً أصلاً."
    )

# -------------------------------------------------------- 2. iterative
with fam_tabs[1]:
    st.subheader("2 · التحسين التكراري بالتدرّج (Iterative gradient-based)")
    st.markdown(
        """
**الفكرة:** حين يستحيل الحل الجبري، نبدأ من تخمين ونُحسّنه خطوة خطوة.
هذه هي الآلية التي شرحناها في الدرس السابق بـ«المقابض».

**مَن يستعملها:** الانحدار اللوجستي، الشبكات العصبية بكل أنواعها،
SVM بصيغة SGD، أنظمة التوصية، النماذج اللغوية.
        """
    )

    it = np.arange(0, 60)
    loss_curve = 2.2 * np.exp(-0.085 * it) + 0.22
    figi = go.Figure()
    figi.add_trace(
        go.Scatter(x=it, y=loss_curve, mode="lines", name="الخسارة",
                   line=dict(color=viz.C_BLUE, width=4),
                   hovertemplate="تكرارة %{x}<br>%{y:.3f}<extra></extra>")
    )
    figi.add_trace(
        go.Scatter(x=it[::8], y=loss_curve[::8], mode="markers", name="تحديث للمقابض",
                   marker=dict(size=11, color=viz.C_ROSE,
                               line=dict(color="#fff", width=2)),
                   hovertemplate="تحديث<extra></extra>")
    )
    viz.style_fig(figi, height=300, xtitle="عدد التكرارات", ytitle="الخسارة",
                  title="التدريب هنا = سلسلة طويلة من التحديثات الصغيرة")
    st.plotly_chart(figi, width="stretch", key="fam_iter")

    st.markdown(
        """
| الميزة | العيب |
|---|---|
| يعمل مع **أي** نموذج قابل للاشتقاق | يحتاج ضبط معدّل التعلّم وعدد الحقب |
| يتعامل مع بيانات ضخمة (دفعة دفعة) | قد يعلق في حدّ أدنى محلّي (في النماذج غير المحدّبة) |
| يستفيد من بطاقات الرسومات بقوّة | نتيجته تعتمد على التهيئة والبذرة العشوائية |
| يسمح بالتدريب المتصل والتدريجي | يحتاج **تقييس الخصائص** وإلا تباطأ بشدّة |
        """
    )
    st.caption(
        "هذه العائلة هي موضوع الجزء الثاني بالتفصيل الكامل "
        "(التشريح الرياضي، دالة الخسارة، النزول التدرّجي، حلقة التدريب الحيّة)."
    )

# ----------------------------------------------------------- 3. greedy
with fam_tabs[2]:
    st.subheader("3 · البناء الجشع (Greedy construction) — أشجار القرار")
    st.markdown(
        """
**الفكرة مختلفة تماماً:** لا توجد أوزان ولا مقابض ولا مشتقّات. التدريب هنا هو
**بناء هيكل**: في كل خطوة تجرّب الخوارزمية **كل** نقاط القطع الممكنة على **كل**
الخصائص، وتختار القطع الذي يُنقص «الفوضى» أكثر، ثم تكرّر داخل كل فرع.

**مَن يستعملها:** أشجار القرار (CART, ID3, C4.5)، الغابات العشوائية،
وكل خوارزميات التعزيز بالأشجار (XGBoost, LightGBM, CatBoost).
        """
    )

    equation(
        r"\text{Gini}=1-\sum_{k}p_k^{2},\qquad"
        r"\text{Entropy}=-\sum_{k}p_k\log_2 p_k,\qquad"
        r"\text{Gain}=I_{\text{parent}}-\sum_{j}\frac{n_j}{n}I_{j}",
        "«معيار الجودة» هنا ليس خسارة تُشتقّ، بل <b>مقياس نقاء</b>: "
        "كم أصبحت المجموعات أكثر تجانساً بعد القطع؟",
    )

    rng = np.random.default_rng(8)
    xs = np.concatenate([rng.normal(2.2, 0.8, 30), rng.normal(6.4, 0.9, 30)])
    lbl = np.array([0] * 30 + [1] * 30)
    depth = st.slider("عمق الشجرة أثناء البناء", 0, 3, 1, 1, key="fam_depth")

    cuts = {0: [], 1: [4.3], 2: [2.6, 4.3, 6.1], 3: [1.7, 2.6, 3.4, 4.3, 5.3, 6.1, 7.0]}[depth]

    figt = go.Figure()
    figt.add_trace(
        go.Scatter(x=xs[lbl == 0], y=np.full(30, 0.35) + rng.normal(0, 0.03, 30),
                   mode="markers", name="الفئة أ",
                   marker=dict(size=12, color=viz.rgba(viz.C_BLUE, 0.8),
                               line=dict(color="#fff", width=1.5)),
                   hoverinfo="skip")
    )
    figt.add_trace(
        go.Scatter(x=xs[lbl == 1], y=np.full(30, 0.65) + rng.normal(0, 0.03, 30),
                   mode="markers", name="الفئة ب",
                   marker=dict(size=12, color=viz.rgba(viz.C_ROSE, 0.8),
                               line=dict(color="#fff", width=1.5)),
                   hoverinfo="skip")
    )
    for cpt in cuts:
        figt.add_vline(x=cpt, line=dict(color=viz.C_GREEN, width=3, dash="dash"))
    figt.update_yaxes(visible=False, range=[0.15, 0.85])
    viz.style_fig(
        figt, height=300, xtitle="قيمة الخصيصة", ytitle="",
        title=f"عمق {depth} ← {len(cuts)} قطع ← {len(cuts) + 1} ورقة",
    )
    st.plotly_chart(figt, width="stretch", key="fam_tree")

    if depth == 0:
        st.info("عمق 0 = لا قطع أصلاً. الشجرة تتنبّأ بالفئة الأغلبية للجميع.")
    elif depth == 1:
        st.success(
            "قطع واحد ذكي يكفي هنا تقريباً لفصل الفئتين! "
            "**لاحظ أن الخوارزمية اختارت مكان القطع بالبحث، لا بالاشتقاق.**"
        )
    else:
        st.warning(
            "كلما زاد العمق زادت القطوع، وبدأت الشجرة تفصل **نقاطاً فردية** — "
            "وهذا بالضبط هو **الإفراط في التعلّم** في عالم الأشجار. "
            "ولهذا يكون `max_depth` من أهم معاملاتها الفائقة."
        )

    st.markdown(
        """
| الميزة | العيب |
|---|---|
| **لا تحتاج تقييس الخصائص إطلاقاً** | تُفرط في التعلّم بسرعة إن تُركت بلا قيود |
| تتعامل مع الفئات النصّية والأرقام معاً | غير مستقرّة: تغيير بسيط في البيانات يغيّر الشجرة |
| قابلة للتفسير: ترى شروط القرار بعينك | لا تلتقط العلاقات الخطّية المائلة بكفاءة |
| سريعة ولا تحتاج معدّل تعلّم | جشعة: القطع الأمثل محلياً قد لا يكون أمثل عالمياً |
        """
    )

# ------------------------------------------------------------- 4. lazy
with fam_tabs[3]:
    st.subheader("4 · التخزين فقط (Lazy / Instance-based learning)")
    st.markdown(
        """
**الصدمة الجميلة:** في خوارزمية **الجيران الأقرب (KNN)**، التدريب هو
**حفظ البيانات في الذاكرة. لا شيء غير ذلك.**

لا مقابض تُضبط، ولا دالة تُصغَّر، ولا تكرارات. `fit()` فيها تعني حرفياً
«خزِّن الجدول». وكل العمل الحقيقي يُؤجَّل إلى لحظة التنبؤ: نحسب المسافة بين
النقطة الجديدة وكل النقاط المحفوظة، ونأخذ رأي الأقربين.
        """
    )

    rng2 = np.random.default_rng(14)
    ax_ = np.concatenate([rng2.normal(3, 1.0, 22), rng2.normal(7, 1.0, 22)])
    ay_ = np.concatenate([rng2.normal(3, 1.0, 22), rng2.normal(6, 1.0, 22)])
    cls = np.array([0] * 22 + [1] * 22)

    kn1, kn2, kn3 = st.columns(3, gap="medium")
    with kn1:
        kq = st.slider("عدد الجيران k", 1, 15, 5, 1, key="fam_k")
    with kn2:
        px = st.slider("موقع النقطة الجديدة — أفقياً", 0.0, 10.0, 5.0, 0.2, key="fam_px")
    with kn3:
        py = st.slider("موقع النقطة الجديدة — رأسياً", 0.0, 10.0, 4.5, 0.2, key="fam_py")

    d = np.sqrt((ax_ - px) ** 2 + (ay_ - py) ** 2)
    nearest = np.argsort(d)[:kq]
    vote = int(cls[nearest].sum() > kq / 2)
    radius = float(d[nearest].max())

    figk2 = go.Figure()
    figk2.add_shape(type="circle", x0=px - radius, x1=px + radius,
                    y0=py - radius, y1=py + radius,
                    line=dict(color=viz.rgba(viz.C_GREEN, 0.75), width=2, dash="dot"),
                    fillcolor=viz.rgba(viz.C_GREEN, 0.07))
    figk2.add_trace(
        go.Scatter(x=ax_[cls == 0], y=ay_[cls == 0], mode="markers", name="الفئة أ",
                   marker=dict(size=11, color=viz.rgba(viz.C_BLUE, 0.75),
                               line=dict(color="#fff", width=1.5)),
                   hoverinfo="skip")
    )
    figk2.add_trace(
        go.Scatter(x=ax_[cls == 1], y=ay_[cls == 1], mode="markers", name="الفئة ب",
                   marker=dict(size=11, color=viz.rgba(viz.C_ROSE, 0.75),
                               line=dict(color="#fff", width=1.5)),
                   hoverinfo="skip")
    )
    figk2.add_trace(
        go.Scatter(x=ax_[nearest], y=ay_[nearest], mode="markers", name=f"أقرب {kq} جيران",
                   marker=dict(size=17, color="rgba(0,0,0,0)",
                               line=dict(color=viz.C_GREEN, width=3)),
                   hoverinfo="skip")
    )
    figk2.add_trace(
        go.Scatter(x=[px], y=[py], mode="markers+text", name="النقطة الجديدة",
                   marker=dict(size=20, symbol="star",
                               color=viz.C_BLUE if vote == 0 else viz.C_ROSE,
                               line=dict(color="#17263F", width=2)),
                   text=[f"القرار: الفئة {'أ' if vote == 0 else 'ب'}"],
                   textposition="top center",
                   textfont=dict(size=12, color=viz.INK), hoverinfo="skip")
    )
    viz.style_fig(figk2, height=400, xtitle="الخصيصة 1", ytitle="الخصيصة 2",
                  title="لا يوجد «نموذج مُدرَّب» هنا — فقط بيانات محفوظة وقياس مسافات")
    st.plotly_chart(figk2, width="stretch", key="fam_knn")

    st.markdown(
        """
| الميزة | العيب |
|---|---|
| التدريب **فوري** (صفر حساب تقريباً) | التنبؤ **بطيء جداً** مع كِبَر البيانات |
| يلتقط حدوداً معقّدة جداً بلا أي افتراض | يحتاج تخزين كل البيانات إلى الأبد |
| سهل الفهم تماماً | **حسّاس جداً لمقياس الخصائص** — التقييس إلزامي |
| يضيف أمثلة جديدة بلا إعادة تدريب | ينهار في الأبعاد العالية (لعنة الأبعاد) |

:material/psychology: **فائدة مفاهيمية:** وجود خوارزمية ناجحة بلا «تدريب» بالمعنى
المعتاد يثبت أن **التدريب وسيلة لا غاية**. الغاية هي التنبؤ الجيد.
        """
    )

# -------------------------------------------------------- 5. counting
with fam_tabs[4]:
    st.subheader("5 · العدّ والإحصاء (Counting / Probabilistic estimation)")
    st.markdown(
        """
**الفكرة:** بعض النماذج تتدرّب بمجرّد **حساب تكرارات ومتوسطات** من البيانات —
ولا تحتاج تحسيناً ولا تكراراً. النتيجة احتمالات تُخزَّن في جدول.

**مَن يستعملها:** Naive Bayes بأنواعه (أشهر مصنّف للنصوص والرسائل المزعجة).
        """
    )
    equation(
        r"P(y\mid x_1,\dots,x_p)\;\propto\;P(y)\prod_{j=1}^{p}P(x_j\mid y)",
        "«التدريب» هو تقدير P(y) و P(x<sub>j</sub>|y) بالعدّ المباشر: "
        "كم مرة ظهرت هذه الكلمة في الرسائل المزعجة مقابل العادية؟",
    )

    demo = pd.DataFrame(
        {
            "الكلمة": ["مجاني", "فوز", "اجتماع", "تقرير", "اربح"],
            "تكرارها في المزعج": [180, 142, 6, 9, 165],
            "تكرارها في العادي": [12, 5, 210, 188, 3],
        }
    )
    demo["P(كلمة | مزعج)"] = (demo["تكرارها في المزعج"] /
                              demo["تكرارها في المزعج"].sum()).round(4)
    demo["P(كلمة | عادي)"] = (demo["تكرارها في العادي"] /
                              demo["تكرارها في العادي"].sum()).round(4)
    st.dataframe(demo, width="stretch", hide_index=True)
    st.caption(
        "هذا الجدول **هو** النموذج المُدرَّب. بناؤه لا يحتاج سوى مرور واحد "
        "على البيانات لعدّ التكرارات."
    )

    st.markdown(
        """
| الميزة | العيب |
|---|---|
| تدريب فائق السرعة: **مرور واحد فقط** | افتراض الاستقلال بين الخصائص غير واقعي غالباً |
| يعمل جيداً مع بيانات قليلة جداً | احتمالاته غير معايَرة (متطرّفة نحو 0 أو 1) |
| ممتاز مع النصوص وآلاف الخصائص | يحتاج تنعيماً (Laplace) للكلمات غير المرئية |
| يدعم التحديث التدريجي بسهولة | ضعيف حين تكون التفاعلات بين الخصائص مهمة |
        """
    )

# -------------------------------------------------------- 6. boosting
with fam_tabs[5]:
    st.subheader("6 · التجميع التتابعي (Boosting) — التدريب على الأخطاء")
    st.markdown(
        """
**فكرة عبقرية ومختلفة:** بدل تدريب نموذج قوي واحد، ندرّب **سلسلة من النماذج
الضعيفة**، كل واحد منها يتخصّص في **إصلاح أخطاء من سبقه**.

- النموذج الأول يتعلّم من البيانات.
- النموذج الثاني يتعلّم من **بواقي** الأول (ما فشل فيه).
- النموذج الثالث يتعلّم من بواقي الاثنين… وهكذا.
- التنبؤ النهائي = مجموع كل النماذج.

**مَن يستعملها:** Gradient Boosting · XGBoost · LightGBM · CatBoost · AdaBoost —
وهي **الأفضل عملياً على البيانات الجدولية** في معظم المسابقات والتطبيقات.
        """
    )

    n_stage = st.slider("عدد النماذج في السلسلة", 1, 25, 5, 1, key="fam_stage")

    rng3 = np.random.default_rng(21)
    xb = np.sort(rng3.uniform(0, 6.2, 70))
    yb = np.sin(1.3 * xb) + 0.2 * xb + rng3.normal(0, 0.18, 70)

    pred_b = np.full_like(yb, yb.mean())
    for _ in range(n_stage):
        resid = yb - pred_b
        # a depth-1 tree: single best split on x
        best_gain, best_cut = -1.0, None
        for cpt in np.quantile(xb, np.linspace(0.08, 0.92, 24)):
            left = xb <= cpt
            if left.sum() < 3 or (~left).sum() < 3:
                continue
            gain = (left.sum() * resid[left].mean() ** 2
                    + (~left).sum() * resid[~left].mean() ** 2)
            if gain > best_gain:
                best_gain, best_cut = gain, cpt
        if best_cut is None:
            break
        left = xb <= best_cut
        step = np.where(left, resid[left].mean(), resid[~left].mean())
        pred_b = pred_b + 0.55 * step

    figb = go.Figure()
    figb.add_trace(
        go.Scatter(x=xb, y=yb, mode="markers", name="البيانات",
                   marker=dict(size=8, color=viz.rgba(viz.C_BLUE, 0.6),
                               line=dict(color="#fff", width=1)),
                   hoverinfo="skip")
    )
    figb.add_trace(
        go.Scatter(x=xb, y=pred_b, mode="lines", name=f"مجموع {n_stage} نموذجاً",
                   line=dict(color=viz.C_VIOLET, width=4, shape="hv"),
                   hovertemplate="ŷ=%{y:.2f}<extra></extra>")
    )
    viz.style_fig(figb, height=360, xtitle="x", ytitle="y",
                  title="حرّك المؤشر: كل نموذج جديد يُصلح ما تبقّى من خطأ")
    st.plotly_chart(figb, width="stretch", key="fam_boost")

    st.metric("متوسط مربع الخطأ", f"{float(np.mean((yb - pred_b) ** 2)):.4f}")

    st.markdown(
        """
| Bagging (الغابة العشوائية) | Boosting (التعزيز) |
|---|---|
| نماذج **متوازية ومستقلة** | نماذج **تتابعية**، كل واحد يعتمد على سابقه |
| كل نموذج على عيّنة عشوائية | كل نموذج على **أخطاء** السابقين |
| يقلّل **التباين** | يقلّل **التحيّز** (والتباين مع التنظيم) |
| يصعب أن يُفرط في التعلّم | **يُفرط بسهولة** إن زادت المراحل |
| يمكن تدريبه بالتوازي | تدريب تتابعي بطبيعته |
        """
    )

rule()

# ======================================================================
# 3. الجدول الجامع
# ======================================================================
st.header(":material/table_view: الجدول الجامع — قارن كل العائلات في نظرة")

comp = pd.DataFrame(
    {
        "الخوارزمية": [
            "الانحدار الخطي (OLS)",
            "الانحدار اللوجستي",
            "شجرة القرار",
            "الغابة العشوائية",
            "التعزيز التدرّجي",
            "KNN",
            "Naive Bayes",
            "SVM",
            "الشبكة العصبية",
            "k-Means",
        ],
        "ما الذي يُضبط؟": [
            "أوزان",
            "أوزان",
            "شروط قطع وبنية شجرة",
            "مجموعة أشجار",
            "سلسلة أشجار + أوزانها",
            "لا شيء (تخزين)",
            "جداول احتمالات",
            "متجهات داعمة وهامش",
            "ملايين الأوزان",
            "مراكز العناقيد",
        ],
        "آلية التدريب": [
            "حل مغلق أو تدرّج",
            "تدرّج تكراري",
            "بحث جشع عن أفضل قطع",
            "جشع + عشوائية + توازٍ",
            "جشع تتابعي على البواقي",
            "حفظ البيانات",
            "عدّ تكرارات",
            "تحسين تربيعي مقيَّد",
            "تدرّج + انتشار عكسي",
            "تبادل: إسناد ثم تحديث",
        ],
        "حقب؟": ["لا", "نعم", "لا", "لا", "مراحل", "لا", "لا", "أحياناً", "نعم", "تكرارات"],
        "تقييس إلزامي؟": [
            "مفيد", "نعم", "لا", "لا", "لا", "**نعم**", "لا", "**نعم**", "**نعم**", "**نعم**",
        ],
        "الإشراف": [
            "مُراقَب", "مُراقَب", "مُراقَب", "مُراقَب", "مُراقَب",
            "مُراقَب", "مُراقَب", "مُراقَب", "مُراقَب", "**غير مُراقَب**",
        ],
    }
)
st.dataframe(comp, width="stretch", hide_index=True, height=400)

st.markdown(
    """
:material/visibility: **لاحظ العمودين الأهم:** «حقب؟» و«تقييس إلزامي؟».
نصف الخوارزميات الشائعة **لا حقب لها أصلاً**، ونصفها **لا تحتاج تقييساً**.
لو ظننت أن كل تدريب يحتاج `epochs` و `StandardScaler`، فستطبّقهما حيث لا فائدة لهما.
    """
)

rule()

# ======================================================================
# 4. تصنيفات أخرى للتدريب
# ======================================================================
st.header(":material/category: ثلاثة تصنيفات أخرى يجب أن تعرفها عن التدريب")

st.subheader("أ · حسب **متى** يتدرّب: دفعي أم متصل؟")

b1, b2 = st.columns(2, gap="medium")
with b1:
    card(
        "التدريب الدفعي / غير المتصل",
        "ندرّب على <b>كل البيانات دفعة واحدة</b> خارج الإنتاج، ثم نُجمّد النموذج "
        "وننشره. إن تغيّر الواقع نعيد التدريب من جديد ونستبدل النسخة.<br>"
        "<b>هذا هو الافتراضي في 95٪ من المشاريع.</b>",
        accent="blue",
        eng="Batch / Offline learning",
    )
with b2:
    card(
        "التدريب المتصل / التدريجي",
        "النموذج يُحدّث مقابضه <b>مع كل دفعة جديدة من البيانات</b> وهو يعمل. "
        "مفيد للبيانات المتدفّقة وللتكيّف السريع، لكنه خطر: "
        "بيانات سيئة واحدة قد تُفسد النموذج فوراً.",
        accent="amber",
        eng="Online / Incremental learning · partial_fit()",
    )

st.code(
    """
# دفعي: درّب مرة واحدة على كل شيء
model.fit(X_train, y_train)

# متصل: حدّث مع كل دفعة واردة (يدعمها SGDClassifier و MultinomialNB وغيرهما)
for X_batch, y_batch in stream:
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
    """,
    language="python",
)

st.subheader("ب · حسب **من أين** يبدأ: من الصفر أم من نموذج جاهز؟")

steps(
    [
        ("التدريب من الصفر",
         "نبدأ بمقابض عشوائية ونتعلّم كل شيء من بياناتنا. "
         "يحتاج <b>بيانات كثيرة ووقتاً طويلاً</b>، لكنه الخيار الوحيد "
         "حين تكون مشكلتك فريدة تماماً."),
        ("ما قبل التدريب",
         "يدرّب أحدهم نموذجاً ضخماً على بيانات هائلة عامة (ملايين الصور أو "
         "مليارات الكلمات) فيتعلّم <b>تمثيلات عامة</b> مفيدة للجميع."),
        ("التعلّم بالنقل",
         "نأخذ ذلك النموذج الجاهز ونستعمل طبقاته الأولى كما هي "
         "(<b>مُجمَّدة</b>)، ونُدرّب الطبقات الأخيرة فقط على بياناتنا الصغيرة."),
        ("الضبط الدقيق",
         "نُكمل تدريب النموذج الجاهز كله على بياناتنا بمعدّل تعلّم "
         "<b>صغير جداً</b>، حتى لا نمحو ما تعلّمه سابقاً."),
        ("الضبط الفعّال بالمعاملات",
         "نُجمّد كل النموذج ونُدرّب <b>عدداً صغيراً جداً</b> من المقابض الإضافية "
         "فقط (مثل LoRA). يوفّر ذاكرة ووقتاً بشكل هائل."),
    ],
    accent="violet",
)

st.success(
    ":material/savings: **قاعدة عملية مهمة:** إن كانت مشكلتك في الصور أو النصوص، "
    "**لا تدرّب من الصفر أبداً** إلا لسبب قوي. ابدأ دائماً من نموذج جاهز "
    "واضبطه على بياناتك — ستحصل على نتائج أفضل بآلاف الأمثلة بدل الملايين."
)

st.subheader("ج · حسب **نوع الإشراف**: ما الذي نُعطيه للنموذج؟")

st.markdown(
    """
| النوع | ماذا نُعطيه؟ | مثال | أين التدريب؟ |
|---|---|---|---|
| **مُراقَب** | X و y كاملة | تصنيف بريد، توقّع سعر | تصغير الفرق بين ŷ و y |
| **غير مُراقَب** | X فقط | تجميع العملاء، PCA | تحسين معيار داخلي (تماسك العناقيد) |
| **شبه مُراقَب** | قليل من y، كثير من X | تصنيف صور بوسوم قليلة | يستفيد من البنية في البيانات غير الموسومة |
| **ذاتي الإشراف** | X فقط، لكن نصنع y منها | إخفاء كلمة وتوقّعها | نفس التدريب المُراقَب لكن بإشراف مُصطنَع |
| **مُعزَّز** | مكافأة من بيئة | لعبة، روبوت، توصية | تعظيم المكافأة التراكمية |

:material/star: **ذاتي الإشراف هو سرّ النماذج اللغوية الحديثة:** نأخذ نصاً، نخفي
الكلمة التالية، ونطلب من النموذج توقّعها. الإشراف مجاني لأن الجواب موجود في النص نفسه.
    """
)

rule()

# ======================================================================
# 5. شجرة القرار
# ======================================================================
st.header(":material/route: أي آلية تدريب تناسب مشكلتك؟")

st.mermaid_chart(
    """
graph TD
    S["ما نوع بياناتك؟"] --> TAB["جدول (صفوف وأعمدة)"]
    S --> IMG["صور أو صوت"]
    S --> TXT["نصوص"]
    S --> SEQ["سلسلة زمنية"]

    TAB --> SM{"حجمها؟"}
    SM -->|"صغيرة < 1000"| L1["انحدار خطي/لوجستي منظَّم<br/>أو شجرة ضحلة"]
    SM -->|"متوسطة وكبيرة"| L2["تعزيز تدرّجي<br/>XGBoost / LightGBM"]

    IMG --> TL1["نموذج رؤية جاهز<br/>+ ضبط دقيق"]
    TXT --> TL2["نموذج لغوي جاهز<br/>+ ضبط دقيق أو تلقين"]
    SEQ --> TS["نماذج زمنية كلاسيكية<br/>أو تعزيز بخصائص زمنية"]

    L1 --> BASE["وقارن دائماً بنموذج أساس بسيط"]
    L2 --> BASE
    TL1 --> BASE
    TL2 --> BASE
    TS --> BASE

    style S fill:#EDF1F6,stroke:#8A94A6,stroke-width:2px,color:#17263F
    style TAB fill:#E3F0FE,stroke:#3B7DD8,color:#17263F
    style IMG fill:#EEE6FB,stroke:#9B6BD6,color:#17263F
    style TXT fill:#FDE6EE,stroke:#EB6F92,color:#17263F
    style SEQ fill:#FDF0DC,stroke:#F0A33C,color:#17263F
    style SM fill:#FFFFFF,stroke:#8A94A6,stroke-width:2px,color:#17263F
    style L1 fill:#DEF3F8,stroke:#2FA8C7,color:#17263F
    style L2 fill:#DEF3F8,stroke:#2FA8C7,color:#17263F
    style TL1 fill:#DEF3F8,stroke:#2FA8C7,color:#17263F
    style TL2 fill:#DEF3F8,stroke:#2FA8C7,color:#17263F
    style TS fill:#DEF3F8,stroke:#2FA8C7,color:#17263F
    style BASE fill:#DFF6EE,stroke:#2FB58B,stroke-width:3px,color:#17263F
    """
)

st.markdown(
    """
:material/code: **الجمال في كل هذا:** رغم اختلاف الآليات اختلافاً جذرياً،
فإن **واجهة الاستعمال واحدة** في scikit-learn. وهذا ليس صدفة، بل تصميم مقصود:
    """
)

st.code(
    """
from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.naive_bayes    import MultinomialNB
from sklearn.svm            import SVC

# آليات تدريب مختلفة تماماً في الداخل... وواجهة واحدة في الخارج
for model in [LogisticRegression(), DecisionTreeClassifier(),
              RandomForestClassifier(), GradientBoostingClassifier(),
              KNeighborsClassifier(), MultinomialNB(), SVC()]:
    model.fit(X_train, y_train)            # ← هنا تختلف الآليات جذرياً
    print(type(model).__name__, model.score(X_test, y_test))
    """,
    language="python",
)

st.caption(
    "ولهذا يستطيع المبتدئ تجريب سبع خوارزميات في سبعة أسطر — "
    "لكن من يفهم **ما يحدث داخل fit** هو من يعرف أيّها يختار ولماذا."
)

terms(
    [
        ("Closed-form", "حل مغلق"),
        ("Iterative", "تكراري"),
        ("Greedy search", "بحث جشع"),
        ("Lazy learning", "تعلّم كسول"),
        ("Batch learning", "تدريب دفعي"),
        ("Online learning", "تدريب متصل"),
        ("Transfer learning", "تعلّم بالنقل"),
        ("Fine-tuning", "ضبط دقيق"),
        ("Self-supervised", "ذاتي الإشراف"),
        ("Bagging", "تجميع متوازٍ"),
        ("Boosting", "تعزيز تتابعي"),
    ]
)

verdict(
    "خلاصة الدرس",
    "التدريب <b>فكرة عامة</b>: «اضبط النموذج على البيانات وفق معيار جودة». "
    "أمّا <b>طريقة</b> الضبط فتختلف جذرياً: جبر مباشر، أو نزول تدرّجي، أو بحث جشع، "
    "أو عدّ، أو حتى تخزين فقط. "
    "فلا تقل «التدريب = النزول التدرّجي» — بل قل «النزول التدرّجي طريقة تدريب».",
    tone="blue",
)

footer("الدرس التالي: دورة حياة التدريب الكاملة — قبله وأثناءه وبعده.")

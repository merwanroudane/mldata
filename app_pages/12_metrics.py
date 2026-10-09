"""الدرس 11 — معايير التقييم: الانحدار والتصنيف."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import viz
from utils.data import score_sample
from utils.style import card, equation, footer, hero, rule, terms, verdict

hero(
    "معايير التقييم — بأي رقم نحكم على النموذج؟",
    "Lesson 11 · Evaluation metrics for regression and classification",
    "دالة الخسارة هي ما يُحسّنه النموذج. أمّا المقياس فهو ما تُحاسَب عليه أنت. "
    "اختيار المقياس الخطأ يجعل نموذجاً فاشلاً يبدو ناجحاً — وهذا يحدث كل يوم.",
)

terms(
    [
        ("Metric", "مقياس"),
        ("Confusion matrix", "مصفوفة الالتباس"),
        ("Precision", "الدقة الموجبة"),
        ("Recall / Sensitivity", "الاستدعاء / الحساسية"),
        ("Specificity", "النوعية"),
        ("F1-score", "مقياس F1"),
        ("ROC-AUC", "المساحة تحت منحنى ROC"),
        ("Threshold", "عتبة القرار"),
    ]
)

main_tabs = st.tabs(
    [":material/show_chart: مقاييس الانحدار", ":material/category: مقاييس التصنيف"]
)

# ======================================================================
# الانحدار
# ======================================================================
with main_tabs[0]:
    st.header(":material/straighten: مقاييس الانحدار")

    st.markdown(
        "في الانحدار نقيس **مدى بُعد** التنبؤ عن الحقيقة. ستة مقاييس تغطّي كل الحالات:"
    )

    e1, e2 = st.columns(2, gap="medium")
    with e1:
        equation(
            r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}\big|y_i-\hat{y}_i\big|",
            "<b>متوسط الخطأ المطلق.</b> أسهل تفسيراً على الإطلاق: "
            "«أخطئ في المتوسط بـ 8 آلاف دينار». بوحدة الهدف، ومتين أمام الشذوذ.",
        )
        equation(
            r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(y_i-\hat{y}_i\big)^{2}}",
            "<b>الجذر التربيعي لمتوسط مربع الخطأ.</b> بوحدة الهدف أيضاً، "
            "لكنه يعاقب الأخطاء الكبيرة أكثر. دائماً RMSE ≥ MAE.",
        )
        equation(
            r"\mathrm{MAPE}=\frac{100}{n}\sum_{i=1}^{n}"
            r"\left|\frac{y_i-\hat{y}_i}{y_i}\right|",
            "<b>الخطأ النسبي المئوي.</b> مفيد للمقارنة بين مشاكل مختلفة الوحدات، "
            "لكنه <b>ينفجر إذا اقتربت y من الصفر</b> ولا يُستعمل مع قيم سالبة.",
        )
    with e2:
        equation(
            r"R^{2}=1-\frac{\sum_i\big(y_i-\hat{y}_i\big)^{2}}"
            r"{\sum_i\big(y_i-\bar{y}\big)^{2}}=1-\frac{\mathrm{SS}_{\text{res}}}"
            r"{\mathrm{SS}_{\text{tot}}}",
            "<b>معامل التحديد.</b> يقارن نموذجك بنموذج ساذج يتنبّأ دائماً بالمتوسط. "
            "R²=1 مثالي، R²=0 يعني «نموذجي يساوي المتوسط»، "
            "و<b>R² سالب يعني أنه أسوأ من المتوسط</b> (وارد جداً على بيانات الاختبار).",
        )
        equation(
            r"R^{2}_{\text{adj}}=1-\big(1-R^{2}\big)\frac{n-1}{n-p-1}",
            "<b>R² المعدّل.</b> يعاقب على إضافة خصائص بلا فائدة. "
            "استعمله عند مقارنة نماذج بأعداد خصائص مختلفة، فـ R² العادي "
            "لا ينخفض أبداً مهما أضفت أعمدة عشوائية.",
        )
        equation(
            r"\mathrm{MSLE}=\frac{1}{n}\sum_{i=1}^{n}"
            r"\Big(\log(1+y_i)-\log(1+\hat{y}_i)\Big)^{2}",
            "<b>خطأ لوغاريتمي.</b> للأهداف الموجبة المتفاوتة الحجم (مبيعات، مشاهدات): "
            "يهتم بالنسبة لا بالفرق المطلق، ويعاقب التقدير الأقل أكثر من الأعلى.",
        )

    st.subheader(":material/calculate: احسبها بنفسك على بيانات حيّة")

    rc1, rc2 = st.columns(2, gap="medium")
    with rc1:
        rnoise = st.slider("ضجيج البيانات σ", 0.5, 12.0, 4.0, 0.5, key="mt_noise")
    with rc2:
        bias_shift = st.slider("انحياز منتظم في النموذج", -8.0, 8.0, 0.0, 0.5,
                               key="mt_bias")

    rr = np.random.default_rng(4)
    nR = 160
    xR = rr.uniform(5, 95, nR)
    yR = 20 + 1.6 * xR + rr.normal(0, rnoise, nR)
    yhat = 20 + 1.6 * xR + bias_shift

    res = yR - yhat
    mae = float(np.mean(np.abs(res)))
    mse_v = float(np.mean(res**2))
    rmse = float(np.sqrt(mse_v))
    mape = float(np.mean(np.abs(res / yR)) * 100)
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((yR - yR.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    r2a = 1 - (1 - r2) * (nR - 1) / (nR - 1 - 1)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("MAE", f"{mae:,.2f}")
    g2.metric("RMSE", f"{rmse:,.2f}")
    g3.metric("MAPE", f"{mape:,.2f}٪")
    g4.metric("R²", f"{r2:,.3f}")
    g5.metric("R² المعدّل", f"{r2a:,.3f}")

    cL, cR = st.columns(2, gap="medium")
    with cL:
        figp = go.Figure()
        lim = [float(min(yR.min(), yhat.min())) - 5, float(max(yR.max(), yhat.max())) + 5]
        figp.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="الخط المثالي y = ŷ",
                                  line=dict(color=viz.rgba(viz.C_SLATE, 0.7), width=2,
                                            dash="dash"),
                                  hoverinfo="skip"))
        figp.add_trace(go.Scatter(x=yR, y=yhat, mode="markers", name="التنبؤات",
                                  marker=dict(size=8, color=viz.rgba(viz.C_BLUE, 0.7),
                                              line=dict(color="#fff", width=1)),
                                  hovertemplate="y=%{x:.1f}<br>ŷ=%{y:.1f}<extra></extra>"))
        viz.style_fig(figp, height=380, xtitle="y الحقيقي", ytitle="ŷ المتوقَّع",
                      title="مخطّط الحقيقي مقابل المتوقَّع")
        st.plotly_chart(figp, width="stretch", key="mt_pred")
    with cR:
        figh = go.Figure()
        figh.add_trace(go.Histogram(x=res, nbinsx=26,
                                    marker=dict(color=viz.rgba(viz.C_VIOLET, 0.65),
                                                line=dict(color="#fff", width=1)),
                                    name="البواقي",
                                    hovertemplate="الباقي=%{x:.2f}<extra></extra>"))
        figh.add_vline(x=0, line=dict(color=viz.C_GREEN, width=3))
        figh.add_vline(x=float(res.mean()),
                       line=dict(color=viz.C_ROSE, width=3, dash="dash"),
                       annotation_text=f"المتوسط={res.mean():+.2f}",
                       annotation_position="top right",
                       annotation_font=dict(size=11, color=viz.C_ROSE))
        viz.style_fig(figh, height=380, xtitle="الباقي  y − ŷ", ytitle="التكرار",
                      title="توزيع البواقي — يجب أن يتمركز حول الصفر", legend=False)
        st.plotly_chart(figh, width="stretch", key="mt_resid")

    if abs(bias_shift) > 1.0:
        st.warning(
            f"متوسط البواقي = **{res.mean():+.2f}** وليس صفراً. هذا **انحياز منتظم**: "
            "النموذج يخطئ في اتجاه واحد دائماً. "
            "لاحظ أن النقاط ابتعدت عن الخط المنقّط بالتوازي."
        )
    else:
        st.success("البواقي متمركزة حول الصفر — لا يوجد انحياز منتظم.")

    st.markdown(
        """
| المقياس | المدى | الأفضل | متى تستعمله |
|---|---|---|---|
| **MAE** | $[0,\\infty)$ | أصغر | تريد رقماً يفهمه غير المختص؛ توجد قيم شاذّة |
| **MSE** | $[0,\\infty)$ | أصغر | التحسين الرياضي (مشتقّة ناعمة) |
| **RMSE** | $[0,\\infty)$ | أصغر | الأخطاء الكبيرة مكلفة فعلاً؛ تريد وحدة الهدف |
| **MAPE** | $[0,\\infty)$ ٪ | أصغر | مقارنة مشاكل مختلفة الوحدات؛ $y$ بعيدة عن الصفر |
| **R²** | $(-\\infty,1]$ | أقرب لـ 1 | شرح نسبة التباين المُفسَّر؛ تقارير |
| **R² المعدّل** | $(-\\infty,1]$ | أقرب لـ 1 | مقارنة نماذج بأعداد خصائص مختلفة |

:material/error: **خطأ شائع:** مقارنة RMSE بين مشكلتين مختلفتين. RMSE = 5 ممتاز
لو كان $y$ بالآلاف، وكارثي لو كان $y$ بين 0 و1. **المقاييس المطلقة لا تُقارَن عبر المشاكل**،
واستعمل R² أو MAPE لذلك.
        """
    )

# ======================================================================
# التصنيف
# ======================================================================
with main_tabs[1]:
    st.header(":material/grid_on: مصفوفة الالتباس — أصل كل مقاييس التصنيف")

    st.markdown(
        """
في التصنيف الثنائي هناك **أربع** نتائج ممكنة فقط. كل مقاييس التصنيف
التي ستسمع بها في حياتك مشتقّة من هذه الأربعة:
        """
    )

    cc1, cc2 = st.columns(2, gap="medium")
    with cc1:
        card("TP — إيجابي صحيح",
             "توقّعنا «موجب» وكان موجباً فعلاً. ✅<br>"
             "<i>مثال: قلنا «مريض» وكان مريضاً.</i>",
             accent="green", eng="True Positive")
        card("FP — إيجابي كاذب (خطأ من النوع الأول)",
             "توقّعنا «موجب» وكان سالباً. ❌ <b>إنذار كاذب</b>.<br>"
             "<i>مثال: قلنا «مريض» وهو سليم ← قلق وفحوص بلا داع.</i>",
             accent="amber", eng="False Positive · Type I error")
    with cc2:
        card("TN — سلبي صحيح",
             "توقّعنا «سالب» وكان سالباً فعلاً. ✅<br>"
             "<i>مثال: قلنا «سليم» وهو سليم.</i>",
             accent="blue", eng="True Negative")
        card("FN — سلبي كاذب (خطأ من النوع الثاني)",
             "توقّعنا «سالب» وكان موجباً. ❌ <b>حالة فاتتنا</b>.<br>"
             "<i>مثال: قلنا «سليم» وهو مريض ← الأخطر في الطب.</i>",
             accent="rose", eng="False Negative · Type II error")

    st.subheader(":material/tune: مختبر العتبة — المصفوفة تتغيّر أمامك")

    st.markdown(
        """
النموذج يُخرج **احتمالاً**، لا فئة. تحويله إلى قرار يحتاج **عتبة (Threshold)**.
العتبة الافتراضية 0.5 **ليست مقدّسة** — بل هي معامل فائق تضبطه حسب تكلفة الأخطاء.
        """
    )

    th1, th2, th3 = st.columns(3, gap="medium")
    with th1:
        thr = st.slider("عتبة القرار", 0.05, 0.95, 0.50, 0.01, key="mt_thr")
    with th2:
        pos_rate = st.slider("نسبة الفئة الموجبة ٪", 2, 50, 35, 1, key="mt_pos")
    with th3:
        sep = st.slider("قوّة النموذج (الفصل بين الفئتين)", 0.2, 3.5, 1.6, 0.1,
                        key="mt_sep")

    df = score_sample(600, pos_rate / 100, sep, seed=3)
    y_true = df["y"].to_numpy()
    p = df["p"].to_numpy()
    y_pred = (p >= thr).astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))

    acc = (TP + TN) / len(y_true)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bal = 0.5 * (rec + spec)
    mcc_den = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = (TP * TN - FP * FN) / mcc_den if mcc_den > 0 else 0.0

    cm_left, cm_right = st.columns([1, 1.15], gap="medium")

    with cm_left:
        z = [[TN, FP], [FN, TP]]
        figcm = go.Figure(
            go.Heatmap(
                z=z,
                x=["توقَّع: سالب", "توقَّع: موجب"],
                y=["الحقيقة: سالب", "الحقيقة: موجب"],
                colorscale=viz.SEQ_LIGHT,
                showscale=False,
                text=[[f"TN<br><b>{TN}</b>", f"FP<br><b>{FP}</b>"],
                      [f"FN<br><b>{FN}</b>", f"TP<br><b>{TP}</b>"]],
                texttemplate="%{text}",
                textfont=dict(size=17, color=viz.INK),
                hovertemplate="%{y} · %{x}<br>العدد=%{z}<extra></extra>",
                xgap=4, ygap=4,
            )
        )
        viz.style_fig(figcm, height=360, title="مصفوفة الالتباس", legend=False)
        figcm.update_xaxes(showgrid=False, showline=False)
        figcm.update_yaxes(showgrid=False, showline=False)
        st.plotly_chart(figcm, width="stretch", key="mt_cm")

    with cm_right:
        figd = go.Figure()
        figd.add_trace(go.Histogram(x=p[y_true == 0], nbinsx=34, name="الفئة السالبة",
                                    marker=dict(color=viz.rgba(viz.C_BLUE, 0.55)),
                                    hovertemplate="p=%{x:.2f}<extra></extra>"))
        figd.add_trace(go.Histogram(x=p[y_true == 1], nbinsx=34, name="الفئة الموجبة",
                                    marker=dict(color=viz.rgba(viz.C_ROSE, 0.60)),
                                    hovertemplate="p=%{x:.2f}<extra></extra>"))
        figd.add_vline(x=thr, line=dict(color=viz.C_GREEN, width=4),
                       annotation_text=f"العتبة = {thr:.2f}",
                       annotation_position="top right",
                       annotation_font=dict(size=12, color=viz.C_GREEN))
        figd.update_layout(barmode="overlay")
        figd.update_traces(opacity=0.72)
        viz.style_fig(figd, height=360, xtitle="الاحتمال المتوقَّع p",
                      ytitle="عدد الأمثلة",
                      title="التداخل بين التوزيعين هو مصدر كل الأخطاء")
        st.plotly_chart(figd, width="stretch", key="mt_dist")

    q = st.columns(6)
    q[0].metric("الدقة Accuracy", f"{acc:.3f}")
    q[1].metric("Precision", f"{prec:.3f}")
    q[2].metric("Recall", f"{rec:.3f}")
    q[3].metric("Specificity", f"{spec:.3f}")
    q[4].metric("F1", f"{f1:.3f}")
    q[5].metric("MCC", f"{mcc:.3f}")

    st.markdown("##### الصيغ — كل واحدة هي نسبة بسيطة من المربعات الأربعة")

    fm1, fm2 = st.columns(2, gap="medium")
    with fm1:
        equation(
            r"\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}",
            "<b>الدقة.</b> نسبة الإجابات الصحيحة من الكل. "
            "بسيطة لكنها <b>خطيرة التضليل</b> عند اللاتوازن.",
        )
        equation(
            r"\text{Precision}=\frac{TP}{TP+FP}",
            "<b>الدقة الموجبة.</b> «من بين كل من قلت عنهم موجب، كم كان موجباً فعلاً؟» "
            "ارفعها عندما يكون <b>الإنذار الكاذب مكلفاً</b> "
            "(بريد مهم يذهب للمزعج، توقيف عميل بريء).",
        )
        equation(
            r"F_1=2\cdot\frac{\text{Precision}\times\text{Recall}}"
            r"{\text{Precision}+\text{Recall}}",
            "<b>المتوسط التوافقي</b> بين الدقة والاستدعاء. "
            "يُعاقب بشدة إذا كان أحدهما ضعيفاً — فلا يمكن خداعه برفع واحد فقط.",
        )
    with fm2:
        equation(
            r"\text{Recall}=\frac{TP}{TP+FN}",
            "<b>الاستدعاء / الحساسية.</b> «من بين كل الموجبين الحقيقيين، كم أمسكت؟» "
            "ارفعه عندما يكون <b>تفويت الحالة كارثياً</b> "
            "(تشخيص سرطان، كشف احتيال، أعطال طيران).",
        )
        equation(
            r"\text{Specificity}=\frac{TN}{TN+FP},\qquad"
            r"\text{Balanced Acc}=\frac{\text{Recall}+\text{Specificity}}{2}",
            "<b>النوعية</b> هي استدعاء الفئة السالبة. "
            "و<b>الدقة المتوازنة</b> متوسطهما — بديل أمين للدقة عند اللاتوازن.",
        )
        equation(
            r"\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}"
            r"{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}",
            "<b>معامل ماثيوز.</b> من −1 إلى +1، ويستعمل المربعات الأربعة كلها. "
            "يعتبره كثير من الباحثين <b>أنزه مقياس مفرد</b> للتصنيف الثنائي.",
        )

    st.subheader(":material/swap_horiz: المقايضة بين Precision و Recall")

    ths = np.linspace(0.02, 0.98, 100)
    P_, R_, F_ = [], [], []
    for t in ths:
        yp = (p >= t).astype(int)
        tp = np.sum((yp == 1) & (y_true == 1))
        fp = np.sum((yp == 1) & (y_true == 0))
        fn = np.sum((yp == 0) & (y_true == 1))
        pr = tp / (tp + fp) if (tp + fp) else 1.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        P_.append(pr)
        R_.append(rc)
        F_.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    best_t = float(ths[int(np.argmax(F_))])

    figpr = go.Figure()
    figpr.add_trace(go.Scatter(x=ths, y=P_, mode="lines", name="Precision",
                               line=dict(color=viz.C_BLUE, width=3.5),
                               hovertemplate="عتبة=%{x:.2f}<br>%{y:.3f}<extra></extra>"))
    figpr.add_trace(go.Scatter(x=ths, y=R_, mode="lines", name="Recall",
                               line=dict(color=viz.C_ROSE, width=3.5),
                               hovertemplate="عتبة=%{x:.2f}<br>%{y:.3f}<extra></extra>"))
    figpr.add_trace(go.Scatter(x=ths, y=F_, mode="lines", name="F1",
                               line=dict(color=viz.C_GREEN, width=4, dash="dot"),
                               hovertemplate="عتبة=%{x:.2f}<br>%{y:.3f}<extra></extra>"))
    figpr.add_vline(x=thr, line=dict(color=viz.rgba(viz.C_VIOLET, 0.8), width=3),
                    annotation_text="عتبتك", annotation_position="top left",
                    annotation_font=dict(size=11, color=viz.C_VIOLET))
    figpr.add_vline(x=best_t, line=dict(color=viz.C_AMBER, width=2, dash="dash"),
                    annotation_text=f"أفضل F1 عند {best_t:.2f}",
                    annotation_position="bottom right",
                    annotation_font=dict(size=11, color=viz.C_AMBER))
    viz.style_fig(figpr, height=390, xtitle="عتبة القرار", ytitle="قيمة المقياس",
                  title="رفع أحدهما يخفض الآخر — دائماً")
    st.plotly_chart(figpr, width="stretch", key="mt_pr")

    verdict(
        "القاعدة العملية لاختيار العتبة",
        "لا تترك 0.5 افتراضياً. اسأل: <b>أيّ الخطأين أغلى؟</b> "
        "تفويت مريض بالسرطان (FN) أغلى بكثير من فحص إضافي لسليم (FP) ← "
        "<b>اخفض العتبة</b> لترفع الاستدعاء. "
        "وحجب بريد مهم (FP) أسوأ من مرور رسالة مزعجة (FN) ← <b>ارفع العتبة</b>. "
        "واضبط العتبة على <b>بيانات التحقق</b>، لا على الاختبار.",
        tone="violet",
    )

    st.subheader(":material/area_chart: منحنى ROC والمساحة تحته (AUC)")

    st.markdown(
        """
بدل اختيار عتبة واحدة، **منحنى ROC** يرسم الأداء عند **كل** العتبات:
المحور الرأسي هو معدّل الإيجابي الصحيح (= Recall) والأفقي معدّل الإيجابي الكاذب
(= 1 − Specificity).

**AUC** هي المساحة تحت هذا المنحنى، ولها تفسير احتمالي جميل:
*احتمال أن يعطي النموذج درجة أعلى لمثال موجب عشوائي من مثال سالب عشوائي.*
        """
    )

    order = np.argsort(-p)
    ys = y_true[order]
    tpr = np.concatenate([[0], np.cumsum(ys) / max(ys.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / max((1 - ys).sum(), 1)])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(
        np.trapz(tpr, fpr)
    )

    cur_fpr = FP / (FP + TN) if (FP + TN) else 0.0
    cur_tpr = rec

    figroc = go.Figure()
    viz.soft_band(figroc, fpr, np.zeros_like(fpr), tpr, viz.C_BLUE, "AUC")
    figroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                name="تخمين عشوائي (AUC = 0.5)",
                                line=dict(color=viz.rgba(viz.C_SLATE, 0.7), width=2,
                                          dash="dash"),
                                hoverinfo="skip"))
    figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC · AUC = {auc:.3f}",
                                line=dict(color=viz.C_BLUE, width=4),
                                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>"))
    figroc.add_trace(go.Scatter(x=[cur_fpr], y=[cur_tpr], mode="markers+text",
                                name="عتبتك الحالية",
                                marker=dict(size=17, color=viz.C_GREEN,
                                            line=dict(color="#fff", width=3)),
                                text=[f"عتبة {thr:.2f}"], textposition="bottom right",
                                textfont=dict(size=11, color=viz.C_GREEN),
                                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>"))
    viz.style_fig(figroc, height=430, xtitle="معدّل الإيجابي الكاذب (1 − Specificity)",
                  ytitle="معدّل الإيجابي الصحيح (Recall)",
                  title="كلما اقترب المنحنى من الزاوية العليا اليسرى كان النموذج أفضل")
    st.plotly_chart(figroc, width="stretch", key="mt_roc")

    aa1, aa2, aa3 = st.columns(3)
    aa1.metric("AUC", f"{auc:.3f}")
    aa2.metric("نسبة الفئة الموجبة", f"{y_true.mean():.1%}")
    aa3.metric("دقة نموذج ساذج (يقول سالب دائماً)", f"{1 - y_true.mean():.1%}")

    st.markdown(
        """
| AUC | التفسير |
|---|---|
| 0.5 | عشوائي تماماً — بلا قيمة |
| 0.6 – 0.7 | ضعيف |
| 0.7 – 0.8 | مقبول |
| 0.8 – 0.9 | جيد |
| أكثر من 0.9 | ممتاز — **وراجع التسرّب إن تجاوز 0.99** |
        """
    )

    st.subheader(":material/warning: مفارقة الدقة — الفخّ الأشهر")

    if y_true.mean() < 0.15:
        st.error(
            f"**انتبه:** الفئة الموجبة تمثّل **{y_true.mean():.1%}** فقط. "
            f"نموذج أحمق يقول «سالب» للجميع يحصل على دقة **{1 - y_true.mean():.1%}** "
            "دون أن يتعلّم شيئاً! "
            "**لا تستعمل الدقة هنا** — استعمل F1 أو Balanced Accuracy أو MCC أو PR-AUC."
        )
    else:
        st.info(
            "خفّض «نسبة الفئة الموجبة» إلى 2٪ أو 5٪ أعلاه، وشاهد كيف تبقى الدقة "
            "عالية جداً بينما ينهار الاستدعاء. **هذه هي مفارقة الدقة.**"
        )

    st.markdown(
        """
> :material/format_quote: **المثال الكلاسيكي:** مرض يصيب 1 من كل 1000.
> نموذج يقول «سليم» دائماً يحقّق **دقة 99.9٪** — وهو عديم الفائدة تماماً،
> لأن استدعاءه **صفر**. الدقة وحدها كذبة مريحة.
        """
    )

    st.markdown(
        """
##### :material/fact_check: أي مقياس أختار؟

| الحالة | المقياس المناسب |
|---|---|
| فئات متوازنة وتكلفة الخطأين متساوية | **Accuracy** |
| فئات غير متوازنة | **F1** · **Balanced Accuracy** · **MCC** |
| الإنذار الكاذب مكلف (بريد مزعج، منع معاملة) | **Precision** عالية |
| تفويت الحالة كارثي (طب، احتيال، سلامة) | **Recall** عالٍ |
| تريد تقييماً مستقلاً عن العتبة | **ROC-AUC** |
| فئة موجبة نادرة جداً (< 5٪) | **PR-AUC** (أصدق من ROC-AUC هنا) |
| تحتاج احتمالات معايَرة (تسعير، مخاطر) | **Log Loss** · **Brier score** |
| تصنيف متعدّد الفئات | **Macro-F1** (كل فئة بوزن متساوٍ) أو **Weighted-F1** |
        """
    )

    st.code(
        """
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)

proba = pipe.predict_proba(X_test)[:, 1]      # احتمالات، لا فئات
pred  = (proba >= 0.35).astype(int)           # عتبتك المضبوطة على التحقق

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
print("ROC-AUC :", roc_auc_score(y_test, proba))
print("PR-AUC  :", average_precision_score(y_test, proba))
print("MCC     :", matthews_corrcoef(y_test, pred))
        """,
        language="python",
    )

rule()

st.header(":material/link: الربط بالدروس السابقة")

st.markdown(
    """
المقياس وحده لا يكفي — **يجب أن تذكر دائماً على أي مجموعة حسبته**:

| ما تكتبه | ماذا يعني فعلاً |
|---|---|
| «دقّتي 0.95» | :material/close: جملة ناقصة لا معنى لها |
| «دقة التدريب 0.95» | :material/close: رقم متفائل بطبيعته، لا يُعلَن وحده |
| «دقة التحقق 0.95 ± 0.02 عبر 5 طيّات» | :material/check: جيد — يصلح لاختيار النموذج |
| «دقة الاختبار 0.93 على 2,400 مثال محجوز» | :material/check: ممتاز — هذا هو الرقم النهائي |

وأضف دائماً **فترة الثقة**: مع 2,400 مثال ودقة 0.93، الفترة ≈ ±1 نقطة.
فرق نقطة واحدة بين نموذجين في هذه الحالة **ليس فرقاً حقيقياً**.
    """
)

verdict(
    "الخلاصة النهائية للدورة كلها",
    "<b>درّب</b> على مجموعة التدريب · <b>اختر</b> بمجموعة التحقق · "
    "<b>أعلن</b> على مجموعة الاختبار (مرة واحدة) · "
    "<b>راقب</b> الفجوة بين التدريب والتحقق لتشخيص الإفراط · "
    "و<b>اختر المقياس</b> الذي يعكس التكلفة الحقيقية لأخطائك، لا الأجمل رقماً.",
    tone="green",
)

footer("انتقل إلى «القاموس والخلاصة» لمراجعة كل المصطلحات والصيغ في صفحة واحدة.")

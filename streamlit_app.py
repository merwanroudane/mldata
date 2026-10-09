"""
تعلّم الآلة من الصفر — Machine Learning From Scratch
الجزء الأول: فكرة التدريب.  الجزء الثاني: تقسيم البيانات.

إعداد: Dr Merwan Roudane
"""

import streamlit as st

from utils.style import AUTHOR, inject_global_css, reset_counters

st.set_page_config(
    page_title="تعلّم الآلة من الصفر | Dr Merwan Roudane",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
reset_counters()

# ----------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------
if "visited" not in st.session_state:
    st.session_state.visited = set()

# ----------------------------------------------------------------------
# Navigation — rendered in the sidebar, which the stylesheet flips
# to the right-hand side of the screen (RTL layout).
# ----------------------------------------------------------------------
page = st.navigation(
    {
        "": [
            st.Page(
                "app_pages/01_intro.py",
                title="البداية · ما هو التعلّم؟",
                icon=":material/rocket_launch:",
                default=True,
            ),
        ],
        "الجزء الأول · التدريب للمبتدئين (General)": [
            st.Page(
                "app_pages/02_basics.py",
                title="التدريب ببساطة · بلا رياضيات",
                icon=":material/child_care:",
            ),
            st.Page(
                "app_pages/03_families.py",
                title="آليات التدريب عبر الخوارزميات",
                icon=":material/account_tree:",
            ),
            st.Page(
                "app_pages/04_lifecycle.py",
                title="دورة حياة التدريب",
                icon=":material/sync:",
            ),
        ],
        "الجزء الثاني · التدريب بالتفصيل (In depth)": [
            st.Page(
                "app_pages/05_training.py",
                title="التشريح الرياضي للتدريب",
                icon=":material/model_training:",
            ),
            st.Page(
                "app_pages/06_loss.py",
                title="دالة الخسارة",
                icon=":material/function:",
            ),
            st.Page(
                "app_pages/07_gradient.py",
                title="النزول التدرّجي",
                icon=":material/trending_down:",
            ),
            st.Page(
                "app_pages/08_loop.py",
                title="حلقة التدريب الحيّة",
                icon=":material/play_circle:",
            ),
        ],
        "الجزء الثالث · تقسيم البيانات (Splitting)": [
            st.Page(
                "app_pages/09_split.py",
                title="التدريب / التحقق / الاختبار",
                icon=":material/call_split:",
            ),
            st.Page(
                "app_pages/10_cv.py",
                title="التحقق المتقاطع",
                icon=":material/repeat:",
            ),
        ],
        "الجزء الرابع · التشخيص (Diagnosis)": [
            st.Page(
                "app_pages/11_fitting.py",
                title="الإفراط والتقصير في التعلّم",
                icon=":material/balance:",
            ),
            st.Page(
                "app_pages/12_metrics.py",
                title="معايير التقييم",
                icon=":material/speed:",
            ),
        ],
        "مراجع (Reference)": [
            st.Page(
                "app_pages/13_glossary.py",
                title="القاموس والخلاصة",
                icon=":material/menu_book:",
            ),
        ],
    },
    position="sidebar",
    expanded=True,  # 13 lessons: show them all instead of hiding the last few
)

# ----------------------------------------------------------------------
# Sidebar branding + reading guide
# ----------------------------------------------------------------------
with st.sidebar:
    st.html(
        f"""
        <div style="padding:14px 14px 12px;margin:2px 0 14px;border-radius:16px;
                    background:linear-gradient(135deg,#F3F9FF 0%,#EFFAF5 100%);
                    border:1px solid #CFE3F7;text-align:right;
                    animation:mr-rise .6s ease both;">
          <div style="font-weight:800;color:#1B3A5F;font-size:1.02rem;line-height:1.6;">
            تعلّم الآلة من الصفر
          </div>
          <div style="direction:ltr;text-align:right;font-size:.74rem;letter-spacing:.09em;
                      color:#6C86A6;font-weight:700;text-transform:uppercase;margin-top:3px;">
            Machine Learning · From Scratch
          </div>
          <div style="margin-top:10px;font-size:.84rem;color:#3C5572;line-height:1.75;">
            دورة تفاعلية للمبتدئين تماماً:<br>التدريب · تقسيم البيانات · التشخيص
          </div>
          <div style="margin-top:11px;padding:5px 12px;border-radius:999px;display:inline-block;
                      background:rgba(255,255,255,.9);border:1px solid #D2E3F6;
                      font-size:.82rem;font-weight:700;color:#2A62B4;">
            {AUTHOR}
          </div>
        </div>
        """
    )

    st.caption("اختر الدرس من القائمة أعلاه")

    with st.expander(":material/help: كيف أستعمل هذه الدورة؟"):
        st.markdown(
            """
- **اقرأ بالترتيب** من الأعلى إلى الأسفل — كل درس يبني على ما قبله.
- كل بطاقة فيها **الشرح العربي** + **المصطلح الإنجليزي** الذي ستقابله في الكتب.
- كل مخطط **تفاعلي**: حرّك المؤشرات واضغط زر ▶ لمشاهدة الحركة.
- الصيغ الرياضية مكتوبة، وتحتها دائماً ترجمة بالكلام العادي.
- لا تحتاج أي خلفية سابقة: نبدأ من الصفر فعلاً.
            """
        )

    st.html(
        """
        <div style="margin-top:8px;padding:10px 12px;border-radius:12px;
                    background:rgba(255,255,255,.72);border:1px dashed #CFE3F7;
                    font-size:.78rem;color:#5B7490;text-align:right;line-height:1.7;">
          ملاحظة: الشريط الجانبي على <b>اليمين</b> لأن الشرح بالعربية (RTL).
        </div>
        """
    )

page.run()

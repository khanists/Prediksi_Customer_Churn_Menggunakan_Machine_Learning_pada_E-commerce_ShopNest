import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor — ShopNest",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:      #0D1117;
    --bg-card:      #161B22;
    --bg-input:     #1C2333;
    --border:       #30363D;
    --border-focus: #0A9396;
    --teal:         #0A9396;
    --teal-light:   #2EC4C8;
    --coral:        #EE9B00;
    --red:          #CF4B4B;
    --green:        #2DA44E;
    --text-primary: #E6EDF3;
    --text-muted:   #7D8590;
    --text-dim:     #484F58;
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 860px !important;
}

/* ── Hero header ── */
.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    border: 1px solid var(--teal);
    border-radius: 100px;
    padding: 0.28rem 1rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.15;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: var(--teal-light);
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 300;
}
.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--coral));
    border-radius: 99px;
    margin: 1.4rem auto 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Form card ── */
.form-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}

/* ── Streamlit input overrides ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(10,147,150,0.2) !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(10,147,150,0.2) !important;
    outline: none !important;
}
label[data-testid="stWidgetLabel"] {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
}
div[data-baseweb="menu"] li {
    color: var(--text-primary) !important;
}
div[data-baseweb="menu"] li:hover {
    background-color: rgba(10,147,150,0.15) !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--teal) !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--teal-light) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--teal) 0%, #005F73 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--teal-light) 0%, var(--teal) 100%) !important;
    box-shadow: 0 0 24px rgba(10,147,150,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-churn {
    background: linear-gradient(135deg, rgba(207,75,75,0.15) 0%, rgba(207,75,75,0.05) 100%);
    border: 1px solid rgba(207,75,75,0.4);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-safe {
    background: linear-gradient(135deg, rgba(45,164,78,0.15) 0%, rgba(45,164,78,0.05) 100%);
    border: 1px solid rgba(45,164,78,0.4);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon {
    font-size: 3.2rem;
    margin-bottom: 0.6rem;
    line-height: 1;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.result-churn .result-label { color: #F87171; }
.result-safe  .result-label { color: #4ADE80; }
.result-desc {
    font-size: 0.88rem;
    color: var(--text-muted);
    font-weight: 300;
    margin-bottom: 1.4rem;
}

/* ── Probability bar ── */
.prob-wrapper {
    margin: 0 auto;
    max-width: 360px;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    font-family: 'DM Sans', sans-serif;
}
.prob-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill-churn {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #EF4444, #F97316);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.prob-bar-fill-safe {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #22C55E, #16A34A);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.prob-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 1rem 0 0.1rem;
    color: var(--text-primary);
}
.prob-pct-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

/* ── Feature summary pills ── */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.pill {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.75rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
}
.pill span {
    color: var(--text-primary);
    font-weight: 500;
}

/* ── Risk factors ── */
.risk-factors {
    margin-top: 1rem;
    text-align: left;
}
.risk-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}
.risk-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.75rem;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    margin-bottom: 0.35rem;
    font-size: 0.84rem;
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
}
.risk-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.risk-dot-red   { background: #EF4444; }
.risk-dot-amber { background: #F59E0B; }
.risk-dot-green { background: #22C55E; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("lgbm_best_model.pkl"), None
    except FileNotFoundError:
        return None, "⚠️ Model file lgbm_best_model.pkl not found."
    except Exception as e:
        return None, f"⚠️ Error loading model: {e}"

model, model_error = load_model()


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">ShopNest E-commerce</div>
    <h1 class="hero-title">Customer <span>Churn</span><br>Prediction</h1>
    <p class="hero-subtitle">LightGBM · Binary Classification · SHAP Explainability</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL STATUS BANNER
# ─────────────────────────────────────────────
if model_error:
    st.error(model_error)
    st.info("**Demo mode active.** The form is functional but predictions use a placeholder response until the model file is provided.")
else:
    st.success("✅ Model loaded successfully — **ROS | LightGBM Hypertuning**", icon=None)


# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)

with st.form("prediction_form"):

    # ── Block 1: Account & Engagement ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**📋 Account & Engagement**")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input(
            "Tenure (bulan)",
            min_value=0.0, max_value=120.0, value=12.0, step=0.5,
            help="Lama pelanggan menggunakan layanan ShopNest (dalam bulan)"
        )
        num_device = st.number_input(
            "Number of Device Registered",
            min_value=1, max_value=10, value=3, step=1,
            help="Jumlah perangkat yang terdaftar pada akun pelanggan"
        )
        satisfaction = st.number_input(
            "Satisfaction Score (1–5)",
            min_value=1, max_value=5, value=3, step=1,
            help="Skor kepuasan pelanggan dari 1 (sangat tidak puas) hingga 5 (sangat puas)"
        )

    with col2:
        warehouse_dist = st.number_input(
            "Warehouse to Home (km)",
            min_value=0.0, max_value=130.0, value=15.0, step=0.5,
            help="Jarak dari gudang ke rumah pelanggan (km)"
        )
        num_address = st.number_input(
            "Number of Address",
            min_value=1, max_value=20, value=2, step=1,
            help="Jumlah alamat pengiriman yang tersimpan di akun"
        )
        day_since_last = st.number_input(
            "Day Since Last Order",
            min_value=0.0, max_value=100.0, value=7.0, step=1.0,
            help="Jumlah hari sejak transaksi terakhir pelanggan"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Block 2: Preferences & Financial ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**🛒 Preferences & Financial**")

    col3, col4 = st.columns(2)
    with col3:
        preferred_cat = st.selectbox(
            "Preferred Order Category",
            options=["Mobile Phone", "Laptop & Accessory", "Fashion", "Grocery", "Others"],
            index=0,
            help="Kategori produk yang paling sering dipesan pelanggan"
        )
        cashback = st.number_input(
            "Cashback Amount (IDR)",
            min_value=0.0, max_value=400000.0, value=150000.0, step=5000.0,
            help="Total cashback yang diterima pelanggan"
        )

    with col4:
        marital_status = st.selectbox(
            "Marital Status",
            options=["Single", "Married", "Divorced"],
            index=1,
            help="Status pernikahan pelanggan"
        )
        complain = st.selectbox(
            "Pernah Komplain?",
            options=["Tidak (0)", "Ya (1)"],
            index=0,
            help="Apakah pelanggan pernah mengajukan komplain?"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ──
    submitted = st.form_submit_button("🔮  Predict Churn")


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if submitted:
    complain_val = 1 if complain.startswith("Ya") else 0

    input_data = pd.DataFrame([{
        "Tenure":                   tenure,
        "WarehouseToHome":          warehouse_dist,
        "NumberOfDeviceRegistered": num_device,
        "PreferedOrderCat":         preferred_cat,
        "SatisfactionScore":        satisfaction,
        "MaritalStatus":            marital_status,
        "NumberOfAddress":          num_address,
        "Complain":                 complain_val,
        "DaySinceLastOrder":        day_since_last,
        "CashbackAmount":           cashback,
    }])

    if model is not None:
        prediction   = model.predict(input_data)[0]
        proba        = model.predict_proba(input_data)[0]
        churn_prob   = proba[1]
        no_churn_prob = proba[0]
    else:
        # Demo mode fallback
        churn_prob    = 0.72 if complain_val == 1 or tenure < 6 else 0.28
        no_churn_prob = 1 - churn_prob
        prediction    = 1 if churn_prob >= 0.5 else 0

    churn_pct    = round(churn_prob    * 100, 1)
    no_churn_pct = round(no_churn_prob * 100, 1)

    # ── Result card ──
    st.markdown('<p class="section-label" style="margin-top:2rem;">Prediction Result</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-churn">
            <div class="result-icon">⚠️</div>
            <div class="result-label">HIGH CHURN RISK</div>
            <div class="result-desc">Pelanggan ini diprediksi akan meninggalkan platform</div>
            <div class="prob-wrapper">
                <div class="prob-pct">{churn_pct}%</div>
                <div class="prob-pct-label">Churn Probability</div>
                <div style="margin-top:1rem;">
                    <div class="prob-label-row"><span>Churn</span><span>{churn_pct}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-churn" style="width:{churn_pct}%"></div></div>
                </div>
                <div style="margin-top:0.6rem;">
                    <div class="prob-label-row"><span>Not Churn</span><span>{no_churn_pct}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-safe" style="width:{no_churn_pct}%"></div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">✅</div>
            <div class="result-label">LOW CHURN RISK</div>
            <div class="result-desc">Pelanggan ini diprediksi akan tetap aktif</div>
            <div class="prob-wrapper">
                <div class="prob-pct">{no_churn_pct}%</div>
                <div class="prob-pct-label">Retention Probability</div>
                <div style="margin-top:1rem;">
                    <div class="prob-label-row"><span>Not Churn</span><span>{no_churn_pct}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-safe" style="width:{no_churn_pct}%"></div></div>
                </div>
                <div style="margin-top:0.6rem;">
                    <div class="prob-label-row"><span>Churn</span><span>{churn_pct}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-churn" style="width:{churn_pct}%"></div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Risk Factor Analysis ──
    st.markdown('<p class="section-label" style="margin-top:1.8rem;">Risk Factor Analysis</p>', unsafe_allow_html=True)

    risk_items = []

    if tenure < 6:
        risk_items.append(("red",   f"Tenure sangat singkat ({tenure:.1f} bln) — pelanggan baru berisiko tinggi"))
    elif tenure < 12:
        risk_items.append(("amber", f"Tenure relatif singkat ({tenure:.1f} bln) — perlu perhatian"))
    else:
        risk_items.append(("green", f"Tenure cukup panjang ({tenure:.1f} bln) — indikator loyalitas"))

    if complain_val == 1:
        risk_items.append(("red",   "Pernah mengajukan komplain — faktor risiko utama churn"))
    else:
        risk_items.append(("green", "Tidak pernah komplain — sinyal kepuasan positif"))

    if cashback < 100000:
        risk_items.append(("amber", f"Cashback rendah (Rp {cashback:,.0f}) — kurang terapresiasi"))
    elif cashback >= 200000:
        risk_items.append(("green", f"Cashback tinggi (Rp {cashback:,.0f}) — insentif retensi kuat"))
    else:
        risk_items.append(("amber", f"Cashback sedang (Rp {cashback:,.0f})"))

    if day_since_last > 20:
        risk_items.append(("red",   f"Inaktif selama {day_since_last:.0f} hari — sinyal meninggalkan platform"))
    elif day_since_last > 10:
        risk_items.append(("amber", f"Belum belanja {day_since_last:.0f} hari — pantau lebih lanjut"))
    else:
        risk_items.append(("green", f"Belanja baru-baru ini ({day_since_last:.0f} hari lalu) — pelanggan aktif"))

    if satisfaction <= 2:
        risk_items.append(("red",   f"Skor kepuasan rendah ({satisfaction}/5) — perlu tindak lanjut segera"))
    elif satisfaction == 3:
        risk_items.append(("amber", f"Skor kepuasan netral ({satisfaction}/5)"))
    else:
        risk_items.append(("green", f"Skor kepuasan tinggi ({satisfaction}/5)"))

    items_html = ""
    for color, text in risk_items:
        items_html += f'<div class="risk-item"><div class="risk-dot risk-dot-{color}"></div>{text}</div>'

    st.markdown(f"""
    <div class="form-card" style="margin-top:0.5rem;">
        <div class="risk-title">Faktor-faktor yang mempengaruhi prediksi</div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Input Summary Pills ──
    st.markdown(f"""
    <div style="margin-top:1rem;">
        <div class="risk-title">Input Summary</div>
        <div class="pill-row">
            <div class="pill">Tenure <span>{tenure:.1f} bln</span></div>
            <div class="pill">Warehouse <span>{warehouse_dist:.1f} km</span></div>
            <div class="pill">Devices <span>{num_device}</span></div>
            <div class="pill">Category <span>{preferred_cat}</span></div>
            <div class="pill">Satisfaction <span>{satisfaction}/5</span></div>
            <div class="pill">Status <span>{marital_status}</span></div>
            <div class="pill">Addresses <span>{num_address}</span></div>
            <div class="pill">Complain <span>{'Yes' if complain_val else 'No'}</span></div>
            <div class="pill">Last Order <span>{day_since_last:.0f}d ago</span></div>
            <div class="pill">Cashback <span>Rp {cashback:,.0f}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CRM Action Recommendation ──
    if prediction == 1:
        st.markdown("""
        <div class="form-card" style="border-color:rgba(207,75,75,0.3); margin-top:1.2rem;">
            <div class="risk-title">💡 Rekomendasi Tim CRM</div>
            <div class="risk-item"><div class="risk-dot risk-dot-red"></div>Kirim <strong>voucher cashback</strong> eksklusif dalam 24 jam ke depan</div>
            <div class="risk-item"><div class="risk-dot risk-dot-red"></div>Hubungi via <strong>push notification / email</strong> dengan penawaran personal</div>
            <div class="risk-item"><div class="risk-dot risk-dot-amber"></div>Evaluasi dan <strong>tindak lanjuti komplain</strong> yang pernah diajukan</div>
            <div class="risk-item"><div class="risk-dot risk-dot-amber"></div>Masukkan ke program <strong>Winback Campaign</strong> bulan ini</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="form-card" style="border-color:rgba(45,164,78,0.3); margin-top:1.2rem;">
            <div class="risk-title">💡 Rekomendasi Tim CRM</div>
            <div class="risk-item"><div class="risk-dot risk-dot-green"></div>Pelanggan dalam kondisi <strong>sehat</strong> — pertahankan dengan program loyalitas</div>
            <div class="risk-item"><div class="risk-dot risk-dot-green"></div>Pantau secara berkala (review setiap <strong>30 hari</strong>)</div>
            <div class="risk-item"><div class="risk-dot risk-dot-amber"></div>Pertimbangkan upsell / cross-sell sesuai kategori preferensi</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    ShopNest Analytics · Customer Churn Prediction · ROS | LightGBM Hypertuning<br>
    Model target: Recall ≥ 80% · PR-AUC ≥ 0.70 · Threshold-optimized
</div>
""", unsafe_allow_html=True)

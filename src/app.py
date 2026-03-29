import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="CyberXAI — Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; color: white; }
    .title-box {
        background: linear-gradient(135deg, #0A9396, #0D1B2A);
        padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 2rem;
    }
    .result-attack {
        background: #E63946; color: white;
        padding: 1.5rem; border-radius: 10px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .result-safe {
        background: #2DC653; color: white;
        padding: 1.5rem; border-radius: 10px;
        text-align: center; font-size: 1.5rem; font-weight: bold;
    }
    .metric-box {
        background: #112236; padding: 1rem;
        border-radius: 8px; border-left: 4px solid #0A9396;
        margin: 0.5rem 0;
    }
    .input-tip {
        background: #112236; padding: 0.8rem;
        border-radius: 8px; border-left: 4px solid #E9C46A;
        margin: 0.8rem 0; font-size: 0.85rem; color: #94D2BD;
    }
</style>
""", unsafe_allow_html=True)

# ─── TITLE ─────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🛡️ CyberXAI — Explainable Threat Detection</h1>
    <p style="color:#94D2BD; font-size:1.1rem;">
        An Explainable AI Framework for Network Intrusion & Phishing Detection
    </p>
    <p style="color:#64748B; font-size:0.9rem;">
        RF + XGBoost Ensemble · SHAP Explainability · 99.95% F1 Score
    </p>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    base = r'C:\Users\Kshaunish\cyber-xai-ml-project\models\saved_models'
    try:
        return {
            'rf_int':  joblib.load(os.path.join(base, 'rf_intrusion.pkl')),
            'rf_phi':  joblib.load(os.path.join(base, 'rf_phishing.pkl')),
            'ens_int': joblib.load(os.path.join(base, 'ensemble_intrusion.pkl')),
            'ens_phi': joblib.load(os.path.join(base, 'ensemble_phishing.pkl')),
        }, True
    except:
        return {}, False

models, loaded = load_models()
if not loaded:
    st.error("⚠️ Models not found! Run the training notebook first.")
    st.stop()

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.selectbox("Detection Mode", ["🔴 Network Intrusion", "🟠 Phishing Website"])

    st.markdown("---")
    st.markdown("### 🖊️ Input Method")
    input_mode = st.radio("Choose how to enter values:",
                          ["🎚️ Sliders", "⌨️ Type Values", "🎚️ + ⌨️ Both"])

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="metric-box">
        <b>Model:</b> RF + XGBoost Ensemble<br>
        <b>Intrusion F1:</b> 99.95%<br>
        <b>Phishing F1:</b> 100.00%<br>
        <b>Explainability:</b> SHAP TreeExplainer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    if "Intrusion" in mode:
        for c, e in [("BENIGN","🟢"),("DoS GoldenEye","🔴"),("DoS Hulk","🔴"),
                     ("DoS Slowhttptest","🔴"),("DoS slowloris","🔴"),("Heartbleed","🔴")]:
            st.markdown(f"{e} {c}")
    else:
        st.markdown("🟢 Legitimate Website")
        st.markdown("🔴 Phishing Website")

# ─── SHAP HELPER ───────────────────────────────────────────
def show_shap(model, X, prediction, feature_names, pos_label, neg_label):
    try:
        import shap, pandas as pd
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        sv_arr = np.array(shap_vals)
        sv = sv_arr[:, :, prediction][0] if len(sv_arr.shape) == 3 else shap_vals[prediction][0]
        top_idx = np.argsort(np.abs(sv))[::-1][:8]
        df = pd.DataFrame({
            'Feature':    [feature_names[i] for i in top_idx],
            'SHAP Value': [round(float(sv[i]), 4) for i in top_idx],
            'Impact':     [f"⬆️ {pos_label}" if sv[i] > 0 else f"⬇️ {neg_label}" for i in top_idx]
        })
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(df.style.format({'SHAP Value': '{:.4f}'}), use_container_width=True)
        with c2:
            st.markdown(f"""
            **How to read this table:**
            - ⬆️ pushed prediction → **{pos_label}**
            - ⬇️ pushed prediction → **{neg_label}**
            - Larger absolute value = stronger influence
            - This is the **XAI novelty** of our framework!
            """)
    except Exception as e:
        st.info(f"SHAP explanation: {e}")

# ─── FIELD RENDERER ────────────────────────────────────────
def render_field(label, mn, mx, default, key_prefix, col, show_slider, show_text):
    with col:
        slider_val = default
        typed_val  = ""

        if show_slider:
            slider_val = st.slider(label, mn, mx, default, key=f"s_{key_prefix}")

        if show_text:
            placeholder = f"default: {default}" if not show_slider else f"override slider (default: {default})"
            typed_val = st.text_input(
                f"{'Or type: ' if show_slider else ''}{label.split('(')[0].strip()}",
                value="", placeholder=placeholder, key=f"t_{key_prefix}"
            )

        # Typed takes priority if filled
        try:
            val = float(typed_val) if typed_val.strip() != "" else slider_val
            val = max(mn, min(mx, val))
        except:
            val = slider_val

        return val

# ═══════════════════════════════════════════════════════════
# INTRUSION MODE
# ═══════════════════════════════════════════════════════════
if "Intrusion" in mode:
    st.markdown("## 🔴 Network Intrusion Detection")

    show_slider = input_mode in ["🎚️ Sliders", "🎚️ + ⌨️ Both"]
    show_text   = input_mode in ["⌨️ Type Values", "🎚️ + ⌨️ Both"]

    if show_text and not show_slider:
        st.markdown("""
        <div class="input-tip">
        💡 Type network traffic feature values below. Leave blank to use defaults.
        </div>
        """, unsafe_allow_html=True)
    elif show_slider and show_text:
        st.markdown("""
        <div class="input-tip">
        💡 Use sliders to set values, or type in the box below each slider to override it precisely.
        </div>
        """, unsafe_allow_html=True)

    int_fields = [
        ("Flow Duration (ms)",      0, 120000,  50000),
        ("Total Fwd Packets",       0, 500,     100),
        ("Total Bwd Packets",       0, 500,     80),
        ("Fwd Packet Length Max",   0, 65535,   1400),
        ("Fwd Packet Length Mean",  0, 65535,   700),
        ("Flow Bytes/s",            0, 1000000, 50000),
        ("Flow Packets/s",          0, 10000,   500),
        ("Flow IAT Mean (ms)",      0, 100000,  1000),
        ("Flow IAT Std",            0, 100000,  500),
        ("Flow IAT Max (ms)",       0, 200000,  5000),
        ("Fwd IAT Total",           0, 200000,  10000),
        ("Bwd IAT Total",           0, 200000,  8000),
        ("Active Mean",             0, 100000,  1000),
        ("Idle Mean",               0, 5000000, 100000),
        ("PSH Flag Count",          0, 10,      2),
    ]

    cols = st.columns(3)
    vals = []
    for i, (label, mn, mx, default) in enumerate(int_fields):
        v = render_field(label, mn, mx, default, f"int_{i}", cols[i % 3], show_slider, show_text)
        vals.append(v)

    # Build 78-feature vector
    feat = [
        0, vals[0], vals[1], vals[2],
        vals[1]*vals[4], vals[2]*700,
        vals[3], 0, vals[4], vals[4]*0.3,
        1400, 0, 700, 200,
        vals[5], vals[6],
        vals[7], vals[8], vals[9], 0,
        vals[10], vals[7], vals[8], vals[9], 0,
        vals[11], vals[7], vals[8], vals[9], 0,
        0, vals[14], 0, 0, 0, 0, 0,
        100, 50, 100, 50, 200, 100,
        vals[12], 500, 2000, 100,
        vals[13], 500000, 2000000, 0,
    ]
    while len(feat) < 78:
        feat.append(0)
    feat = feat[:78]

    st.markdown("")
    if st.button("🔍  Predict Network Traffic", use_container_width=True):
        with st.spinner("Analyzing traffic patterns..."):
            X = np.array(feat).reshape(1, -1)
            pred  = models['ens_int'].predict(X)[0]
            proba = models['ens_int'].predict_proba(X)[0]
            conf  = proba[pred] * 100
            lmap  = {0:"BENIGN", 1:"DoS GoldenEye", 2:"DoS Hulk",
                     3:"DoS Slowhttptest", 4:"DoS slowloris", 5:"Heartbleed"}
            label = lmap[pred]

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                css_class = "result-safe" if label == "BENIGN" else "result-attack"
                icon = "✅" if label == "BENIGN" else "⚠️"
                msg  = "BENIGN Traffic — No Threat Detected" if label == "BENIGN" else f"ATTACK DETECTED: {label}"
                st.markdown(f'<div class="{css_class}">{icon} {msg}</div>', unsafe_allow_html=True)
            with c2:
                st.metric("Confidence", f"{conf:.2f}%")
                st.metric("Predicted Class", label)

            st.markdown("---")
            st.markdown("### 🔍 SHAP Explanation — Why this prediction?")
            fnames = [
                'Destination Port','Flow Duration','Total Fwd Packets','Total Bwd Packets',
                'Total Length Fwd','Total Length Bwd','Fwd Pkt Len Max','Fwd Pkt Len Min',
                'Fwd Pkt Len Mean','Fwd Pkt Len Std','Bwd Pkt Len Max','Bwd Pkt Len Min',
                'Bwd Pkt Len Mean','Bwd Pkt Len Std','Flow Bytes/s','Flow Packets/s',
                'Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
                'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min',
                'Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min',
                'Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
                'Fwd Header Len','Bwd Header Len','Fwd Packets/s','Bwd Packets/s',
                'Pkt Len Min','Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var',
                'FIN Flag','SYN Flag','RST Flag','PSH Flag','ACK Flag','URG Flag','CWE Flag',
                'ECE Flag','Down/Up Ratio','Avg Pkt Size','Avg Fwd Seg','Avg Bwd Seg',
                'Fwd Blk Rate','Fwd Seg Size','Bwd Blk Rate','Bwd Seg Size',
                'Subflow Fwd Pkts','Subflow Fwd Bytes','Subflow Bwd Pkts','Subflow Bwd Bytes',
                'Init Fwd Win','Init Bwd Win','Fwd Act Data Pkts','Fwd Seg Min',
                'Active Mean','Active Std','Active Max','Active Min',
                'Idle Mean','Idle Std','Idle Max','Idle Min',
            ][:78]
            show_shap(models['rf_int'], X, pred, fnames, "Towards Attack", "Towards Benign")

# ═══════════════════════════════════════════════════════════
# PHISHING MODE
# ═══════════════════════════════════════════════════════════
else:
    st.markdown("## 🟠 Phishing Website Detection")

    show_slider = input_mode in ["🎚️ Sliders", "🎚️ + ⌨️ Both"]
    show_text   = input_mode in ["⌨️ Type Values", "🎚️ + ⌨️ Both"]

    if show_text and not show_slider:
        st.markdown("""
        <div class="input-tip">
        💡 Type website/URL feature values below. Leave blank to use defaults.
        </div>
        """, unsafe_allow_html=True)
    elif show_slider and show_text:
        st.markdown("""
        <div class="input-tip">
        💡 Use sliders OR type in the box below each slider to set an exact value.
        </div>
        """, unsafe_allow_html=True)

    phi_fields = [
        ("URL Length",              0, 500,  75),
        ("Number of Dots in URL",   0, 20,   3),
        ("Number of Hyphens",       0, 20,   1),
        ("Number of Digits",        0, 50,   5),
        ("Special Characters",      0, 30,   4),
        ("Domain Length",           0, 100,  20),
        ("Number of Subdomains",    0, 10,   1),
        ("Path Length",             0, 200,  30),
        ("Title Length",            0, 200,  50),
        ("Number of Links",         0, 200,  20),
        ("Number of Images",        0, 100,  10),
    ]

    cols = st.columns(3)
    vals = []
    for i, (label, mn, mx, default) in enumerate(phi_fields):
        v = render_field(label, mn, mx, default, f"phi_{i}", cols[i % 3], show_slider, show_text)
        vals.append(v)

    st.markdown("**🔘 Boolean Features:**")
    bcols = st.columns(4)
    bool_fields = [
        ("Has HTTPS?",       "has_https",    1),
        ("Has IP Address?",  "has_ip",       0),
        ("Has Login Form?",  "has_form",     0),
        ("Has Copyright?",   "has_copyright",1),
    ]
    bvals = {}
    for i, (label, key, default) in enumerate(bool_fields):
        with bcols[i]:
            opts = [1, 0] if default == 1 else [0, 1]
            bvals[key] = st.selectbox(label, opts,
                format_func=lambda x: "Yes ✅" if x else "No ❌", key=f"b_{key}")

    # Build 50-feature vector
    phi_vec = [
        vals[0], vals[1], vals[2], vals[3], vals[4],
        vals[5], bvals['has_https'], bvals['has_ip'],
        vals[6], vals[7], vals[8], bvals['has_form'],
        vals[9], vals[10], bvals['has_copyright'],
        vals[0] / max(vals[5], 1),
        vals[3] / max(vals[0], 1),
        1 if vals[2] > 3 else 0,
        1 if vals[1] > 5 else 0,
        1 if vals[0] > 100 else 0,
    ]
    while len(phi_vec) < 50:
        phi_vec.append(0)
    phi_vec = phi_vec[:50]

    st.markdown("")
    if st.button("🔍  Predict Website", use_container_width=True):
        with st.spinner("Analyzing website features..."):
            X = np.array(phi_vec).reshape(1, -1)
            pred  = models['ens_phi'].predict(X)[0]
            proba = models['ens_phi'].predict_proba(X)[0]
            conf  = proba[pred] * 100

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                if pred == 0:
                    st.markdown('<div class="result-safe">✅ LEGITIMATE Website — Safe to Visit</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-attack">⚠️ PHISHING DETECTED — Do Not Visit!</div>', unsafe_allow_html=True)
            with c2:
                st.metric("Confidence", f"{conf:.2f}%")
                st.metric("Result", "Phishing 🔴" if pred == 1 else "Legitimate 🟢")

            st.markdown("---")
            st.markdown("### 🔍 SHAP Explanation — Why this prediction?")
            phi_fnames = [
                'URL Length','Num Dots','Num Hyphens','Num Digits','Num Special',
                'Domain Length','Has HTTPS','Has IP','Subdomain Count','Path Length',
                'Title Length','Has Form','Num Links','Num Images','Has Copyright',
                'URL/Domain Ratio','Digit Ratio','Hyphen Flag','Dot Flag','Long URL Flag',
            ] + [f'Feature_{i}' for i in range(30)]
            show_shap(models['rf_phi'], X, pred, phi_fnames, "Towards Phishing", "Towards Legitimate")

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.85rem;">
    🛡️ CyberXAI Framework · RF+XGBoost Ensemble · SHAP Explainability · ML Lab Research Project
</div>
""", unsafe_allow_html=True)
"""
╔══════════════════════════════════════════════════════════════╗
║   CHURN RADAR — Customer Analytics & ML Prediction Dashboard  ║
║   Run: streamlit run churn_analytics_app.py                   ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Cyber Aesthetic
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg:          #080C14;
    --surface:     #0D1422;
    --card:        #111927;
    --border:      #1E2D45;
    --accent:      #00E5FF;
    --accent2:     #FF4D8F;
    --accent3:     #AAFF00;
    --text:        #C8D8F0;
    --text-muted:  #5A7A9A;
    --font-head:   'Syne', sans-serif;
    --font-mono:   'Space Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-mono) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1600px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.7rem !important; letter-spacing: 2px; text-transform: uppercase; }
[data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; font-size: 0.75rem !important; }

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px !important; }

/* ── Inputs ── */
input, textarea, select, [data-baseweb="select"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.4) !important;
}

/* ── Dataframe ── */
.dataframe-container, [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
details summary {
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
}
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 0.5rem !important; }

/* ── Custom card ── */
.radar-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.radar-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}

/* ── Headers ── */
h1, h2, h3 { font-family: var(--font-head) !important; }

/* ── Custom badge ── */
.churn-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-danger { background: rgba(255,77,143,0.15); color: #FF4D8F; border: 1px solid #FF4D8F; }
.badge-success { background: rgba(170,255,0,0.15); color: #AAFF00; border: 1px solid #AAFF00; }
.badge-warn { background: rgba(0,229,255,0.15); color: #00E5FF; border: 1px solid #00E5FF; }

/* ── Sidebar radio ── */
[data-testid="stRadio"] label { color: var(--text) !important; font-size: 0.85rem !important; }

/* ── Section divider ── */
.sec-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS & PLOTLY THEME
# ──────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono", color="#C8D8F0", size=11),
    title_font=dict(family="Syne", color="#C8D8F0", size=14),
    xaxis=dict(gridcolor="#1E2D45", zerolinecolor="#1E2D45", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1E2D45", zerolinecolor="#1E2D45", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2D45", font=dict(size=10)),
    margin=dict(l=40, r=20, t=45, b=40),
)
COLOR_YES   = "#FF4D8F"
COLOR_NO    = "#00E5FF"
COLOR_SEQ   = ["#0D1422", "#003A6B", "#005E9E", "#0087CC", "#00B4FF", "#00E5FF"]
COLOR_DIV   = [COLOR_YES, "#1E2D45", COLOR_NO]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ──────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_path_or_bytes):
    if isinstance(file_path_or_bytes, str):
        df = pd.read_csv(file_path_or_bytes)
    else:
        df = pd.read_csv(file_path_or_bytes)
    return df

@st.cache_data
def preprocess(df: pd.DataFrame):
    d = df.copy()
    d["Churn_Binary"] = (d["Churn"] == "Yes").astype(int)
    for col in CATEGORICAL_COLS:
        d[col] = d[col].astype(str)
    for col in NUMERIC_COLS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d.dropna(subset=NUMERIC_COLS, inplace=True)
    return d

@st.cache_resource
def train_model(df: pd.DataFrame):
    d = df.copy()
    d["Churn_Binary"] = (d["Churn"] == "Yes").astype(int)
    le = {}
    for col in CATEGORICAL_COLS:
        enc = LabelEncoder()
        d[col] = enc.fit_transform(d[col].astype(str))
        le[col] = enc
    features = CATEGORICAL_COLS + NUMERIC_COLS
    X = d[features]
    y = d["Churn_Binary"]
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_tr)
    return model, sc, le, X_te, y_te, features

# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; padding:0.5rem 0 1.5rem 0; border-bottom:1px solid #1E2D45; margin-bottom:1.5rem;">
        <div style="font-size:2.8rem;">📡</div>
        <div>
            <div style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#00E5FF; line-height:1; letter-spacing:-1px;">
                CHURN RADAR
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#5A7A9A; letter-spacing:3px; text-transform:uppercase; margin-top:3px;">
                Customer Analytics & Predictive Intelligence Dashboard
            </div>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#5A7A9A; letter-spacing:2px;">MODEL</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#AAFF00;">LOGISTIC REGRESSION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:#00E5FF; letter-spacing:1px; margin-bottom:1rem;">
            ⚙ CONTROL PANEL
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
        st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.7rem; letter-spacing:2px; color:#5A7A9A; text-transform:uppercase; margin-bottom:0.5rem;">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("", [
            "🏠  Overview",
            "📊  Exploratory Analysis",
            "🤖  ML Performance",
            "🔮  Predict Single Customer",
            "📋  Data Table",
        ], label_visibility="collapsed")

        st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem; letter-spacing:2px; color:#5A7A9A; text-transform:uppercase; margin-bottom:0.5rem;">Filters</div>', unsafe_allow_html=True)

        filter_contract = st.multiselect("Contract Type", ["Month-to-month", "One year", "Two year"], default=["Month-to-month", "One year", "Two year"])
        filter_internet = st.multiselect("Internet Service", ["DSL", "Fiber optic", "No"], default=["DSL", "Fiber optic", "No"])

        st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.65rem; color:#5A7A9A; line-height:1.8;">
        <span style="color:#00E5FF;">■</span> CHURN<br>
        <span style="color:#FF4D8F;">■</span> AT RISK<br>
        <span style="color:#AAFF00;">■</span> RETAINED
        </div>
        """, unsafe_allow_html=True)

    return uploaded, page, filter_contract, filter_internet

# ──────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ──────────────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame):
    d = preprocess(df)
    total    = len(d)
    churned  = d["Churn_Binary"].sum()
    retained = total - churned
    rate     = churned / total * 100
    avg_tenure     = d["tenure"].mean()
    avg_monthly    = d["MonthlyCharges"].mean()
    avg_total      = d["TotalCharges"].mean()
    churn_tenure   = d[d["Churn_Binary"]==1]["tenure"].mean()
    churn_monthly  = d[d["Churn_Binary"]==1]["MonthlyCharges"].mean()

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Total Customers", f"{total:,}")
    with c2: st.metric("Churn Rate",       f"{rate:.1f}%",       f"-{churned} customers", delta_color="inverse")
    with c3: st.metric("Churned",          f"{churned:,}",       f"↑ High Risk",          delta_color="off")
    with c4: st.metric("Retained",         f"{retained:,}",      f"↓ {(retained/total*100):.1f}%", delta_color="normal")
    with c5: st.metric("Avg Tenure",       f"{avg_tenure:.1f}m", f"Churn avg: {churn_tenure:.1f}m")
    with c6: st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}", f"Churn avg: ${churn_monthly:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Churn Gauge + Churn by Contract + Churn by Internet
    col_a, col_b, col_c = st.columns([1, 1.4, 1.4])

    with col_a:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rate,
            delta={"reference": 26, "increasing": {"color": COLOR_YES}},
            number={"suffix": "%", "font": {"size": 36, "family": "Syne", "color": COLOR_YES}},
            title={"text": "Churn Rate", "font": {"size": 13, "family": "Syne", "color": "#C8D8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E2D45"},
                "bar": {"color": COLOR_YES, "thickness": 0.25},
                "bgcolor": "#0D1422",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20],  "color": "rgba(170,255,0,0.15)"},
                    {"range": [20, 40], "color": "rgba(0,229,255,0.1)"},
                    {"range": [40, 100],"color": "rgba(255,77,143,0.1)"},
                ],
                "threshold": {"line": {"color": "#AAFF00", "width": 2}, "thickness": 0.75, "value": 26},
            }
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        ct = d.groupby("Contract")["Churn_Binary"].agg(["sum","count"]).reset_index()
        ct["rate"] = ct["sum"] / ct["count"] * 100
        ct.columns = ["Contract","Churned","Total","Rate"]
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=ct["Contract"], y=ct["Total"]-ct["Churned"], name="Retained",
                               marker_color=COLOR_NO, opacity=0.8))
        fig_c.add_trace(go.Bar(x=ct["Contract"], y=ct["Churned"], name="Churned",
                               marker_color=COLOR_YES, opacity=0.9,
                               text=[f"{r:.0f}%" for r in ct["Rate"]],
                               textposition="outside", textfont=dict(size=10, color=COLOR_YES)))
        fig_c.update_layout(**PLOTLY_LAYOUT, title="Churn by Contract", barmode="stack", height=250,
                            xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(fig_c, use_container_width=True)

    with col_c:
        it = d.groupby("InternetService")["Churn_Binary"].agg(["sum","count"]).reset_index()
        it["rate"] = it["sum"] / it["count"] * 100
        fig_i = go.Figure(go.Pie(
            labels=it["InternetService"],
            values=it["sum"],
            hole=0.6,
            marker=dict(colors=[COLOR_YES, COLOR_NO, "#AAFF00"], line=dict(color="#080C14", width=2)),
            textinfo="label+percent",
            textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>Churned: %{value}<br>Share: %{percent}<extra></extra>"
        ))
        fig_i.update_layout(**PLOTLY_LAYOUT, title="Churned by Internet Service", height=250,
                            showlegend=False)
        st.plotly_chart(fig_i, use_container_width=True)

    # Row 3: Tenure distribution + Monthly charges heatmap
    col_d, col_e = st.columns(2)

    with col_d:
        fig_t = go.Figure()
        fig_t.add_trace(go.Histogram(x=d[d["Churn_Binary"]==0]["tenure"],
                                     name="Retained", nbinsx=20,
                                     marker_color=COLOR_NO, opacity=0.75))
        fig_t.add_trace(go.Histogram(x=d[d["Churn_Binary"]==1]["tenure"],
                                     name="Churned", nbinsx=20,
                                     marker_color=COLOR_YES, opacity=0.75))
        fig_t.update_layout(**PLOTLY_LAYOUT, title="Tenure Distribution by Churn",
                            barmode="overlay", height=260, xaxis_title="Months", yaxis_title="Count")
        st.plotly_chart(fig_t, use_container_width=True)

    with col_e:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=d[d["Churn_Binary"]==0]["tenure"], y=d[d["Churn_Binary"]==0]["MonthlyCharges"],
            mode="markers", name="Retained",
            marker=dict(color=COLOR_NO, size=5, opacity=0.5, line=dict(width=0))
        ))
        fig_s.add_trace(go.Scatter(
            x=d[d["Churn_Binary"]==1]["tenure"], y=d[d["Churn_Binary"]==1]["MonthlyCharges"],
            mode="markers", name="Churned",
            marker=dict(color=COLOR_YES, size=5, opacity=0.7, line=dict(width=0))
        ))
        fig_s.update_layout(**PLOTLY_LAYOUT, title="Tenure vs Monthly Charges",
                            height=260, xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)")
        st.plotly_chart(fig_s, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# PAGE 2: EXPLORATORY ANALYSIS
# ──────────────────────────────────────────────────────────────────
def page_eda(df: pd.DataFrame):
    d = preprocess(df)

    st.markdown("""<div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00E5FF; letter-spacing:1px; margin-bottom:1rem;">
    ■ EXPLORATORY DATA ANALYSIS</div>""", unsafe_allow_html=True)

    # Churn rate per categorical feature
    cat_opts = [c for c in CATEGORICAL_COLS if c != "customerID"]
    sel_cat = st.selectbox("Feature to Analyze", cat_opts, index=cat_opts.index("PaymentMethod"))

    grp = d.groupby(sel_cat)["Churn_Binary"].agg(["mean","sum","count"]).reset_index()
    grp.columns = [sel_cat, "Rate", "Churned", "Total"]
    grp["Rate"] = grp["Rate"] * 100
    grp = grp.sort_values("Rate", ascending=True)

    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        colors = [COLOR_YES if r > grp["Rate"].median() else COLOR_NO for r in grp["Rate"]]
        fig_h = go.Figure(go.Bar(
            y=grp[sel_cat], x=grp["Rate"], orientation="h",
            marker_color=colors, opacity=0.85,
            text=[f"{r:.1f}%" for r in grp["Rate"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_h.update_layout(**PLOTLY_LAYOUT, title=f"Churn Rate by {sel_cat}",
                            height=300, xaxis_title="Churn Rate (%)", yaxis_title=None,
                            xaxis_range=[0, grp["Rate"].max() * 1.25])
        st.plotly_chart(fig_h, use_container_width=True)

    with col_r:
        fig_v = go.Figure()
        for label, color in [("Yes", COLOR_YES), ("No", COLOR_NO)]:
            sub = d[d["Churn"]==label]["MonthlyCharges"]
            fig_v.add_trace(go.Violin(
                y=sub, name=f"Churn={label}",
                fillcolor=color.replace(")", ",0.3)").replace("rgb","rgba") if "rgb" in color else color,
                line_color=color, opacity=0.8, box_visible=True,
                meanline_visible=True,
            ))
        fig_v.update_layout(**PLOTLY_LAYOUT, title="Monthly Charges Distribution",
                            height=300, yaxis_title="Monthly Charges ($)", xaxis_title=None)
        st.plotly_chart(fig_v, use_container_width=True)

    # Correlation matrix (numeric + encoded)
    st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:0.85rem; font-weight:700; color:#5A7A9A; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.75rem;">Correlation Matrix</div>', unsafe_allow_html=True)
    d_enc = d[NUMERIC_COLS + ["Churn_Binary"]].copy()
    for col in ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]:
        try:
            d_enc[col] = LabelEncoder().fit_transform(d[col].astype(str))
        except: pass

    corr = d_enc.corr()
    fig_cor = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, COLOR_YES], [0.5, "#1E2D45"], [1, COLOR_NO]],
        zmin=-1, zmax=1, text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont_size=9,
        hoverongaps=False,
    ))
    fig_cor.update_layout(**PLOTLY_LAYOUT, height=320, title="Feature Correlation")
    st.plotly_chart(fig_cor, use_container_width=True)

    # Value counts for binary/categorical features
    st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)

    with col_x:
        pm = d.groupby("PaymentMethod")["Churn_Binary"].mean() * 100
        fig_pm = go.Figure(go.Bar(
            x=pm.index, y=pm.values,
            marker_color=[COLOR_YES if v > pm.mean() else COLOR_NO for v in pm.values],
            text=[f"{v:.1f}%" for v in pm.values], textposition="outside"
        ))
        fig_pm.update_layout(**PLOTLY_LAYOUT, title="Churn Rate by Payment Method",
                             height=280, yaxis_title="Churn Rate (%)", xaxis_title=None)
        st.plotly_chart(fig_pm, use_container_width=True)

    with col_y:
        sc_grp = d.groupby("SeniorCitizen")["Churn_Binary"].agg(["sum","count"]).reset_index()
        sc_grp["label"] = sc_grp["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        sc_grp["rate"] = sc_grp["sum"] / sc_grp["count"] * 100
        fig_sc = go.Figure(go.Bar(
            x=sc_grp["label"], y=sc_grp["rate"],
            marker_color=[COLOR_NO, COLOR_YES],
            text=[f"{v:.1f}%" for v in sc_grp["rate"]], textposition="outside"
        ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, title="Churn Rate: Senior vs Non-Senior",
                             height=280, yaxis_title="Churn Rate (%)", xaxis_title=None)
        st.plotly_chart(fig_sc, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# PAGE 3: ML PERFORMANCE
# ──────────────────────────────────────────────────────────────────
def page_ml(df: pd.DataFrame):
    d = preprocess(df)
    model, sc, le, X_te, y_te, features = train_model(d)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)

    st.markdown("""<div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00E5FF; letter-spacing:1px; margin-bottom:1rem;">
    ■ MODEL PERFORMANCE</div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.3f}",  "↑ Good" if acc > 0.75 else "↓ Low")
    m2.metric("Precision", f"{prec:.3f}", "↑ Good" if prec > 0.70 else "↓ Low")
    m3.metric("Recall",    f"{rec:.3f}",  "↑ Good" if rec > 0.65 else "↓ Low")
    m4.metric("F1 Score",  f"{f1:.3f}",   "↑ Good" if f1 > 0.70 else "↓ Low")
    m5.metric("ROC AUC",   f"{roc_auc:.3f}", "↑ Strong" if roc_auc > 0.80 else "↓ Weak")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Confusion Matrix
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#0D1422"], [1, COLOR_NO]],
            text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=20, color="#fff"),
            showscale=False,
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, title="Confusion Matrix", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_b:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="#1E2D45", width=1.5), name="Random"))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color=COLOR_YES, width=2.5),
                                     name=f"AUC = {roc_auc:.3f}",
                                     fill="tozeroy", fillcolor="rgba(255,77,143,0.1)"))
        fig_roc.update_layout(**PLOTLY_LAYOUT, title="ROC Curve", height=300,
                              xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance (coefficients)
    st.markdown('<hr class="sec-divider"/>', unsafe_allow_html=True)
    coefs = model.coef_[0]
    feat_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
    feat_df = feat_df.reindex(feat_df["Coefficient"].abs().sort_values(ascending=True).index)
    colors_fi = [COLOR_YES if c > 0 else COLOR_NO for c in feat_df["Coefficient"]]

    fig_fi = go.Figure(go.Bar(
        y=feat_df["Feature"], x=feat_df["Coefficient"], orientation="h",
        marker_color=colors_fi, opacity=0.85,
        text=[f"{c:+.3f}" for c in feat_df["Coefficient"]],
        textposition="outside", textfont=dict(size=9),
    ))
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="Feature Coefficients (Logistic Regression)",
                         height=500, xaxis_title="Coefficient Value", yaxis_title=None)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Probability distribution
    col_c, col_d = st.columns(2)
    with col_c:
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(x=y_proba[y_te==0], nbinsx=25,
                                        name="Actual: No", marker_color=COLOR_NO, opacity=0.75))
        fig_prob.add_trace(go.Histogram(x=y_proba[y_te==1], nbinsx=25,
                                        name="Actual: Yes", marker_color=COLOR_YES, opacity=0.75))
        fig_prob.update_layout(**PLOTLY_LAYOUT, title="Predicted Probability Distribution",
                               barmode="overlay", height=260,
                               xaxis_title="Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_d:
        thresholds = np.linspace(0.01, 0.99, 100)
        prec_list, rec_list, f1_list = [], [], []
        for t in thresholds:
            yp = (y_proba >= t).astype(int)
            prec_list.append(precision_score(y_te, yp, zero_division=0))
            rec_list.append(recall_score(y_te, yp, zero_division=0))
            f1_list.append(f1_score(y_te, yp, zero_division=0))

        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(x=thresholds, y=prec_list, name="Precision",
                                     line=dict(color=COLOR_NO, width=2)))
        fig_thr.add_trace(go.Scatter(x=thresholds, y=rec_list, name="Recall",
                                     line=dict(color=COLOR_YES, width=2)))
        fig_thr.add_trace(go.Scatter(x=thresholds, y=f1_list, name="F1",
                                     line=dict(color="#AAFF00", width=2)))
        fig_thr.add_vline(x=0.5, line_dash="dash", line_color="#5A7A9A", line_width=1)
        fig_thr.update_layout(**PLOTLY_LAYOUT, title="Precision / Recall / F1 vs Threshold",
                              height=260, xaxis_title="Decision Threshold", yaxis_title="Score")
        st.plotly_chart(fig_thr, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# PAGE 4: SINGLE CUSTOMER PREDICTION
# ──────────────────────────────────────────────────────────────────
def page_predict(df: pd.DataFrame):
    d = preprocess(df)
    model, sc, le, X_te, y_te, features = train_model(d)

    st.markdown("""<div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00E5FF; letter-spacing:1px; margin-bottom:1rem;">
    ■ SINGLE CUSTOMER CHURN PREDICTION</div>""", unsafe_allow_html=True)

    st.markdown('<div class="radar-card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Customer Profile**")
        gender           = st.selectbox("Gender",           ["Male", "Female"])
        senior           = st.selectbox("Senior Citizen",   ["No", "Yes"])
        partner          = st.selectbox("Partner",          ["Yes", "No"])
        dependents       = st.selectbox("Dependents",       ["No", "Yes"])
        tenure           = st.slider("Tenure (months)",     1, 72, 24)

    with c2:
        st.markdown("**Services**")
        phone            = st.selectbox("Phone Service",    ["Yes", "No"])
        multi_lines      = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
        internet         = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec       = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
        online_bkp       = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot      = st.selectbox("Device Protection",["No", "Yes", "No internet service"])

    with c3:
        st.markdown("**Billing & Contract**")
        tech_sup         = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        stream_tv        = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        stream_movies    = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract         = st.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing",["Yes", "No"])
        payment          = st.selectbox("Payment Method",   ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_ch       = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0, 0.5)
        total_ch         = st.slider("Total Charges ($)",   30.0, 8320.0, float(monthly_ch * tenure), 10.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮  Run Prediction"):
        senior_int = 1 if senior == "Yes" else 0
        input_row = {
            "gender": gender, "Partner": partner, "Dependents": dependents,
            "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "tenure": tenure, "MonthlyCharges": monthly_ch, "TotalCharges": total_ch,
            "SeniorCitizen": senior_int,
        }
        row_df = pd.DataFrame([input_row])

        # Encode
        for col in CATEGORICAL_COLS:
            if col in le:
                try:
                    row_df[col] = le[col].transform(row_df[col].astype(str))
                except:
                    row_df[col] = 0
            else:
                enc = LabelEncoder().fit(d[col].astype(str))
                try: row_df[col] = enc.transform(row_df[col].astype(str))
                except: row_df[col] = 0

        X_input = sc.transform(row_df[features])
        prob    = model.predict_proba(X_input)[0][1]
        pred    = int(prob >= 0.5)

        st.markdown("<br>", unsafe_allow_html=True)

        col_res, col_gauge = st.columns([1.2, 1])
        with col_res:
            if pred == 1:
                risk_label = "HIGH RISK"
                badge_class = "badge-danger"
                color = COLOR_YES
                msg = "⚠️ This customer is **likely to churn**. Consider proactive retention strategies: discounted plan, loyalty rewards, or dedicated support."
            elif prob > 0.35:
                risk_label = "MODERATE RISK"
                badge_class = "badge-warn"
                color = COLOR_NO
                msg = "ℹ️ This customer shows **moderate churn signals**. Monitor engagement and offer service upgrades."
            else:
                risk_label = "LOW RISK"
                badge_class = "badge-success"
                color = "#AAFF00"
                msg = "✅ This customer is likely to **stay**. Continue standard service quality."

            st.markdown(f"""
            <div class="radar-card">
                <div style="font-family:'Syne',sans-serif; font-size:0.7rem; letter-spacing:3px; color:#5A7A9A; text-transform:uppercase;">Prediction Result</div>
                <div style="margin: 0.75rem 0; font-size:3rem; font-family:'Syne',sans-serif; font-weight:800; color:{color};">{prob*100:.1f}%</div>
                <div style="font-family:'Space Mono',sans-serif; font-size:0.7rem; color:#5A7A9A; margin-bottom:0.75rem;">Probability of Churn</div>
                <span class="churn-badge {badge_class}">{risk_label}</span>
                <div style="margin-top:1rem; font-size:0.8rem; color:#C8D8F0; line-height:1.7;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40, "family": "Syne", "color": color}},
                title={"text": "Churn Probability", "font": {"size": 12, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E2D45"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#0D1422", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],   "color": "rgba(170,255,0,0.12)"},
                        {"range": [35, 60],  "color": "rgba(0,229,255,0.1)"},
                        {"range": [60, 100], "color": "rgba(255,77,143,0.12)"},
                    ],
                }
            ))
            fig_g.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_g, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# PAGE 5: DATA TABLE
# ──────────────────────────────────────────────────────────────────
def page_table(df: pd.DataFrame, filter_contract, filter_internet):
    d = preprocess(df)
    filtered = d[d["Contract"].isin(filter_contract) & d["InternetService"].isin(filter_internet)]

    st.markdown("""<div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00E5FF; letter-spacing:1px; margin-bottom:1rem;">
    ■ DATA TABLE</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filtered Rows",  f"{len(filtered):,}")
    c2.metric("Churn Count",    f"{filtered['Churn_Binary'].sum():,}")
    c3.metric("Churn Rate",     f"{filtered['Churn_Binary'].mean()*100:.1f}%")

    search = st.text_input("🔍 Search by Customer ID", placeholder="e.g. 1234-ABCDE")
    if search:
        filtered = filtered[filtered["customerID"].str.contains(search, case=False, na=False)]

    show_cols = st.multiselect("Columns to display", list(d.columns),
                               default=["customerID","gender","tenure","Contract","MonthlyCharges","TotalCharges","Churn"])
    st.dataframe(
        filtered[show_cols].style
            .applymap(lambda v: f"color: {COLOR_YES}; font-weight:700;" if v == "Yes" else
                                f"color: {COLOR_NO}; font-weight:700;"  if v == "No"  else "",
                      subset=["Churn"] if "Churn" in show_cols else [])
            .format({"MonthlyCharges": "${:.2f}", "TotalCharges": "${:.2f}"} if "MonthlyCharges" in show_cols else {}),
        use_container_width=True, height=420
    )

    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button("⬇  Export Filtered CSV", csv_bytes, "churn_filtered.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    render_header()
    uploaded, page, filter_contract, filter_internet = render_sidebar()

    # Load data
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        try:
            df = load_data("customer_churn_prediction_dataset.csv")
            st.sidebar.success("✓ Dataset loaded from disk")
        except:
            st.warning("⚠️ No dataset found. Please upload a CSV file using the sidebar.")
            st.markdown("""
            <div class="radar-card" style="margin-top:2rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:1rem;">📁</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#00E5FF;">
                    Upload Your Customer Dataset
                </div>
                <div style="color:#5A7A9A; margin-top:0.5rem; font-size:0.8rem;">
                    Expected columns: customerID, gender, SeniorCitizen, Partner, Dependents, tenure,<br>
                    PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,<br>
                    DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,<br>
                    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

    # Route pages
    page_key = page.strip().split("  ")[1] if "  " in page else page.strip()

    if "Overview" in page:
        page_overview(df)
    elif "Exploratory" in page:
        page_eda(df)
    elif "ML Performance" in page:
        page_ml(df)
    elif "Predict" in page:
        page_predict(df)
    elif "Data Table" in page:
        page_table(df, filter_contract, filter_internet)

if __name__ == "__main__":
    main()
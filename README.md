# 📡 CHURN RADAR
### Customer Analytics & Predictive Intelligence Dashboard

> A dark-themed, production-grade Streamlit application for customer churn analysis and machine learning prediction — built around a telecom customer dataset and a Logistic Regression model.

---

## 🖥️ Preview

```
╔══════════════════════════════════════════════════════╗
║  📡 CHURN RADAR                                      ║
║  Customer Analytics & Predictive Intelligence        ║
║                              MODEL: LOGISTIC REG     ║
╚══════════════════════════════════════════════════════╝
```

**5 interactive pages:**
- 🏠 Overview — KPI cards, gauges, churn breakdowns
- 📊 Exploratory Analysis — dynamic charts, correlation heatmap
- 🤖 ML Performance — confusion matrix, ROC curve, feature importance
- 🔮 Predict Single Customer — real-time churn probability
- 📋 Data Table — searchable, filterable, exportable

---

## 📁 Project Structure

```
churn-radar/
│
├── churn_analytics_app.py                  ← Main Streamlit app
├── requirements.txt                        ← Python dependencies
├── README.md                               ← This file
│
├── customer_churn_prediction_dataset.csv   ← Dataset (place here)
└── Log.pkl                                 ← Trained model (optional)
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/churn-radar.git
cd churn-radar
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data files

Place both files in the **same folder** as `churn_analytics_app.py`:

| File | Description |
|------|-------------|
| `customer_churn_prediction_dataset.csv` | Customer dataset (300+ rows, 21 columns) |
| `Log.pkl` | Pre-trained LogisticRegression (optional — app trains its own) |

### 5. Run the app

```bash
streamlit run churn_analytics_app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📊 Dataset Schema

The app expects a CSV with the following 21 columns:

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | string | Unique customer identifier |
| `gender` | string | Male / Female |
| `SeniorCitizen` | int | 0 = No, 1 = Yes |
| `Partner` | string | Yes / No |
| `Dependents` | string | Yes / No |
| `tenure` | int | Months with company |
| `PhoneService` | string | Yes / No |
| `MultipleLines` | string | Yes / No / No phone service |
| `InternetService` | string | DSL / Fiber optic / No |
| `OnlineSecurity` | string | Yes / No / No internet service |
| `OnlineBackup` | string | Yes / No / No internet service |
| `DeviceProtection` | string | Yes / No / No internet service |
| `TechSupport` | string | Yes / No / No internet service |
| `StreamingTV` | string | Yes / No / No internet service |
| `StreamingMovies` | string | Yes / No / No internet service |
| `Contract` | string | Month-to-month / One year / Two year |
| `PaperlessBilling` | string | Yes / No |
| `PaymentMethod` | string | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | float | Monthly billing amount ($) |
| `TotalCharges` | float | Total amount billed ($) |
| `Churn` | string | **Target** — Yes / No |

---

## 🤖 Machine Learning

The app trains a **Logistic Regression** model live on startup using `@st.cache_resource` (trains once, cached for the session).

**Pipeline:**
1. Label-encode all categorical features
2. StandardScaler normalization
3. 75/25 train-test split (stratified)
4. `LogisticRegression(max_iter=1000, random_state=42)`

**Metrics reported:**
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC curve
- Confusion Matrix
- Feature coefficients (importance)
- Precision / Recall / F1 vs decision threshold

---

## 🎨 Design System

| Token | Value | Usage |
|-------|-------|-------|
| Background | `#080C14` | Page background |
| Surface | `#0D1422` | Sidebar, cards |
| Accent (Cyan) | `#00E5FF` | Primary highlight, metrics |
| Accent (Pink) | `#FF4D8F` | Churn / at-risk indicators |
| Accent (Lime) | `#AAFF00` | Retained / safe indicators |
| Border | `#1E2D45` | Card borders, dividers |
| Heading Font | Syne (800) | Titles, labels |
| Body Font | Space Mono | Data, code, UI text |

---

## 🗺️ Pages Guide

### 🏠 Overview
- 6 KPI metrics: total customers, churn rate, churned, retained, avg tenure, avg charges
- Churn rate gauge with threshold marker
- Stacked bar — churn by contract type
- Donut — churn share by internet service
- Histogram — tenure distribution (churn vs retained)
- Scatter — tenure vs monthly charges

### 📊 Exploratory Analysis
- Dynamic feature selector → horizontal bar chart of churn rates
- Violin plot — monthly charges by churn status
- Correlation heatmap (numeric features)
- Churn by payment method (bar)
- Senior vs non-senior churn comparison

### 🤖 ML Performance
- Metric row: Accuracy / Precision / Recall / F1 / AUC
- Confusion matrix heatmap
- ROC curve with AUC fill
- Feature coefficient chart (positive = churn risk, negative = retention)
- Predicted probability distribution
- Precision / Recall / F1 vs threshold sweep

### 🔮 Predict Single Customer
- 3-column form with all 20 input features
- Click **Run Prediction** to get:
  - Churn probability percentage
  - Risk badge: HIGH / MODERATE / LOW
  - Personalized retention recommendation
  - Probability gauge chart

### 📋 Data Table
- Sidebar filters: Contract type, Internet service
- Free-text search by Customer ID
- Column selector
- Styled table (churn = red, retained = cyan)
- Export filtered data as CSV

---

## 🔧 Configuration

### Sidebar Filters
The **Contract Type** and **Internet Service** multiselects in the sidebar apply globally to the **Data Table** page. Other pages use the full dataset for statistical validity.

### Uploading a Custom Dataset
Use the **Upload Dataset** button in the sidebar to load any CSV matching the schema above. The app auto-detects and caches it for the session.

---

## 🐍 Python Version

Tested on **Python 3.10+**. Python 3.8 minimum required.

```bash
python --version   # Should be 3.10+
```

---

## 🚀 Deploy to Streamlit Cloud

1. Push the project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `churn_analytics_app.py`
4. Add dataset via the **Upload** widget at runtime (or commit the CSV to the repo)
5. Click **Deploy**

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.35.0 | Web app framework |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.26.0 | Numerical computing |
| `scikit-learn` | ≥ 1.3.0 | ML model, preprocessing, metrics |
| `plotly` | ≥ 5.18.0 | Interactive charts |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| App says "No dataset found" | Place `customer_churn_prediction_dataset.csv` in the same folder |
| Blank charts / no data | Check CSV has correct column names (case-sensitive) |
| `TotalCharges` NaN error | Ensure `TotalCharges` column has numeric values, not spaces |
| Port already in use | Run `streamlit run churn_analytics_app.py --server.port 8502` |
| Slow first load | Normal — model trains once on startup, then caches |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with Streamlit · scikit-learn · Plotly*

live demo:-https://abhishekgupta121-customer-churn-prediction-ml-app-oq7tiz.streamlit.app/

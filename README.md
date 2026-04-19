# DataScience-Based Autonomous Wealth & Expense Intelligence Ecosystem

A **two-tier** expense intelligence project — a Basic Expense Tracker satisfying the core Data Science problem statement, plus **AuraFi V2**, an industry-grade AI extension.

---

## 📁 Project Structure

```
├── basic_tracker.py           # ✅ Core problem statement implementation
├── app.py                     # 🚀 AuraFi V2 — Advanced AI Extension
├── ai_agents.py               # Multi-Agent AI system (AuraFi)
├── data_generator.py          # Synthetic data + live transaction simulator
├── historical_transactions.csv # Pre-generated dataset (12 months)
├── aura_watch_model.pkl       # Trained Isolation Forest anomaly model
├── requirements.txt
├── BLUEPRINT.md               # AuraFi architectural blueprint
└── README.md
```

---

## ✅ Part 1: Basic Expense Tracker (`basic_tracker.py`)

> **Directly addresses the problem statement requirements.**

### Features
| Requirement | Implementation |
|---|---|
| Input expense data | CSV upload, manual form entry, or sample data |
| Clean & analyze using Pandas | Full cleaning, type inference, null handling |
| Charts using Matplotlib/Seaborn | Bar, Pie, Line, Stacked Bar, Histogram, Heatmap |
| Category-wise spending | Horizontal bar + pie charts per category |
| Monthly trends | Line chart + stacked category-month breakdown |
| Insights | Auto-generated text insights + anomaly flags |
| Streamlit dashboard | Full responsive dashboard with sidebar filters |

### Run
```bash
streamlit run basic_tracker.py
```

---

## 🚀 Part 2: AuraFi V2 — Advanced AI Extension (`app.py`)

> **An autonomous, multi-agent FinTech intelligence platform — far beyond the basic problem statement.**

### Architecture

```
[ Synthetic Data Webhook ] --> { FastAPI Ingestion Engine }
                                        |
                                        v
                          [ Real-Time Data Stream Bus ]
                               /                 \
                              v                   v
                [ Anomaly Detection AI ]   [ Time-Series Forecaster AI ]
                              \                   /
                               v                 v
                       [ AuraFi V2 Dashboard (Streamlit) ]
```

### Multi-Agent System
| Agent | Role |
|---|---|
| **Aura-Watch** | Isolation Forest anomaly detection on live transactions |
| **Aura-Brain** | Time-series cash-flow forecasting (end-of-month projection) |
| **Aura-Action** | Prescriptive actions: auto-sweep, debt-freeze recommendations |
| **Aura-Wellness** | Financial health score (0–100) + debt-trap prevention |
| **Aura-Green** | Carbon footprint tracker mapped to transaction categories |

### Advanced Features
- 🔴 **Real-Time Streaming Dashboard** — simulated webhook injects live transactions
- 📈 **Cash-Flow Forecasting** — predicts end-of-month spend from historical patterns
- 🌍 **Green Ledger** — CO₂ footprint analysis per spending category
- 🧠 **Debt-Trap AI** — financial wellness gauge with prescriptive autonomous actions
- 💚 **Charity Roundup** — auto-rounds transactions and tracks micro-donations

### Run
```bash
streamlit run app.py
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate historical data (if CSV is missing)
python data_generator.py

# 3. Run the Basic Tracker
streamlit run basic_tracker.py

# 4. Run AuraFi V2 (Advanced)
streamlit run app.py
```

---

## 🛠️ Tech Stack

**Basic Tracker:** Pandas · Matplotlib · Seaborn · Streamlit  
**AuraFi V2:** + Plotly · Scikit-learn · NumPy · Joblib

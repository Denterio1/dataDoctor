# 🩺 dataDoctor — Autonomous Data Inspection Agent

> Drop your data. Get answers.

**dataDoctor** is an open-source autonomous agent that inspects, cleans, and prepares your data for Machine Learning — in seconds.

No code required. Just upload your file.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Inspection** | Missing values, duplicates, outliers, column stats |
| 🧹 **Auto Cleaning** | Fill missing values, remove duplicates |
| 📊 **HTML Reports** | Professional reports you can share |
| 🤖 **ML Readiness Score** | Know if your data is ready for ML |
| 🔗 **Relationship Detector** | Find correlations between columns |
| ⚡ **Data Preparation** | Encode + scale data for ML in one step |
| 📉 **Drift Detection** | Compare two datasets and find what changed |
| 🧠 **Data Memory** | Track inspection history across sessions |
| 💡 **AI Suggestions** | Get smart recommendations from any LLM |
| 🌐 **Web UI** | Beautiful browser interface — no Terminal needed |

**Supported file formats:** CSV · Excel (.xlsx) · JSON

---

## 🚀 Quick Start

### Option 1 — Web UI (Recommended for most users)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

Upload any CSV, Excel, or JSON file and explore your data instantly.

---

### Option 2 — CLI (For developers)

```bash
# Full inspection report
python cli.py inspect examples/sample_sales.csv

# Interactive mode (guided, no flags needed)
python cli.py interactive

# See all commands
python cli.py --help
```

---

## 📦 Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Denterio1/dataDoctor.git
cd dataDoctor
pip install -r requirements.txt
```

---

## 🖥️ Web UI Guide

1. Run `streamlit run app.py`
2. Upload your file in the sidebar
3. Explore 8 tabs:

| Tab | What you get |
|-----|-------------|
| 📊 Overview | Data preview + column type chart |
| 🔍 Quality | Missing values + outlier charts |
| 📈 Statistics | Interactive histograms per column |
| 🤖 ML Readiness | Score + detailed checks |
| 🔗 Relationships | Correlation heatmap |
| 🧹 Cleaning | Clean + download your data |
| 📉 Drift | Compare two datasets |
| 📂 Multi-File | Analyse and compare multiple files |

---

## ⌨️ CLI Commands

```bash
python cli.py inspect    <file>              # Full report
python cli.py clean      <file>              # Clean data
python cli.py stats      <file>              # Column statistics
python cli.py missing    <file>              # Missing values
python cli.py duplicates <file>              # Duplicate rows
python cli.py outliers   <file>              # Outlier detection
python cli.py export     <file>              # Export cleaned CSV
python cli.py report     <file>              # HTML report
python cli.py ml         <file>              # ML Readiness Score
python cli.py prepare    <file>              # Prepare for ML
python cli.py relations  <file>              # Column relationships
python cli.py suggest    <file>              # AI suggestions
python cli.py memory     list                # Show tracked files
python cli.py memory     compare <file>      # Compare last 2 runs
python cli.py drift      <baseline> <current># Drift detection
python cli.py interactive                    # Guided mode
```

**Options:**
```bash
--strategy mean    # Fill missing with mean (default)
--strategy median  # Fill missing with median
--strategy mode    # Fill missing with most frequent value
--strategy drop    # Drop rows with missing values
--no-dedup         # Keep duplicate rows
```

---

## 💡 AI Suggestions Setup

dataDoctor supports any OpenAI-compatible API. Create a `.env` file:

```env
# Groq (free) — recommended
DATADOCTOR_API_KEY=your_groq_key
DATADOCTOR_BASE_URL=https://api.groq.com/openai/v1
DATADOCTOR_MODEL=llama-3.3-70b-versatile

# Google Gemini (free)
# DATADOCTOR_API_KEY=your_gemini_key
# DATADOCTOR_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# DATADOCTOR_MODEL=gemini-1.5-flash

# OpenAI (paid)
# DATADOCTOR_API_KEY=your_openai_key
# DATADOCTOR_BASE_URL=https://api.openai.com/v1
# DATADOCTOR_MODEL=gpt-4o-mini
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📁 Project Structure

```
dataDoctor/
├── app.py                  ← Web UI (Streamlit)
├── cli.py                  ← Command line interface
├── requirements.txt
├── .env                    ← API keys (not committed)
│
├── src/
│   ├── core/
│   │   └── agent.py        ← Main DataDoctor agent
│   ├── data/
│   │   ├── loader.py       ← CSV / Excel / JSON reader
│   │   ├── analyzer.py     ← Quality checks & statistics
│   │   ├── cleaner.py      ← Missing values & deduplication
│   │   ├── ml_readiness.py ← ML Readiness Score
│   │   ├── relationships.py← Column relationship detector
│   │   ├── preparator.py   ← ML data preparation pipeline
│   │   ├── drift.py        ← Data drift detection
│   │   ├── memory.py       ← Persistent inspection history
│   │   └── ai_suggestions.py ← AI-powered recommendations
│   └── report.py           ← HTML report generator
│
├── tests/
│   └── test_datadoctor.py  ← 38 pytest tests
│
└── examples/
    ├── sample_sales.csv
    ├── sample_employees.xlsx
    └── sample_products.json
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

38 tests — all passing.

---

## 🗺️ Roadmap

| Version | Features |
|---------|----------|
| **v0.1.0** ✅ | Core inspection, cleaning, ML prep, web UI |
| **v0.2.0** | Auto Feature Engineering, Schema Validator |
| **v0.3.0** | ML Baseline, Feature Importance, Pipeline Export |
| **v0.4.0** | Database Connector, REST API, Parquet support |
| **v1.0.0** | Docker, Cloud Deploy, Plugin System |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request
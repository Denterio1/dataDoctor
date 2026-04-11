"""
app.py — dataDoctor Web UI (Streamlit)

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO



from src.data.encoding_advisor  import encoding_advisor
from src.data.schema_validator  import validate_schema, infer_schema, schema_to_dict, schema_from_dict
from src.data.relationships     import detect_relationships
from src.data.ml_readiness      import ml_readiness
from src.data.preparator        import prepare_for_ml

from src.data.analyzer import full_report, detect_outliers
from src.data.loader       import load_file
from src.data.cleaner      import handle_missing, remove_duplicates
from src.data.drift        import detect_drift
from src.core.agent        import DataDoctor

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="dataDoctor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 2.8rem;
        font-weight: 300;
        color: #e8e6e1;
        line-height: 1.2;
    }
    .main-title em {
        font-style: italic;
        color: #c8f06e;
    }
    .subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        color: #7a7875;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #17171a;
        border: 0.5px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1;
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #7a7875;
        margin-top: 4px;
    }
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #7a7875;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
    }
    div[data-testid="stSidebarContent"] {
        background: #0f0f11;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_uploaded(uploaded_file) -> dict:
    """Load an uploaded file into a data dict."""
    ext  = os.path.splitext(uploaded_file.name)[1].lower()
    name = uploaded_file.name

    if ext == ".csv":
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    elif ext == ".json":
        df = pd.read_json(uploaded_file)
    else:
        st.error(f"Unsupported file type: {ext}")
        st.stop()

    return {"columns": list(df.columns), "df": df, "source": name}


def _health_color(score: int) -> str:
    if score >= 85: return "#c8f06e"
    if score >= 65: return "#f0c46e"
    return "#f06e6e"


def _severity_color(sev: str) -> str:
    return {"none": "#c8f06e", "low": "#f0c46e", "medium": "#f0c46e", "high": "#f06e6e"}.get(sev, "#7a7875")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem'>
        <div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875;letter-spacing:0.12em;margin-bottom:0.5rem'>🩺 DATADOCTOR</div>
        <div style='font-family:Fraunces,serif;font-size:1.4rem;font-weight:300;color:#e8e6e1'>Data Inspection<br><em style='color:#c8f06e'>Agent</em></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Upload your file")
    uploaded = st.file_uploader(
        "CSV, Excel, or JSON",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Settings")

    missing_strategy = st.selectbox(
        "Missing value strategy",
        ["mean", "median", "mode", "drop"],
        index=0,
    )

    remove_dupes = st.checkbox("Remove duplicate rows", value=True)
    run_ml       = st.checkbox("Run ML Readiness check", value=True)
    run_rels     = st.checkbox("Detect column relationships", value=True)

    st.markdown("---")
    st.markdown("### Drift Detection")
    uploaded_baseline = st.file_uploader(
        "Baseline file (optional)",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="visible",
        key="baseline",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875'>"
        "dataDoctor v0.1.0<br>Open Source</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("""
    <div style='text-align:center;padding:6rem 2rem'>
        <div style='font-family:Fraunces,serif;font-size:3.5rem;font-weight:300;color:#e8e6e1;line-height:1.2'>
            Drop your data.<br><em style='color:#c8f06e'>Get answers.</em>
        </div>
        <div style='font-family:DM Mono,monospace;font-size:13px;color:#7a7875;margin-top:1.5rem'>
            Upload a CSV, Excel, or JSON file in the sidebar to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load & analyse ────────────────────────────────────────────────────────────

with st.spinner("Analysing your data..."):
    data     = _load_uploaded(uploaded)
    analysis = full_report(data)
    outliers = detect_outliers(data)
    ml       = ml_readiness(data, outliers) if run_ml else None
    rels     = detect_relationships(data, threshold=0.4) if run_rels else []

    # Clean
    clean_data = data
    clean_log  = {}
    if remove_dupes and analysis["duplicate_rows"] > 0:
        clean_data, n = remove_duplicates(clean_data)
        clean_log["duplicates_removed"] = n
    if sum(analysis["missing_values"].values()) > 0:
        clean_data, changes = handle_missing(clean_data, strategy=missing_strategy)
        clean_log["missing"] = changes


# ── Header ────────────────────────────────────────────────────────────────────

col_title, col_score = st.columns([3, 1])

with col_title:
    st.markdown(f"""
    <div class='main-title'>Report for<br><em>{uploaded.name}</em></div>
    <div class='subtitle'>{analysis['shape']['rows']:,} rows · {analysis['shape']['columns']} columns · {uploaded.name.split('.')[-1].upper()}</div>
    """, unsafe_allow_html=True)

with col_score:
    if ml:
        score = ml["score"]
        color = _health_color(score)
        st.markdown(f"""
        <div style='text-align:right;padding-top:1rem'>
            <div style='font-family:Fraunces,serif;font-size:3.5rem;font-weight:600;color:{color};line-height:1'>{score}</div>
            <div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875'>ML Health Score / 100</div>
            <div style='width:100%;height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-top:8px'>
                <div style='width:{score}%;height:100%;background:{color};border-radius:2px'></div>
            </div>
            <div style='font-family:DM Mono,monospace;font-size:11px;color:{color};margin-top:4px'>Grade {ml["grade"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Overview metrics ──────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
total_missing = sum(analysis["missing_values"].values())

def _metric(col, val, label, color):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val' style='color:{color}'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

_metric(m1, f"{analysis['shape']['rows']:,}", "rows", "#5ce0c6")
_metric(m2, analysis["shape"]["columns"],      "columns", "#5ce0c6")
_metric(m3, total_missing,  "missing cells",  "#f06e6e" if total_missing > 0 else "#c8f06e")
_metric(m4, analysis["duplicate_rows"], "duplicate rows", "#f06e6e" if analysis["duplicate_rows"] > 0 else "#c8f06e")
_metric(m5, len(outliers),  "cols with outliers", "#f0c46e" if outliers else "#c8f06e")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📊 Overview", "🔍 Quality", "📈 Statistics", "🤖 ML Readiness", "🔗 Relationships", "🧹 Cleaning", "📉 Drift", "📂 Multi-File", "🔬 Lab"])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Data Preview")
    st.dataframe(data["df"].head(20), use_container_width=True)

    st.markdown("#### Column Types")
    type_data = {
        col: "Numeric" if s["type"] == "numeric" else "Text"
        for col, s in analysis["column_stats"].items()
    }
    fig = px.pie(
        names=list(type_data.values()),
        title="Column type distribution",
        color_discrete_map={"Numeric": "#5ce0c6", "Text": "#c8f06e"},
        hole=0.5,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Quality ────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Missing Values")
    mv = analysis["missing_values"]
    if total_missing == 0:
        st.success("No missing values — data is complete!")
    else:
        mv_df = pd.DataFrame([
            {"Column": col, "Missing": cnt, "Percentage": round(cnt / analysis["shape"]["rows"] * 100, 1)}
            for col, cnt in mv.items()
        ])
        fig = px.bar(
            mv_df, x="Column", y="Percentage",
            color="Percentage",
            color_continuous_scale=["#c8f06e", "#f0c46e", "#f06e6e"],
            title="Missing values per column (%)",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mv_df, use_container_width=True)

    st.markdown("#### Outliers")
    if not outliers:
        st.success("No outliers detected!")
    else:
        for col, info in outliers.items():
            st.warning(f"**{col}**: {info['count']} outlier(s) — safe range: {info['lower']} → {info['upper']}")
            fig = px.box(data["df"], y=col, title=f"{col} distribution", color_discrete_sequence=["#5ce0c6"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Statistics ─────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Column Statistics")
    stats = analysis["column_stats"]

    num_cols = [col for col, s in stats.items() if s["type"] == "numeric"]
    cat_cols = [col for col, s in stats.items() if s["type"] == "text"]

    if num_cols:
        st.markdown("##### Numeric columns")
        num_df = pd.DataFrame([
            {"Column": col, "Min": stats[col]["min"], "Max": stats[col]["max"],
             "Mean": stats[col]["mean"], "Unique": stats[col]["unique"], "Count": stats[col]["count"]}
            for col in num_cols
        ])
        st.dataframe(num_df, use_container_width=True)

        selected = st.selectbox("Select column to visualize", num_cols)
        fig = px.histogram(
            data["df"], x=selected, nbins=20,
            title=f"{selected} distribution",
            color_discrete_sequence=["#5ce0c6"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
        st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.markdown("##### Categorical columns")
        cat_df = pd.DataFrame([
            {"Column": col, "Unique Values": stats[col]["unique"],
             "Most Common": stats[col]["most_common"], "Count": stats[col]["count"]}
            for col in cat_cols
        ])
        st.dataframe(cat_df, use_container_width=True)

        selected_cat = st.selectbox("Select column to visualize", cat_cols, key="cat_select")
        vc = data["df"][selected_cat].value_counts().reset_index()
        vc.columns = [selected_cat, "count"]
        fig = px.bar(vc, x=selected_cat, y="count", title=f"{selected_cat} value counts",
                     color_discrete_sequence=["#c8f06e"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: ML Readiness ───────────────────────────────────────────────────────
with tabs[3]:
    if not ml:
        st.info("Enable ML Readiness check in the sidebar.")
    else:
        score = ml["score"]
        color = _health_color(score)

        st.markdown(f"""
        <div style='text-align:center;padding:2rem 0'>
            <div style='font-family:Fraunces,serif;font-size:5rem;font-weight:600;color:{color};line-height:1'>{score}</div>
            <div style='font-family:DM Mono,monospace;font-size:13px;color:#7a7875'>out of 100 — Grade {ml["grade"]}</div>
            <div style='font-size:15px;color:#e8e6e1;margin-top:1rem'>{ml["summary"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Checks")
        for check in ml["checks"]:
            icon  = "✅" if check["status"] == "pass" else "⚠️" if check["status"] == "warn" else "❌"
            color_text = "#c8f06e" if check["status"] == "pass" else "#f0c46e" if check["status"] == "warn" else "#f06e6e"
            pts   = check["points"]
            max_p = check["max"]
            pct   = int(pts / max_p * 100)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"{icon} **{check['name']}**  \n{check['detail']}")
                st.progress(pct / 100)
            with col2:
                st.markdown(f"<div style='text-align:right;font-family:DM Mono,monospace;color:{color_text};font-size:1.2rem;padding-top:0.5rem'>{pts}/{max_p}</div>", unsafe_allow_html=True)
            st.markdown("---")


# ── Tab 5: Relationships ──────────────────────────────────────────────────────
with tabs[4]:
    if not run_rels:
        st.info("Enable relationship detection in the sidebar.")
    elif not rels:
        st.success("No strong relationships found between columns.")
    else:
        st.markdown(f"#### Found {len(rels)} relationship(s)")
        for r in rels:
            strength = r["strength"]
            color    = "#c8f06e" if strength >= 0.8 else "#f0c46e" if strength >= 0.6 else "#7a7875"
            level    = "Strong" if strength >= 0.8 else "Moderate" if strength >= 0.6 else "Weak"
            st.markdown(f"""
            **{r['col_a']}** ↔ **{r['col_b']}**  
            `{r['method']}` · {r['type']} · {r['direction']}
            """)
            st.progress(strength)
            st.markdown(f"<span style='color:{color};font-family:DM Mono,monospace;font-size:12px'>{level} ({strength})</span>", unsafe_allow_html=True)
            st.markdown("---")

        # Correlation heatmap for numeric columns
        num_cols = [c for c in data["df"].columns if pd.api.types.is_numeric_dtype(data["df"][c])]
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Heatmap")
            corr = data["df"][num_cols].corr()
            fig  = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale=["#f06e6e", "#17171a", "#c8f06e"],
                title="Pearson correlation matrix",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 6: Cleaning ───────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("#### Cleaning Log")
    if not clean_log:
        st.success("No cleaning was necessary.")
    else:
        if "duplicates_removed" in clean_log:
            st.warning(f"Removed **{clean_log['duplicates_removed']}** duplicate row(s).")
        if "missing" in clean_log:
            mv = clean_log["missing"]
            if "rows_dropped" in mv:
                st.warning(f"Dropped **{mv['rows_dropped']}** row(s) with missing values.")
            else:
                for col, info in mv.items():
                    st.info(f"**{col}**: filled {info['filled']} value(s) with `{info['replacement']}` ({info['strategy']})")

    st.markdown("#### Cleaned Data Preview")
    st.dataframe(clean_data["df"].head(20), use_container_width=True)

    st.markdown("#### Download Cleaned File")
    csv_bytes = clean_data["df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download cleaned CSV",
        data=csv_bytes,
        file_name=f"{os.path.splitext(uploaded.name)[0]}_cleaned.csv",
        mime="text/csv",
    )

    st.markdown("#### Download ML-Ready File")
    prepared, _ = prepare_for_ml(clean_data, missing_strategy=missing_strategy)
    ml_bytes = prepared["df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download ML-ready CSV",
        data=ml_bytes,
        file_name=f"{os.path.splitext(uploaded.name)[0]}_ready.csv",
        mime="text/csv",
    )


# ── Tab 7: Drift ──────────────────────────────────────────────────────────────
with tabs[6]:
    if not uploaded_baseline:
        st.info("Upload a baseline file in the sidebar to detect drift.")
    else:
        with st.spinner("Detecting drift..."):
            baseline_data = _load_uploaded(uploaded_baseline)
            drift_result  = detect_drift(baseline_data, data)

        sev   = drift_result["severity"]
        color = _severity_color(sev)

        st.markdown(f"""
        <div style='text-align:center;padding:1.5rem 0'>
            <div style='font-family:Fraunces,serif;font-size:2.5rem;font-weight:600;color:{color}'>{sev.upper()}</div>
            <div style='font-family:DM Mono,monospace;font-size:13px;color:#7a7875'>Drift Severity</div>
            <div style='font-size:15px;color:#e8e6e1;margin-top:0.5rem'>{drift_result["summary"]}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Baseline rows", drift_result["base_shape"]["rows"])
        col2.metric("Current rows",  drift_result["new_shape"]["rows"])

        if drift_result["drifted_columns"]:
            st.markdown("#### Drifted Columns")
            for d in drift_result["drifted_columns"]:
                sev_c = "#f06e6e" if d["severity"] == "high" else "#f0c46e"
                with st.expander(f"⚠️ {d['column']} — {d['severity'].upper()}"):
                    for issue in d["issues"]:
                        st.markdown(f"- {issue}")

        if drift_result["stable_columns"]:
            st.markdown("#### Stable Columns")
            st.success(", ".join(drift_result["stable_columns"]))

# ── Tab 9: Lab ────────────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("## 🔬 Lab — Advanced Features")
    st.markdown("---")

    feature = st.selectbox("Choose a feature:", [
        "⚙️ Auto Feature Engineering",
        "🎯 Target Column Detection",
        "🕸️ Correlation Network Graph",
        "🔤 Smart Encoding Advisor",
        "📋 Data Schema Validator",
    ])

    st.markdown("---")

    # ── Auto Feature Engineering ──────────────────────────────────────────────
    if feature == "⚙️ Auto Feature Engineering":
        st.markdown("### ⚙️ Auto Feature Engineering")
        from src.data.preparator import prepare_for_ml

        prepared, log = prepare_for_ml(data, missing_strategy=missing_strategy)

        st.markdown("#### New Features Created")
        original_cols = set(data["df"].columns)
        new_cols = [c for c in prepared["df"].columns if c not in original_cols]

        if new_cols:
            st.success(f"✓ Created {len(new_cols)} new feature(s)")
            st.dataframe(prepared["df"][new_cols].head(10), use_container_width=True)
        else:
            st.info("No new features generated for this dataset.")

        st.markdown("#### Full Prepared Dataset")
        st.dataframe(prepared["df"].head(20), use_container_width=True)

        csv_b = prepared["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prepared Dataset",
            data=csv_b,
            file_name=f"{os.path.splitext(uploaded.name)[0]}_features.csv",
            mime="text/csv",
        )

    # ── Target Column Detection ───────────────────────────────────────────────
    elif feature == "🎯 Target Column Detection":
        st.markdown("### 🎯 Target Column Detection")
        from src.data.ml_readiness import ml_readiness
        from src.data.analyzer import detect_outliers

        outliers = detect_outliers(data)
        ml = ml_readiness(data, outliers)
        df = data["df"]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if not numeric_cols:
            st.warning("No numeric columns found for target detection.")
        else:
            scores = {}
            for col in numeric_cols:
                n_unique = df[col].nunique()
                null_pct = df[col].isna().mean()
                is_id = n_unique == len(df)
                score = 0
                if not is_id:
                    score += 30
                if null_pct < 0.05:
                    score += 20
                if n_unique > 5:
                    score += 20
                if df[col].std() > 0:
                    score += 30
                scores[col] = score

            scores_df = pd.DataFrame([
                {"Column": col, "Score": sc, "Unique": df[col].nunique(),
                 "Null %": f"{df[col].isna().mean():.1%}",
                 "Std": round(float(df[col].std()), 3)}
                for col, sc in sorted(scores.items(), key=lambda x: -x[1])
            ])

            st.dataframe(scores_df, use_container_width=True)

            best = max(scores, key=scores.get)
            st.success(f"✓ Recommended target column: **{best}** (score {scores[best]}/100)")

            fig = px.bar(
                scores_df, x="Column", y="Score",
                color="Score",
                color_continuous_scale=["#f06e6e", "#f0c46e", "#c8f06e"],
                title="Target Column Scores",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Network Graph ─────────────────────────────────────────────
    elif feature == "🕸️ Correlation Network Graph":
        st.markdown("### 🕸️ Correlation Network Graph")
        from src.data.relationships import detect_relationships

        threshold = st.slider("Minimum correlation strength:", 0.1, 0.9, 0.4, 0.05)
        rels = detect_relationships(data, threshold=threshold)

        if not rels:
            st.info(f"No relationships found above threshold {threshold}.")
        else:
            rel_df = pd.DataFrame([{
                "Column A": r["col_a"],
                "Column B": r["col_b"],
                "Strength": r["strength"],
                "Direction": r["direction"],
                "Method": r["method"],
            } for r in rels])
            st.dataframe(rel_df, use_container_width=True)

            fig = px.scatter(
                rel_df, x="Column A", y="Column B",
                size="Strength", color="Strength",
                color_continuous_scale=["#f06e6e", "#f0c46e", "#c8f06e"],
                title="Column Relationships",
                size_max=40,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
            st.plotly_chart(fig, use_container_width=True)

    # ── Smart Encoding Advisor ────────────────────────────────────────────────
    elif feature == "🔤 Smart Encoding Advisor":
        st.markdown("### 🔤 Smart Encoding Advisor")

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model type:", ["tree", "linear", "neural"])
        with col2:
            df = data["df"]
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_col = st.selectbox("Target column (optional):", ["None"] + numeric_cols)
            target_col = None if target_col == "None" else target_col

        result = encoding_advisor(data, target_col=target_col, model_type=model_type)
        st.info(result["summary"])

        for r in result["columns"]:
            risk_color = "green" if r["risk"] == "low" else "orange" if r["risk"] == "medium" else "red"
            with st.expander(f"**{r['column']}** → {r['strategy']} ({r['cardinality']} cardinality)"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Unique Values", r["n_unique"])
                col2.metric("Entropy", r["entropy"])
                col3.metric("Imbalance", f"{r['imbalance']}x")
                if r["target_corr"] is not None:
                    st.metric("Target Correlation", r["target_corr"])
                st.markdown(f"**Reason:** {r['reason']}")
                st.markdown(f"**sklearn:** `{r['sklearn_tip']}`")
                if r["warnings"]:
                    for w in r["warnings"]:
                        st.warning(w)
                st.code(r["code"], language="python")

        st.markdown("#### 📋 Pipeline Code")
        st.code(result["pipeline_code"], language="python")

    # ── Data Schema Validator ─────────────────────────────────────────────────
    elif feature == "📋 Data Schema Validator":
        st.markdown("### 📋 Data Schema Validator")

        mode = st.radio("Mode:", ["Auto-infer schema", "Upload schema JSON"], horizontal=True)

        if mode == "Upload schema JSON":
            schema_file = st.file_uploader("Upload schema JSON", type=["json"])
            if schema_file:
                import json

                schema = schema_from_dict(json.load(schema_file))
            else:
                st.info("Please upload a schema JSON file.")
                st.stop()
        else:
            schema = infer_schema(data)
            schema_json = schema_to_dict(schema)
            st.markdown("#### Inferred Schema")
            st.json(schema_json)

            import json

            st.download_button(
                label="⬇️ Download Schema JSON",
                data=json.dumps(schema_json, indent=2).encode("utf-8"),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_schema.json",
                mime="application/json",
            )

        result = validate_schema(data, schema)

        valid_color = "success" if result["valid"] else "error"
        if result["valid"]:
            st.success(result["summary"])
        else:
            st.error(result["summary"])

        for r in result["results"]:
            if r["status"] == "pass":
                with st.expander(f"✅ {r['column']}"):
                    for p in r["passed"]:
                        st.markdown(f"✓ {p}")
            else:
                with st.expander(f"❌ {r['column']}", expanded=True):
                    for e in r["errors"]:
                        st.error(e)
                    for w in r["warnings"]:
                        st.warning(w)
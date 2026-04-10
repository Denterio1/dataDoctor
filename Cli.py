"""
cli.py — dataDoctor Full CLI
==============================
Covers every operation the agent can perform:

  python cli.py inspect    <file>
  python cli.py clean      <file>
  python cli.py stats      <file>
  python cli.py missing    <file>
  python cli.py duplicates <file>
  python cli.py outliers   <file>
  python cli.py export     <file>
  python cli.py report     <file>
  python cli.py ml         <file>
  python cli.py prepare    <file>
  python cli.py relations  <file>
  python cli.py suggest    <file>
  python cli.py memory     [list|compare <file>|clear]
  python cli.py drift      <baseline> <current>
  python cli.py interactive
"""

from __future__ import annotations
import pandas as pd
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


from src.data.schema_validator import validate_schema, infer_schema, schema_to_dict, schema_from_dict, FieldSchema
from src.data.encoding_advisor import encoding_advisor
from src.data.correlation_network import build_correlation_network
from src.data.target_detector import detect_target
from src.data.feature_engineer import engineer_features
from src.data.loader        import load_csv
from src.data.analyzer      import full_report, shape, missing_values, duplicate_rows, basic_stats, detect_outliers
from src.data.cleaner       import handle_missing, remove_duplicates
from src.core.agent         import DataDoctor
from src.report             import generate_html_report
from src.data.ml_readiness  import ml_readiness
from src.data.relationships import detect_relationships
from src.data.preparator    import prepare_for_ml
from src.data.drift         import detect_drift
from src.data.memory        import save_snapshot, get_history, compare_last_two, get_all_files, clear_history, clear_all
from src.data.ai_suggestions import get_ai_suggestions

# ── Terminal colours ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"

SEPARATOR = c("─" * 60, DIM)


# ── Shared helpers ────────────────────────────────────────────────────────────

def banner() -> None:
    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║", CYAN) + c("        🩺  dataDoctor — Data Inspection Agent         ", BOLD + WHITE) + c("║", CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()


def usage() -> None:
    banner()
    print(c("  COMMANDS", BOLD + CYAN))
    print()
    cmds = [
        ("inspect   <file>",   "Full inspection: issues + cleaning + stats + report"),
        ("clean     <file>",   "Clean data (remove dupes + fill missing)"),
        ("stats     <file>",   "Show column statistics only"),
        ("missing   <file>",   "Show missing value counts per column"),
        ("duplicates <file>",  "Detect and show duplicate rows"),
        ("outliers  <file>",   "Detect outliers using IQR method"),
        ("export    <file>",   "Inspect + export cleaned CSV to disk"),
        ("report    <file>",   "Generate professional HTML report"),
        ("ml        <file>",   "ML Readiness Score"),
        ("prepare   <file>",   "Prepare data for ML (encode + scale)"),
        ("relations <file>",   "Detect column relationships"),
        ("suggest   <file>",   "AI-powered smart suggestions"),
        ("memory    [list|compare <file>|clear]", "Data memory & history"),
        ("drift     <baseline> <current>",        "Detect data drift between two files"),
        ("interactive",        "Guided interactive mode — no flags needed"),
    ]
    for cmd, desc in cmds:
        print(f"  {c('python cli.py', DIM)} {c(cmd, YELLOW)}  {c(desc, DIM)}")

    print()
    print(c("  OPTIONS", BOLD + CYAN))
    opts = [
        ("--strategy mean",   "Fill missing values with column mean  (default)"),
        ("--strategy median", "Fill missing values with column median"),
        ("--strategy mode",   "Fill missing values with most frequent value"),
        ("--strategy drop",   "Drop rows that contain any missing value"),
        ("--no-dedup",        "Do NOT remove duplicate rows"),
    ]
    for opt, desc in opts:
        print(f"  {c(opt, YELLOW)}  {c(desc, DIM)}")
    print()


def require_file(args: list[str]) -> str:
    if not args:
        print(c("  ✗ Error: no file path provided.", RED))
        sys.exit(1)
    path = args[0]
    if not os.path.exists(path):
        print(c(f"  ✗ File not found: {path}", RED))
        sys.exit(1)
    return path


def parse_options(args: list[str]) -> dict:
    opts = {"strategy": "mean", "dedup": True}
    i = 0
    while i < len(args):
        if args[i] == "--strategy" and i + 1 < len(args):
            opts["strategy"] = args[i + 1]
            i += 2
        elif args[i] == "--no-dedup":
            opts["dedup"] = False
            i += 1
        else:
            i += 1
    return opts


def safe_load(path: str) -> dict:
    try:
        return load_csv(path)
    except Exception as e:
        print(c(f"  ✗ Could not load file: {e}", RED))
        sys.exit(1)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_inspect(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Inspecting: {path}", CYAN))
    print(SEPARATOR)
    doctor = DataDoctor(remove_dupes=opts["dedup"], missing_strategy=opts["strategy"])
    result = doctor.inspect(path)
    print(result["summary"])

    outliers = detect_outliers(result["clean_data"])
    ml       = ml_readiness(result["clean_data"], outliers)
    save_snapshot(path, result["raw_analysis"], ml, outliers)
    print(c("  ✓ Snapshot saved to memory.", DIM))


def cmd_clean(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Cleaning: {path}", CYAN))
    print(c(f"  Strategy: {opts['strategy']}  |  Remove dupes: {opts['dedup']}", DIM))
    print(SEPARATOR)

    data          = safe_load(path)
    original_rows = len(data["df"])
    dupes_removed = 0

    if opts["dedup"]:
        data, dupes_removed = remove_duplicates(data)

    total_missing = sum(v for v in missing_values(data).values())
    changes = {}
    if total_missing > 0:
        data, changes = handle_missing(data, strategy=opts["strategy"])

    print()
    print(c("  ── Results ──────────────────────────────────────────", CYAN))
    print(f"  Original rows   : {c(str(original_rows), WHITE)}")
    print(f"  Rows after clean: {c(str(len(data['df'])), GREEN)}")

    if dupes_removed:
        print(f"  Duplicates removed : {c(str(dupes_removed), YELLOW)}")
    else:
        print(f"  Duplicates removed : {c('0 (none found)', DIM)}")

    if changes:
        print()
        print(c("  ── Missing Values Fixed ─────────────────────────────", CYAN))
        if "rows_dropped" in changes:
            print(f"  Rows dropped: {c(str(changes['rows_dropped']), YELLOW)}")
        else:
            for col, info in changes.items():
                print(f"  {c(col, WHITE)}: filled {c(str(info['filled']), YELLOW)} "
                      f"value(s) → {c(str(info['replacement']), GREEN)}")
    else:
        print(f"\n  {c('✓ No missing values found.', GREEN)}")
    print()


def cmd_stats(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Column Statistics: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    stats = basic_stats(data)
    shp   = shape(data)
    print(f"\n  {c(str(shp['rows']), WHITE)} rows  ×  {c(str(shp['columns']), WHITE)} columns\n")
    for col, s in stats.items():
        print(c(f"  ┌─ {col}", CYAN))
        print(f"  │  type   : {c(s['type'], YELLOW)}")
        print(f"  │  unique : {c(str(s['unique']), WHITE)}")
        print(f"  │  count  : {c(str(s['count']), WHITE)}")
        if s["type"] == "numeric":
            print(f"  │  min    : {c(str(s['min']), GREEN)}")
            print(f"  │  max    : {c(str(s['max']), GREEN)}")
            print(f"  │  mean   : {c(str(s['mean']), GREEN)}")
        else:
            print(f"  │  top    : {c(str(s['most_common']), GREEN)}")
        print(c("  └" + "─" * 40, DIM))
    print()


def cmd_missing(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Missing Values: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    mv    = missing_values(data)
    total = sum(mv.values())
    rows  = len(data["df"])
    print()
    if total == 0:
        print(c("  ✓ No missing values found — data is complete!", GREEN))
    else:
        print(f"  {c('Column', BOLD):<30} {c('Missing', BOLD):<12} {c('%', BOLD)}")
        print(c("  " + "─" * 50, DIM))
        for col, cnt in mv.items():
            pct  = cnt / rows * 100
            flag = c(f"{cnt}", YELLOW if cnt > 0 else GREEN)
            bar  = c("█" * int(pct / 5), YELLOW if cnt > 0 else GREEN)
            print(f"  {col:<28} {flag:<12} {pct:5.1f}%  {bar}")
    print(f"\n  Total missing cells: {c(str(total), YELLOW if total else GREEN)}")
    print()


def cmd_duplicates(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Duplicate Detection: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    df    = data["df"]
    count = duplicate_rows(data)
    print()
    if count == 0:
        print(c("  ✓ No duplicate rows found.", GREEN))
    else:
        print(c(f"  ⚠  Found {count} duplicate row(s):", YELLOW))
        print()
        dupes = df[df.duplicated(keep=False)]
        for idx, row in dupes.iterrows():
            vals = "  |  ".join(f"{col}={row[col]}" for col in df.columns[:4])
            print(c(f"  Row {idx + 1}: {vals} ...", YELLOW))
    print()


def cmd_outliers(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Outlier Detection: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    result = detect_outliers(data)

    print()
    if not result:
        print(c("  ✓ No outliers detected — data looks clean!", GREEN))
    else:
        print(c(f"  ⚠  Outliers found in {len(result)} column(s):\n", YELLOW))
        for col, info in result.items():
            print(c(f"  ┌─ {col}", CYAN))
            print(f"  │  count  : {c(str(info['count']), YELLOW)}")
            print(f"  │  safe range : {c(str(info['lower']), GREEN)} → {c(str(info['upper']), GREEN)}")
            print(f"  │  values : {c(str(info['values']), RED)}")
            print(c("  └" + "─" * 40, DIM))
    print()


def cmd_export(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Export: {path}", CYAN))
    print(SEPARATOR)
    doctor = DataDoctor(remove_dupes=opts["dedup"], missing_strategy=opts["strategy"])
    result = doctor.inspect(path)
    print(result["summary"])
    base, ext = os.path.splitext(path)
    out_path  = f"{base}_cleaned{ext}"
    result["clean_data"]["df"].to_csv(out_path, index=False, encoding="utf-8")
    print(c(f"  ✓ Cleaned file saved to: {out_path}", GREEN))
    print()


def cmd_report(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Generating HTML report for: {path}", CYAN))
    print(SEPARATOR)
    doctor   = DataDoctor(remove_dupes=opts["dedup"], missing_strategy=opts["strategy"])
    result   = doctor.inspect(path)
    base, _  = os.path.splitext(path)
    out_path = f"{base}_report.html"
    generate_html_report(result, out_path)
    print(c(f"  ✓ Report saved to: {out_path}", GREEN))
    print(c(f"  ✓ Open it in your browser!", CYAN))
    print()


def cmd_ml(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  ML Readiness Check: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    outliers = detect_outliers(data)
    result   = ml_readiness(data, outliers)

    score       = result["score"]
    grade       = result["grade"]
    score_color = GREEN if score >= 75 else YELLOW if score >= 50 else RED

    print()
    print(f"  {c('ML Readiness Score', BOLD)}  {c(str(score), score_color + BOLD)}{c('/100', DIM)}  {c(f'[ {grade} ]', score_color)}")
    print()
    print(f"  {c(result['summary'], DIM)}")
    print()

    for check in result["checks"]:
        if check["status"] == "pass":
            icon  = c("✓", GREEN)
            color = DIM
        elif check["status"] == "warn":
            icon  = c("⚠", YELLOW)
            color = YELLOW
        else:
            icon  = c("✗", RED)
            color = RED

        bar_w   = int(check["points"] / check["max"] * 20)
        bar     = c("█" * bar_w + "░" * (20 - bar_w), color)
        pts_str = c(f"{check['points']}/{check['max']}", color)

        print(f"  {icon}  {c(check['name'], BOLD):<30} {pts_str}  {bar}")
        print(f"     {c(check['detail'], DIM)}")
        print()


def cmd_prepare(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Preparing for ML: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    prepared, log = prepare_for_ml(data, missing_strategy=opts["strategy"], scale=True, encode=True)

    print()
    print(c("  ── Pipeline Results ─────────────────────────────────", CYAN))
    print(f"  Duplicates removed : {c(str(log['duplicates_removed']), YELLOW)}")

    if log.get("missing_filled"):
        for col, info in log["missing_filled"].items():
            print(f"  {c(col, WHITE)}: filled {c(str(info['filled']), YELLOW)} value(s) → {c(str(info['value']), GREEN)}")

    if log.get("columns_dropped"):
        for col, reason in log["columns_dropped"]:
            print(f"  Dropped {c(col, YELLOW)} ({reason})")

    if log.get("encoded"):
        print(f"  Encoded {c(str(len(log['encoded'])), YELLOW)} text column(s)")

    if log.get("scaled"):
        print(f"  Scaled  {c(str(len(log['scaled'])), YELLOW)} numeric column(s)")

    shp = log["final_shape"]
    print()
    print(f"  {c('Final shape:', BOLD)} {c(str(shp['rows']), GREEN)} rows × {c(str(shp['columns']), GREEN)} columns")

    base, ext = os.path.splitext(path)
    out_path  = f"{base}_ready.csv"
    prepared["df"].to_csv(out_path, index=False, encoding="utf-8")

    print()
    print(c(f"  ✓ ML-ready file saved to: {out_path}", GREEN))
    print()


def cmd_relationships(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Column Relationships: {path}", CYAN))
    print(SEPARATOR)

    data    = safe_load(path)
    results = detect_relationships(data, threshold=0.4)

    print()
    if not results:
        print(c("  ✓ No strong relationships found between columns.", GREEN))
    else:
        print(c(f"  Found {len(results)} relationship(s):\n", CYAN))
        for r in results:
            strength = r["strength"]
            if strength >= 0.8:
                level = c("● strong", GREEN)
            elif strength >= 0.6:
                level = c("● moderate", YELLOW)
            else:
                level = c("● weak", DIM)

            bar_w = int(strength * 20)
            bar   = c("█" * bar_w + "░" * (20 - bar_w), GREEN if strength >= 0.8 else YELLOW)

            print(f"  {c(r['col_a'], WHITE)} ↔ {c(r['col_b'], WHITE)}")
            print(f"     {level}  {bar}  {c(str(strength), WHITE)}")
            print(f"     {c(r['method'], DIM)} · {c(r['type'], DIM)} · {c(r['direction'], DIM)}")
            print()
    print()

def cmd_encoding(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Smart Encoding Advisor: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)

    # detect target column
    df = data["df"]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols     = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    print()
    print(c("  Model type:", BOLD))
    print(f"  1. Tree / Boosting   (XGBoost, RandomForest, LightGBM)")
    print(f"  2. Linear            (LogisticRegression, LinearSVC)")
    print(f"  3. Neural Network    (MLP, Keras)")
    model_choice = input(c("\n  Your choice (1-3, default=1): ", YELLOW)).strip() or "1"
    model_map = {"1": "tree", "2": "linear", "3": "neural"}
    model_type = model_map.get(model_choice, "tree")

    target_col = None
    if numeric_cols:
        print()
        print(c("  Numeric columns (potential targets):", BOLD))
        for i, col in enumerate(numeric_cols, 1):
            print(f"  {i}. {col}")
        print(f"  0. No target column")
        t_choice = input(c("\n  Select target column (0 to skip): ", YELLOW)).strip()
        if t_choice.isdigit() and 1 <= int(t_choice) <= len(numeric_cols):
            target_col = numeric_cols[int(t_choice) - 1]

    print()
    print(c("  Analyzing...", DIM))

    result = encoding_advisor(data, target_col=target_col, model_type=model_type)

    print()
    print(c(f"  {result['summary']}", CYAN))
    print()

    for r in result["columns"]:
        risk_color = GREEN if r["risk"] == "low" else YELLOW if r["risk"] == "medium" else RED

        print(c(f"  ┌─ {r['column']}", CYAN))
        print(f"  │  strategy   : {c(r['strategy'], risk_color + BOLD)}")
        print(f"  │  cardinality: {c(str(r['n_unique']), WHITE)} unique ({r['cardinality']})")
        print(f"  │  entropy    : {c(str(r['entropy']), DIM)}")
        print(f"  │  imbalance  : {c(str(r['imbalance']) + 'x', DIM)}")
        if r["target_corr"] is not None:
            print(f"  │  target corr: {c(str(r['target_corr']), YELLOW)}")
        if r["is_ordinal"]:
            print(f"  │  order      : {c(' < '.join(r['ordinal_scale']), GREEN)}")
        print(f"  │  reason     : {c(r['reason'], DIM)}")
        print(f"  │  sklearn    : {c(r['sklearn_tip'], DIM)}")
        if r["warnings"]:
            for w in r["warnings"]:
                print(f"  │  {c('⚠ ' + w, YELLOW)}")
        print(c("  └" + "─" * 50, DIM))
        print()

    print(c("  ── Pipeline Code ────────────────────────────────────", CYAN))
    print()
    for line in result["pipeline_code"].splitlines():
        print(f"  {c(line, WHITE)}")
    print()

def cmd_schema(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Schema Validator: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)

    print()
    print(c("  What do you want to do?", BOLD))
    print(f"  1. Auto-infer schema from data and validate")
    print(f"  2. Load schema from JSON file")
    print(f"  3. Export inferred schema to JSON")
    choice = input(c("\n  Your choice (1-3): ", YELLOW)).strip()

    if choice == "3":
        schema   = infer_schema(data)
        base, _  = os.path.splitext(path)
        out_path = f"{base}_schema.json"
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(schema_to_dict(schema), f, indent=2)
        print(c(f"\n  ✓ Schema exported to: {out_path}", GREEN))
        print(c("  Edit it and re-run with option 2 to validate.", DIM))
        print()
        return

    if choice == "2":
        schema_path = input(c("  Schema JSON file path: ", YELLOW)).strip().strip('"')
        try:
            import json
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = schema_from_dict(json.load(f))
        except Exception as e:
            print(c(f"\n  ✗ Could not load schema: {e}", RED))
            return
    else:
        schema = infer_schema(data)

    result = validate_schema(data, schema)

    print()
    valid_color = GREEN if result["valid"] else RED
    print(c(f"  {result['summary']}", valid_color))
    print()
    print(f"  Coverage : {c(str(result['coverage']) + '%', CYAN)} of columns validated")
    if result["uncovered"]:
        print(f"  Uncovered: {c(', '.join(result['uncovered']), DIM)}")
    print()

    for r in result["results"]:
        if r["status"] == "pass":
            icon  = c("✓", GREEN)
            color = GREEN
        elif r["status"] == "missing":
            icon  = c("?", YELLOW)
            color = YELLOW
        else:
            icon  = c("✗", RED)
            color = RED

        print(f"  {icon}  {c(r['column'], color + BOLD)}")

        for p in r["passed"]:
            print(f"     {c('✓ ' + p, DIM)}")
        for w in r["warnings"]:
            print(f"     {c('⚠ ' + w, YELLOW)}")
        for e in r["errors"]:
            print(f"     {c('✗ ' + e, RED)}")
        print()
    print()


def cmd_suggest(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  AI Suggestions: {path}", CYAN))
    print(SEPARATOR)

    api_key  = os.environ.get("DATADOCTOR_API_KEY", "")
    base_url = os.environ.get("DATADOCTOR_BASE_URL", "https://api.groq.com/openai/v1")
    model    = os.environ.get("DATADOCTOR_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        print(c("\n  ✗ No API key found.", RED))
        print(c("  Add these to your .env file:\n", DIM))
        print(c("  DATADOCTOR_API_KEY=your_key", YELLOW))
        print(c("  DATADOCTOR_BASE_URL=https://api.groq.com/openai/v1", YELLOW))
        print(c("  DATADOCTOR_MODEL=llama-3.3-70b-versatile", YELLOW))
        print(c("\n  Supported providers: Groq (free), Gemini (free), OpenRouter, OpenAI", DIM))
        return

    data     = safe_load(path)
    analysis = full_report(data)
    outliers = detect_outliers(data)
    rels     = detect_relationships(data, threshold=0.4)
    ml       = ml_readiness(data, outliers)

    print()
    print(c(f"  Provider : {base_url}", DIM))
    print(c(f"  Model    : {model}", DIM))
    print(c("  Thinking...\n", DIM))

    try:
        import requests
        suggestions = get_ai_suggestions(
            api_key       = api_key,
            base_url      = base_url,
            model         = model,
            source        = path,
            shape         = analysis["shape"],
            missing       = analysis["missing_values"],
            dupes         = analysis["duplicate_rows"],
            stats         = analysis["column_stats"],
            outliers      = outliers,
            relationships = rels,
            ml_score      = ml,
        )
        print(c("  ── AI Suggestions ───────────────────────────────────", CYAN))
        print()
        for line in suggestions.splitlines():
            if line.strip():
                print(f"  {c(line, WHITE)}")
    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))
    print()


def cmd_memory(args: list[str]) -> None:
    banner()
    print(c("  Data Memory", CYAN))
    print(SEPARATOR)
    print()

    sub = args[0].lower() if args else "list"

    if sub == "list":
        files = get_all_files()
        if not files:
            print(c("  No files in memory yet.", DIM))
            print(c("  Run 'inspect' on any file to start tracking.", DIM))
        else:
            print(c(f"  {len(files)} file(s) tracked:\n", CYAN))
            for f in files:
                score_color = GREEN if f["best_score"] >= 75 else YELLOW if f["best_score"] >= 50 else RED
                print(f"  {c(f['file_name'], WHITE)}")
                print(f"     inspections : {c(str(f['inspections']), CYAN)}")
                print(f"     last seen   : {c(f['last_seen'][:19].replace('T', ' '), DIM)}")
                print(f"     best score  : {c(str(f['best_score']), score_color)}/100")
                print()

    elif sub == "compare":
        if len(args) < 2:
            print(c("  Usage: python cli.py memory compare <file>", RED))
            return
        path    = args[1]
        changes = compare_last_two(path)
        if not changes:
            print(c("  Need at least 2 inspections to compare.", YELLOW))
            print(c("  Run 'inspect' on this file again later.", DIM))
            return

        print(c(f"  Comparing last 2 inspections of: {changes['file']}\n", CYAN))
        print(f"  {c('Previous', DIM)} : {changes['old_time']}")
        print(f"  {c('Current',  DIM)} : {changes['new_time']}\n")

        def _show(label, info):
            delta = info["delta"]
            if delta == 0:
                icon, col = "→", DIM
            elif info["improved"]:
                icon, col = "↑", GREEN
            else:
                icon, col = "↓", RED
            sign = "+" if delta > 0 else ""
            print(f"  {label:<20} {info['old']} → {c(str(info['new']), col)}  {c(f'{sign}{delta}', col)}  {icon}")

        _show("Rows",         changes["rows"])
        _show("Columns",      changes["columns"])
        _show("Missing",      changes["missing"])
        _show("Duplicates",   changes["duplicates"])
        _show("ML Score",     changes["ml_score"])
        _show("Outlier cols", changes["outliers"])

    elif sub == "clear":
        if len(args) < 2:
            n = clear_all()
            print(c(f"  ✓ Cleared all memory ({n} snapshots).", GREEN))
        else:
            n = clear_history(args[1])
            print(c(f"  ✓ Cleared {n} snapshot(s) for {args[1]}.", GREEN))

    else:
        print(c(f"  Unknown sub-command: {sub}", RED))
        print(c("  Usage: memory list | memory compare <file> | memory clear", DIM))

    print()


def cmd_drift(args: list[str]) -> None:
    if len(args) < 2:
        print(c("\n  Usage: python cli.py drift <baseline_file> <current_file>\n", RED))
        return

    base_path = args[0]
    curr_path = args[1]

    banner()
    print(c(f"  Drift Detection", CYAN))
    print(c(f"  Baseline : {base_path}", DIM))
    print(c(f"  Current  : {curr_path}", DIM))
    print(SEPARATOR)

    base   = safe_load(base_path)
    curr   = safe_load(curr_path)
    result = detect_drift(base, curr)

    sev_color = GREEN if result["severity"] == "none" else YELLOW if result["severity"] == "low" else RED

    print()
    print(f"  {c('Severity', BOLD)} : {c(result['severity'].upper(), sev_color)}")
    print(f"  {c('Summary',  BOLD)} : {c(result['summary'], DIM)}")
    print()
    print(f"  Baseline : {result['base_shape']['rows']} rows × {result['base_shape']['cols']} cols")
    print(f"  Current  : {result['new_shape']['rows']} rows × {result['new_shape']['cols']} cols")
    print()

    if result["drifted_columns"]:
        print(c(f"  ── Drifted Columns ({len(result['drifted_columns'])}) ─────────────────────────", RED))
        print()
        for d in result["drifted_columns"]:
            sev = d["severity"]
            sc  = RED if sev == "high" else YELLOW if sev == "medium" else DIM
            print(c(f"  ┌─ {d['column']}  [{sev}]", sc))
            for issue in d["issues"]:
                print(f"  │  {c(issue, WHITE)}")
            print(c("  └" + "─" * 40, DIM))
            print()

    if result["stable_columns"]:
        print(c(f"  ── Stable Columns ({len(result['stable_columns'])}) ──────────────────────────", GREEN))
        print(f"  {c(', '.join(result['stable_columns']), DIM)}")

    print()

def cmd_engineer(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Auto Feature Engineering: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    enriched, log = engineer_features(data)

    total = sum(len(v) for v in log.values())
    print()
    print(c(f"  Generated {total} new feature(s):\n", CYAN))

    if log["date"]:
        print(c("  ── Date Features ────────────────────────────────────", CYAN))
        for f in log["date"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["numeric"]:
        print(c("  ── Numeric Features ─────────────────────────────────", CYAN))
        for f in log["numeric"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["interaction"]:
        print(c("  ── Interaction Features ─────────────────────────────", CYAN))
        for f in log["interaction"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["text"]:
        print(c("  ── Text Features ────────────────────────────────────", CYAN))
        for f in log["text"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    base, ext = os.path.splitext(path)
    out_path  = f"{base}_engineered.csv"
    enriched["df"].to_csv(out_path, index=False, encoding="utf-8")

    print(c(f"  ✓ Saved to: {out_path}", GREEN))
    print(f"  Original : {len(data['df'].columns)} columns")
    print(f"  New      : {c(str(len(enriched['df'].columns)), GREEN)} columns")
    print()

def cmd_target(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Target Column Detection: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    result = detect_target(data)

    conf_color = GREEN if result["confidence"] == "high" else YELLOW if result["confidence"] == "medium" else RED
    task_color = CYAN if result["task_type"] == "classification" else YELLOW if result["task_type"] == "regression" else DIM

    print()
    print(f"  {c('Recommended target', BOLD)} : {c(result['recommended'], GREEN)}")
    print(f"  {c('Task type', BOLD)}          : {c(result['task_type'].upper(), task_color)}")
    print(f"  {c('Confidence', BOLD)}         : {c(result['confidence'].upper(), conf_color)}")
    print(f"  {c('Reason', BOLD)}             : {c(result['reason'], DIM)}")
    print()

    print(c("  ── All Candidates ───────────────────────────────────", CYAN))
    print()
    for i, cand in enumerate(result["candidates"]):
        rank  = c(f"#{i+1}", YELLOW if i == 0 else DIM)
        score = c(str(cand["score"]), GREEN if i == 0 else DIM)
        print(f"  {rank}  {c(cand['column'], WHITE if i == 0 else DIM):<25} score={score}")
        if cand["signals"]:
            print(f"      {c(' | '.join(cand['signals'][:2]), DIM)}")
        print()

def cmd_network(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Correlation Network: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    result = build_correlation_network(data, threshold=0.3)

    print()
    print(f"  {c(result['summary'], DIM)}")
    print()
    print(f"  Nodes : {c(str(len(result['nodes'])), CYAN)} columns")
    print(f"  Edges : {c(str(result['stats']['total_edges']), CYAN)} relationships")
    print(f"  Strong: {c(str(result['stats']['strong']), GREEN)}  "
          f"Moderate: {c(str(result['stats']['moderate']), YELLOW)}  "
          f"Weak: {c(str(result['stats']['weak']), DIM)}")
    print()

    if result["edges"]:
        print(c("  ── Relationships (sorted by strength) ───────────────", CYAN))
        print()
        for e in result["edges"]:
            strength = e["strength"]
            if strength >= 0.7:
                color = GREEN
                level = "strong"
            elif strength >= 0.5:
                color = YELLOW
                level = "moderate"
            else:
                color = DIM
                level = "weak"

            bar = c("█" * int(strength * 20) + "░" * (20 - int(strength * 20)), color)
            print(f"  {c(e['source'], WHITE)} ↔ {c(e['target'], WHITE)}")
            print(f"     {bar}  {c(str(strength), color)}  {c(level, color)}  {c(e['method'], DIM)}")
            print()
    else:
        print(c("  ✓ No significant relationships found.", GREEN))
    print()


def cmd_interactive() -> None:
    banner()
    print(c("  Interactive Mode — type 'exit' anytime to quit.", CYAN))
    print(SEPARATOR)

    while True:
        print()

        while True:
            path = input(c("  File path — CSV, Excel, JSON (or 'exit'): ", YELLOW)).strip().strip('"')
            if path.lower() == "exit":
                print(c("\n  Goodbye! 👋\n", GREEN))
                return
            if os.path.exists(path):
                break
            print(c(f"  ✗ File not found: {path}. Try again.", RED))

        print()
        print(c("  What do you want to do?", BOLD))
        options = {
            "1":  ("Full inspection report",     "inspect"),
            "2":  ("Show column statistics",     "stats"),
            "3":  ("Show missing values",        "missing"),
            "4":  ("Detect duplicates",          "duplicates"),
            "5":  ("Detect outliers",            "outliers"),
            "6":  ("Clean and save to new file", "export"),
            "7":  ("Generate HTML report",       "report"),
            "8":  ("ML Readiness Score",         "ml"),
            "9":  ("Prepare data for ML",        "prepare"),
            "10": ("Column relationships",       "relations"),
            "11": ("AI smart suggestions",       "suggest"),
            "12": ("Data memory & history",      "memory"),
            "13": ("Drift detection",            "drift"),
            "14": ("Auto feature engineering", "engineer"),
            "15": ("Target column detection", "target"),
            "16": ("Correlation network", "network"),
            "17": ("Smart Encoding Advisor", "encoding"),
            "18": ("Schema Validator", "schema"),

        }
        for key, (label, _) in options.items():
            print(f"  {c(key, YELLOW)}. {label}")

        choice = input(c("\n  Your choice (1-18): ", YELLOW)).strip()
        action = options.get(choice, ("", "inspect"))[1]

        strategy = "mean"
        if action in ("inspect", "export", "report", "prepare", "suggest"):
            print()
            print(c("  Missing value strategy:", BOLD))
            strategies = {"1": "mean", "2": "median", "3": "mode", "4": "drop"}
            for k, v in strategies.items():
                print(f"  {c(k, YELLOW)}. {v}")
            sc = input(c("\n  Your choice (1-4, default=1): ", YELLOW)).strip()
            strategy = strategies.get(sc, "mean")

        print()
        print(SEPARATOR)

        file_args = [path]
        if action == "inspect":
            file_args += ["--strategy", strategy]
            cmd_inspect(file_args)
        elif action == "stats":
            cmd_stats(file_args)
        elif action == "missing":
            cmd_missing(file_args)
        elif action == "duplicates":
            cmd_duplicates(file_args)
        elif action == "outliers":
            cmd_outliers(file_args)
        elif action == "export":
            file_args += ["--strategy", strategy]
            cmd_export(file_args)
        elif action == "report":
            file_args += ["--strategy", strategy]
            cmd_report(file_args)
        elif action == "ml":
            cmd_ml(file_args)
        elif action == "prepare":
            file_args += ["--strategy", strategy]
            cmd_prepare(file_args)
        elif action == "relations":
            cmd_relationships(file_args)
        elif action == "suggest":
            cmd_suggest(file_args)
        elif action == "memory":
            cmd_memory([])
        elif action == "drift":
            print(c("\n  For drift: python cli.py drift <baseline> <current>\n", YELLOW))
        elif action == "engineer":
            cmd_engineer(file_args)
        elif action == "target":
            cmd_target(file_args)
        elif action == "network":
            cmd_network(file_args)
        elif action == "encoding":
            cmd_encoding(file_args)
        elif action == "schema":
            cmd_schema(file_args)

        print(c("  ✓ Done! Ready for next operation.", GREEN))
        print(SEPARATOR)


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "inspect":    cmd_inspect,
    "clean":      cmd_clean,
    "stats":      cmd_stats,
    "missing":    cmd_missing,
    "duplicates": cmd_duplicates,
    "outliers":   cmd_outliers,
    "export":     cmd_export,
    "report":     cmd_report,
    "ml":         cmd_ml,
    "prepare":    cmd_prepare,
    "relations":  cmd_relationships,
    "suggest":    cmd_suggest,
    "memory":     cmd_memory,
    "drift":      cmd_drift,
    "engineer":   cmd_engineer,
    "target":     cmd_target,
    "network":    cmd_network,
    "encoding":   cmd_encoding,
    "schema":     cmd_schema,
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        usage()
        return

    cmd = args[0].lower()

    if cmd == "interactive":
        cmd_interactive()
        return

    if cmd == "memory":
        cmd_memory(args[1:])
        return

    if cmd == "drift":
        cmd_drift(args[1:])
        return

    if cmd not in COMMANDS:
        print(c(f"\n  ✗ Unknown command: '{cmd}'", RED))
        usage()
        sys.exit(1)

    COMMANDS[cmd](args[1:])


if __name__ == "__main__":
    main()
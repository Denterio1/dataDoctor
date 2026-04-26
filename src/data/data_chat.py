"""
data_chat.py — Data Chat: Talk to your data in natural language (v0.5.0)

Allows users to ask questions about their dataset in plain English or Arabic:
    "What is the average salary?"
    "Which region has the most sales?"
    "Show me rows where quantity > 5"
    "What columns have missing values?"
    "Which product sold the most?"

Architecture:
    QueryParser       — Parse natural language to structured query
    DataQueryEngine   — Execute structured queries on dataframe
    ResponseFormatter — Format results into human-readable text
    DataChatBot       — Main orchestrator
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedQuery:
    """Structured representation of a natural language query."""
    intent:     str              # aggregate | filter | sort | describe | compare | count
    columns:    list[str]        # columns involved
    conditions: list[dict]       # filter conditions
    aggregation:str              # sum | mean | max | min | count | std
    group_by:   str | None       # group by column
    order_by:   str | None       # sort by column
    ascending:  bool             # sort direction
    limit:      int              # max rows to return
    raw_query:  str              # original query text


@dataclass
class QueryResult:
    """Result of executing a query."""
    success:   bool
    data:      Any               # DataFrame, scalar, or dict
    answer:    str               # human-readable answer
    query:     ParsedQuery
    chart_type:str | None = None # "bar" | "line" | "pie" | None
    error:     str = ""


@dataclass
class ChatMessage:
    """A single message in the chat history."""
    role:    str       # "user" | "assistant"
    content: str
    result:  QueryResult | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Query Parser
# ══════════════════════════════════════════════════════════════════════════════

class QueryParser:
    """
    Parse natural language queries into structured QueryParser objects.

    Supports English and Arabic patterns.
    Handles: aggregations, filters, sorts, describes, counts.
    """

    # ── Intent patterns ───────────────────────────────────────────────────────
    AGGREGATE_PATTERNS = [
        r'\b(average|avg|mean|متوسط)\b',
        r'\b(sum|total|مجموع|إجمالي)\b',
        r'\b(max|maximum|highest|أعلى|أكثر)\b',
        r'\b(min|minimum|lowest|أدنى|أقل)\b',
        r'\b(count|number of|كم عدد)\b',
        r'\b(std|standard deviation|الانحراف المعياري)\b',
    ]

    FILTER_PATTERNS = [
        r'\b(where|filter|show.*where|أين|عرض.*حيث)\b',
        r'\b(greater than|more than|above|أكبر من|أكثر من)\b',
        r'\b(less than|below|under|أقل من|أصغر من)\b',
        r'\b(equal|equals|is|يساوي|=)\b',
        r'\b(contains|like|يحتوي)\b',
    ]

    SORT_PATTERNS = [
        r'\b(sort|order|rank|ترتيب|رتب)\b',
        r'\b(top|best|highest|أعلى|أفضل)\b',
        r'\b(bottom|worst|lowest|أدنى|أسوأ)\b',
    ]

    DESCRIBE_PATTERNS = [
        r'\b(describe|summary|stats|statistics|وصف|ملخص)\b',
        r'\b(what is|what are|tell me about|ما هو|أخبرني)\b',
        r'\b(missing|null|empty|مفقود|فارغ)\b',
        r'\b(unique|distinct|فريد|مختلف)\b',
    ]

    AGGREGATION_MAP = {
        "average": "mean", "avg": "mean", "mean": "mean", "متوسط": "mean",
        "sum": "sum", "total": "sum", "مجموع": "sum", "إجمالي": "sum",
        "max": "max", "maximum": "max", "highest": "max", "أعلى": "max",
        "min": "min", "minimum": "min", "lowest": "min", "أدنى": "min",
        "count": "count", "number": "count", "كم": "count",
        "std": "std", "standard": "std",
    }

    def parse(self, query: str, df: pd.DataFrame) -> ParsedQuery:
        """Parse a natural language query."""
        q_lower = query.lower().strip()
        columns = self._extract_columns(q_lower, df.columns.tolist())
        intent  = self._detect_intent(q_lower)
        agg     = self._detect_aggregation(q_lower)
        conds   = self._detect_conditions(q_lower, df, columns)
        group   = self._detect_group_by(q_lower, df.columns.tolist(), columns)
        order, asc, limit = self._detect_sort(q_lower, df.columns.tolist())

        return ParsedQuery(
            intent      = intent,
            columns     = columns,
            conditions  = conds,
            aggregation = agg,
            group_by    = group,
            order_by    = order,
            ascending   = asc,
            limit       = limit,
            raw_query   = query,
        )

    def _extract_columns(self, query: str, col_names: list[str]) -> list[str]:
        """Find column names mentioned in the query."""
        found = []
        for col in col_names:
            col_lower = col.lower().replace("_", " ")
            if col_lower in query or col.lower() in query:
                found.append(col)
        return found if found else col_names[:3]

    def _detect_intent(self, query: str) -> str:
        if any(re.search(p, query) for p in self.DESCRIBE_PATTERNS):
            return "describe"
        if any(re.search(p, query) for p in self.AGGREGATE_PATTERNS):
            return "aggregate"
        if any(re.search(p, query) for p in self.FILTER_PATTERNS):
            return "filter"
        if any(re.search(p, query) for p in self.SORT_PATTERNS):
            return "sort"
        if re.search(r'\b(compare|comparison|vs|versus|قارن)\b', query):
            return "compare"
        return "describe"

    def _detect_aggregation(self, query: str) -> str:
        for keyword, func in self.AGGREGATION_MAP.items():
            if keyword in query:
                return func
        return "mean"

    def _detect_conditions(self, query: str, df: pd.DataFrame, cols: list[str]) -> list[dict]:
        conditions = []
        # Pattern: column > value
        gt_match = re.search(r'(\w+)\s*(?:>|greater than|more than|above)\s*([\d.]+)', query)
        if gt_match:
            col_name = self._match_column(gt_match.group(1), df.columns.tolist())
            if col_name:
                conditions.append({"column": col_name, "op": ">", "value": float(gt_match.group(2))})

        lt_match = re.search(r'(\w+)\s*(?:<|less than|below|under)\s*([\d.]+)', query)
        if lt_match:
            col_name = self._match_column(lt_match.group(1), df.columns.tolist())
            if col_name:
                conditions.append({"column": col_name, "op": "<", "value": float(lt_match.group(2))})

        eq_match = re.search(r'(\w+)\s*(?:=|equals?|is)\s*["\']?(\w+)["\']?', query)
        if eq_match:
            col_name = self._match_column(eq_match.group(1), df.columns.tolist())
            if col_name:
                conditions.append({"column": col_name, "op": "==", "value": eq_match.group(2)})

        return conditions

    def _detect_group_by(self, query: str, all_cols: list[str], mentioned: list[str]) -> str | None:
        gb_match = re.search(r'by\s+(\w+)', query)
        if gb_match:
            col = self._match_column(gb_match.group(1), all_cols)
            if col:
                return col
        # If aggregation + column = likely group by
        if re.search(r'\b(per|each|every|by|لكل)\b', query):
            cat_cols = [c for c in mentioned if c in all_cols]
            if cat_cols:
                return cat_cols[0]
        return None

    def _detect_sort(self, query: str, all_cols: list[str]) -> tuple[str | None, bool, int]:
        order_col = None
        ascending = False
        limit     = 10

        # Top N
        top_match = re.search(r'top\s+(\d+)', query)
        if top_match:
            limit = int(top_match.group(1))

        # Sort direction
        if re.search(r'\b(ascending|asc|lowest|bottom|أقل)\b', query):
            ascending = True

        # Find sort column
        sort_match = re.search(r'(?:sort|order|rank)\s+by\s+(\w+)', query)
        if sort_match:
            order_col = self._match_column(sort_match.group(1), all_cols)

        return order_col, ascending, limit

    def _match_column(self, word: str, columns: list[str]) -> str | None:
        word = word.lower()
        for col in columns:
            if word == col.lower() or word in col.lower() or col.lower() in word:
                return col
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Data Query Engine
# ══════════════════════════════════════════════════════════════════════════════

class DataQueryEngine:
    """Execute parsed queries on a pandas DataFrame."""

    def execute(self, query: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        """Execute a parsed query and return result."""
        try:
            if query.intent == "describe":
                return self._execute_describe(query, df)
            elif query.intent == "aggregate":
                return self._execute_aggregate(query, df)
            elif query.intent == "filter":
                return self._execute_filter(query, df)
            elif query.intent == "sort":
                return self._execute_sort(query, df)
            elif query.intent == "compare":
                return self._execute_compare(query, df)
            else:
                return self._execute_describe(query, df)
        except Exception as e:
            return QueryResult(
                success = False,
                data    = None,
                answer  = f"Could not process query: {str(e)}",
                query   = query,
                error   = str(e),
            )

    def _apply_conditions(self, df: pd.DataFrame, conditions: list[dict]) -> pd.DataFrame:
        for cond in conditions:
            col, op, val = cond["column"], cond["op"], cond["value"]
            if col not in df.columns:
                continue
            if op == ">":
                df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
            elif op == "<":
                df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
            elif op == "==":
                df = df[df[col].astype(str).str.lower() == str(val).lower()]
        return df

    def _execute_describe(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        raw = q.raw_query.lower()

        # Missing values question
        if re.search(r'\b(missing|null|empty|مفقود)\b', raw):
            mv = df.isnull().sum()
            mv = mv[mv > 0]
            if mv.empty:
                answer = "Great news! Your dataset has no missing values."
                data   = {}
            else:
                data   = mv.to_dict()
                lines  = [f"• **{col}**: {cnt} missing ({cnt/len(df)*100:.1f}%)" for col, cnt in mv.items()]
                answer = f"Found missing values in {len(mv)} column(s):\n" + "\n".join(lines)
            return QueryResult(success=True, data=data, answer=answer, query=q)

        # Unique values question
        if re.search(r'\b(unique|distinct|فريد)\b', raw):
            col = q.columns[0] if q.columns else df.columns[0]
            if col in df.columns:
                unique_vals = df[col].dropna().unique().tolist()[:20]
                answer = f"**{col}** has {df[col].nunique()} unique values:\n{', '.join(str(v) for v in unique_vals)}"
                return QueryResult(success=True, data=unique_vals, answer=answer, query=q)

        # General describe
        col = q.columns[0] if q.columns else None
        if col and col in df.columns:
            s = df[col].dropna()
            if pd.api.types.is_numeric_dtype(s):
                data   = {"mean": round(s.mean(), 4), "std": round(s.std(), 4),
                          "min": s.min(), "max": s.max(), "median": s.median()}
                answer = (f"**{col}** statistics:\n"
                          f"• Mean: {data['mean']:,.2f}\n"
                          f"• Median: {data['median']:,.2f}\n"
                          f"• Min: {data['min']:,.2f} | Max: {data['max']:,.2f}\n"
                          f"• Std Dev: {data['std']:,.2f}")
            else:
                top = s.value_counts().head(5)
                data   = top.to_dict()
                answer = f"**{col}** top values:\n" + "\n".join(f"• {k}: {v}" for k, v in data.items())
            return QueryResult(success=True, data=data, answer=answer, query=q, chart_type="bar")

        # Full summary
        data   = {"rows": len(df), "columns": len(df.columns), "missing": int(df.isnull().sum().sum())}
        answer = (f"Dataset overview:\n"
                  f"• **{data['rows']:,}** rows × **{data['columns']}** columns\n"
                  f"• **{data['missing']}** missing cells\n"
                  f"• Columns: {', '.join(df.columns.tolist()[:8])}")
        return QueryResult(success=True, data=data, answer=answer, query=q)

    def _execute_aggregate(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        df_filtered = self._apply_conditions(df, q.conditions)
        func        = q.aggregation
        col         = next((c for c in q.columns if c in df.columns and
                           pd.api.types.is_numeric_dtype(df[c])), None)

        if not col:
            col = df.select_dtypes(include="number").columns[0] if not df.select_dtypes(include="number").empty else None

        if not col:
            return QueryResult(success=False, data=None,
                               answer="No numeric column found for aggregation.", query=q)

        if q.group_by and q.group_by in df.columns:
            grouped = df_filtered.groupby(q.group_by)[col].agg(func).round(4)
            data    = grouped.to_dict()
            top     = grouped.sort_values(ascending=False).head(5)
            lines   = [f"• **{k}**: {v:,.2f}" for k, v in top.items()]
            answer  = (f"**{func.title()}** of **{col}** by **{q.group_by}**:\n" +
                       "\n".join(lines))
            chart_type = "bar"
        else:
            func_map = {"mean": "mean", "sum": "sum", "max": "max",
                        "min": "min", "count": "count", "std": "std"}
            result = getattr(df_filtered[col], func_map.get(func, "mean"))()
            data   = {f"{func}_{col}": round(float(result), 4)}
            answer = f"The **{func}** of **{col}** is **{result:,.4f}**"
            chart_type = None

        return QueryResult(success=True, data=data, answer=answer, query=q, chart_type=chart_type)

    def _execute_filter(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        filtered = self._apply_conditions(df, q.conditions)
        cols     = [c for c in q.columns if c in df.columns] or df.columns.tolist()
        result   = filtered[cols].head(q.limit)
        n        = len(filtered)
        answer   = f"Found **{n:,}** rows matching your filter (showing {min(q.limit, n)}):"
        return QueryResult(success=True, data=result, answer=answer, query=q)

    def _execute_sort(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        sort_col = q.order_by
        if not sort_col:
            num_cols = df.select_dtypes(include="number").columns
            sort_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]

        if sort_col not in df.columns:
            return QueryResult(success=False, data=None,
                               answer=f"Column '{sort_col}' not found.", query=q)

        sorted_df = df.sort_values(sort_col, ascending=q.ascending).head(q.limit)
        direction = "lowest" if q.ascending else "highest"
        answer    = f"**Top {q.limit} {direction}** by **{sort_col}**:"
        return QueryResult(success=True, data=sorted_df, answer=answer, query=q, chart_type="bar")

    def _execute_compare(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        if len(q.columns) >= 2:
            col1, col2 = q.columns[0], q.columns[1]
            if col1 in df.columns and col2 in df.columns:
                corr = df[[col1, col2]].corr().iloc[0, 1]
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                direction = "positive" if corr > 0 else "negative"
                answer = (f"Comparison: **{col1}** vs **{col2}**\n"
                          f"• Correlation: **{corr:.4f}** ({strength} {direction})\n"
                          f"• {col1} mean: {df[col1].mean():.2f} | {col2} mean: {df[col2].mean():.2f}")
                return QueryResult(success=True, data={"correlation": corr},
                                   answer=answer, query=q, chart_type="line")

        return self._execute_describe(q, df)


# ══════════════════════════════════════════════════════════════════════════════
# DataChatBot — Main Interface
# ══════════════════════════════════════════════════════════════════════════════

class DataChatBot:
    """
    Main DataChat interface.

    Combines QueryParser + DataQueryEngine to process
    natural language questions about any dataset.

    Optionally uses an LLM for complex queries beyond rule-based parsing.
    """

    GREETINGS = {"hello", "hi", "hey", "مرحبا", "سلام", "أهلاً", "أهلا"}
    HELP_WORDS = {"help", "مساعدة", "what can you do", "ماذا تفعل"}

    def __init__(self):
        self.parser  = QueryParser()
        self.engine  = DataQueryEngine()
        self.history: list[ChatMessage] = []
        self._df: pd.DataFrame | None   = None
        self._col_names: list[str]       = []

    def load_data(self, data: dict) -> None:
        """Load dataset into the chatbot."""
        self._df = data["df"].copy() if "df" in data else pd.DataFrame(data.get("rows", []))
        self._col_names = list(self._df.columns)
        self.history = []

    def chat(self, user_message: str) -> QueryResult:
        """
        Process a user message and return a result.

        Args:
            user_message: Natural language question about the data.

        Returns:
            QueryResult with answer and optional data.
        """
        if self._df is None:
            return QueryResult(
                success = False,
                data    = None,
                answer  = "Please load a dataset first.",
                query   = ParsedQuery("describe", [], [], "mean", None, None, False, 10, user_message),
            )

        msg_lower = user_message.lower().strip()

        # Handle greetings
        if any(g in msg_lower for g in self.GREETINGS):
            result = QueryResult(
                success = True,
                data    = {"columns": self._col_names, "rows": len(self._df)},
                answer  = (f"Hello! 👋 I'm ready to answer questions about your dataset.\n\n"
                           f"It has **{len(self._df):,}** rows and **{len(self._col_names)}** columns: "
                           f"{', '.join(self._col_names[:6])}{'...' if len(self._col_names) > 6 else ''}\n\n"
                           f"Ask me anything! For example:\n"
                           f"• *What is the average salary?*\n"
                           f"• *Show top 5 by revenue*\n"
                           f"• *Which region has the most orders?*"),
                query   = ParsedQuery("describe", [], [], "mean", None, None, False, 10, user_message),
            )
            self._add_to_history(user_message, result)
            return result

        # Handle help
        if any(h in msg_lower for h in self.HELP_WORDS):
            result = QueryResult(
                success = True,
                data    = {},
                answer  = (f"I can answer questions like:\n\n"
                           f"**Aggregations:**\n"
                           f"• What is the average {self._col_names[0]}?\n"
                           f"• What is the total revenue?\n\n"
                           f"**Filters:**\n"
                           f"• Show rows where quantity > 10\n"
                           f"• Filter by region = North\n\n"
                           f"**Sorting:**\n"
                           f"• Top 10 by sales\n"
                           f"• Sort by price ascending\n\n"
                           f"**Descriptions:**\n"
                           f"• Describe the salary column\n"
                           f"• How many missing values?\n"
                           f"• What are the unique categories?"),
                query = ParsedQuery("describe", [], [], "mean", None, None, False, 10, user_message),
            )
            self._add_to_history(user_message, result)
            return result

        # Parse and execute
        parsed = self.parser.parse(user_message, self._df)
        result = self.engine.execute(parsed, self._df)
        self._add_to_history(user_message, result)
        return result

    def chat_with_llm(
        self,
        user_message: str,
        api_key:  str,
        base_url: str = "https://api.groq.com/openai/v1",
        model:    str = "llama-3.3-70b-versatile",
    ) -> QueryResult:
        """
        Process query using LLM for complex questions.
        Falls back to rule-based if LLM fails.
        """
        if self._df is None:
            return self.chat(user_message)

        # Build context for LLM
        sample    = self._df.head(3).to_dict(orient="records")
        col_info  = {col: str(self._df[col].dtype) for col in self._col_names}
        stats     = {col: {
            "mean": round(float(self._df[col].mean()), 2) if pd.api.types.is_numeric_dtype(self._df[col]) else None,
            "unique": int(self._df[col].nunique()),
        } for col in self._col_names}

        system_prompt = f"""You are a data analyst assistant. Answer questions about this dataset.

Dataset info:
- Columns: {json.dumps(col_info)}
- Sample rows: {json.dumps(sample, default=str)}
- Statistics: {json.dumps(stats, default=str)}
- Total rows: {len(self._df)}

Rules:
1. Answer directly and concisely
2. Use actual numbers from the data
3. Format numbers with commas for thousands
4. Use bullet points for lists
5. If you need to compute something, use the statistics provided
6. Answer in the same language as the question"""

        try:
            import requests as req
            response = req.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model":       model,
                    "max_tokens":  500,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": user_message},
                    ],
                },
                timeout=20,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()

            result = QueryResult(
                success = True,
                data    = {},
                answer  = answer,
                query   = ParsedQuery("describe", self._col_names, [], "mean",
                                      None, None, False, 10, user_message),
            )
            self._add_to_history(user_message, result)
            return result

        except Exception:
            # Fall back to rule-based
            return self.chat(user_message)

    def get_suggested_questions(self) -> list[str]:
        """Generate suggested questions based on the loaded dataset."""
        if self._df is None:
            return []

        suggestions = []
        num_cols = [c for c in self._col_names if pd.api.types.is_numeric_dtype(self._df[c])]
        cat_cols = [c for c in self._col_names if self._df[c].dtype == object]

        if num_cols:
            suggestions.append(f"What is the average {num_cols[0]}?")
            suggestions.append(f"What is the maximum {num_cols[0]}?")
            if len(num_cols) >= 2:
                suggestions.append(f"Compare {num_cols[0]} and {num_cols[1]}")

        if cat_cols:
            suggestions.append(f"What are the unique values in {cat_cols[0]}?")
            if num_cols:
                suggestions.append(f"What is the total {num_cols[0]} by {cat_cols[0]}?")
                suggestions.append(f"Which {cat_cols[0]} has the highest {num_cols[0]}?")

        suggestions.append("How many missing values are there?")
        suggestions.append(f"Show top 5 rows by {num_cols[0] if num_cols else self._col_names[0]}")
        suggestions.append("Describe the dataset")

        return suggestions[:8]

    def clear_history(self) -> None:
        self.history = []

    def _add_to_history(self, user_msg: str, result: QueryResult) -> None:
        self.history.append(ChatMessage(role="user", content=user_msg))
        self.history.append(ChatMessage(role="assistant", content=result.answer, result=result))
        # Keep last 20 messages
        if len(self.history) > 40:
            self.history = self.history[-40:]


# ── Singleton factory ─────────────────────────────────────────────────────────
_chatbots: dict[str, DataChatBot] = {}


def get_chatbot(session_id: str = "default") -> DataChatBot:
    """Get or create a DataChatBot for a session."""
    if session_id not in _chatbots:
        _chatbots[session_id] = DataChatBot()
    return _chatbots[session_id]
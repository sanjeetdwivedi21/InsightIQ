# InsightIQ – AI Business Analyst Assistant

Ask business questions in plain English and get back SQL, analysis, charts, insights and reports.

InsightIQ is a portfolio project that mimics what a business analyst does every day: write SQL, analyse the results, build charts and explain what the numbers mean. Non-technical users should be able to ask *"Why did revenue decrease in March?"* and get an answer that is backed by real data.

> **Project status: in development (Phase 4 of 9 complete).**
> The data layer, the natural-language-to-SQL engine and the analytics engine are finished and tested. The visualization, insight and dashboard layers are planned and are being built phase by phase. Sections below clearly mark what is **done** and what is **planned**.

---

## Status

| Phase | What | Status |
|---|---|---|
| 1 | System architecture and project setup | Done |
| 2 | Synthetic dataset and SQLite database | Done (18/18 data-quality checks, 10/10 tests pass) |
| 3 | Natural language to SQL engine and SQL validator | Done (30/30 tests pass; 19-question live benchmark below) |
| 4 | Python business analytics engine (incl. revenue-drop investigation) | Done (25 tests pass) |
| 5 | Automated Plotly visualization engine | Planned |
| 6 | AI business insight generator | Planned |
| 7 | Streamlit dashboard (5 pages) | Planned |
| 8 | PDF and CSV report generation | Planned |
| 9 | Testing, security review, optimization, deployment | Planned |

---

## Target workflow

```
Question (Streamlit)
   -> Orchestrator
      -> Schema retrieval
      -> LLM generates SQL
      -> SQL validation (SELECT only)
      -> Read-only execution
      -> Pandas/NumPy analysis
      -> Plotly charts
      -> LLM writes insights from the computed numbers
      -> Report (PDF / CSV)
```

**Design rule:** the LLM never produces numbers. It writes SQL and wording only. Every figure comes from SQL or Pandas, and the insight step receives only those computed results. This is how the project avoids invented findings.

---

## Tech stack

| Area | Tools |
|---|---|
| Language and data | Python, SQL, Pandas, NumPy |
| Database | SQLite (schema is PostgreSQL-compatible) via SQLAlchemy |
| AI | Configurable LLM provider: NVIDIA NIM (tested live); Anthropic and OpenAI adapters are included but not yet tested live |
| Frontend (planned) | Streamlit |
| Visualization (planned) | Plotly |
| Reports (planned) | ReportLab |
| Config and validation | Pydantic, python-dotenv |
| Testing | pytest |

---

## What is built so far

### 1. Synthetic retail dataset (amounts in INR)

Generated with a fixed random seed (`42`), so the data is reproducible.

| Table | Rows |
|---|---|
| customers | 2,200 |
| products | 120 (6 categories) |
| orders | 11,643 (Jan 2024 to Dec 2025) |
| order_items | 22,564 |
| payments | 11,643 |

- 3 customer segments, 5 regions, 6 payment methods.
- Orders are about 87.5% Completed, 7.2% Cancelled and 5.3% Returned.
- The raw customer file deliberately contains 15 duplicate rows and 25 missing cities. The setup script removes the duplicates and fills missing cities with `Unknown`.

### 2. Business scenarios planted in the data

So that analytical questions have a real answer, these patterns are built into the data (not into any AI response):

- **Electronics declines** steadily through 2025.
- **Eight high-value customers** sharply cut their purchases from February to March 2025.
- **Seasonal peaks** in October and November, with higher discounts in those months.
- **Regional differences:** North is the largest region and Central the smallest.
- **Average order value varies** by month (discounts, basket size, category mix).

Verified from the database: between February and March 2025, revenue fell from about ₹65.5 lakh to ₹59.8 lakh, Electronics revenue fell from about ₹29.2 lakh to ₹22.3 lakh, and average order value fell from about ₹16,066 to ₹14,868.

### 3. Business definitions (one source of truth)

The view `vw_sales` defines sales once, and everything else should query it:

- **Revenue** = `quantity * unit_price * (1 - discount_pct)`, for **Completed** orders only.
- **Profit** = revenue minus `quantity * unit_cost`.
- Cancelled and Returned orders are excluded from revenue and tracked separately.
- **Average Order Value** = revenue / number of Completed orders.

### 4. Safe database access

`database/connection.py` opens the business database in SQLite **read-only mode**. Even if generated SQL were wrong, `INSERT`, `UPDATE`, `DELETE` and `DROP` are rejected by the database itself. A test checks this.

### 5. Data-quality checks and tests

`database/validate_data.py` runs 18 checks (row counts, foreign keys, date range, statuses, cost below price, payment amount equals order total, no orders before signup, no duplicate emails, and more). `tests/test_database.py` adds 10 tests covering quality checks, read-only enforcement, reproducibility, cleaning, the revenue formula and the planted scenarios.

### 6. Natural-language-to-SQL engine

```
Question -> LLM -> SQL -> validator -> read-only executor -> result
                              ^              |
                              +-- one repair attempt on a fixable SQL error
```

- **SQL validator:** allows exactly one `SELECT` / `WITH ... SELECT`; blocks write, schema and admin keywords, SQLite internals, unknown tables, schema-qualified names and dangerous functions. Comments are stripped first, and keywords inside string literals are not mistaken for commands.
- **Executor:** read-only connection, query timeout (10 s) and row limit (1,000). It re-validates every query itself.
- **Schema-aware prompts:** the live schema, the exact category/region/status values and the business definitions are given to the model. The model's reply is parsed for SQL only; its reasoning text is ignored.
- **Repair loop:** a fixable error is sent back to the model once. Forbidden operations are rejected immediately and never sent back for "fixing".
- **Follow-up support:** prior question and SQL pairs can be passed as conversation history (the UI part comes in Phase 7).

**Benchmark (author-written, 2026-11-26):** 19 questions, each with a hand-written reference query; the model's result must contain the reference result (labels and numbers, ignoring column names, order and extra columns).

| Model | Data questions correct | First attempt | Safety questions handled |
|---|---|---|---|
| `nvidia/nemotron-3.5-super-vl-120b-a12b` | 17 / 17 | 17 / 17 | 2 / 2 |

The two safety questions are an unrelated question (weather) and a destructive request ("delete all orders"). **Limits of this benchmark:** it is small, the questions are fairly direct, and the author wrote both the questions and the reference answers, so treat it as a regression check and not as proof of general accuracy. Harder, ambiguous and follow-up questions will be added.

Additional checks: the validator was fuzzed with 60,000 random SQL strings and the accepted ones were executed against the read-only database without any change to the data. This shows no bug was found, not that none exists.

### 7. Business analytics engine

Pandas/NumPy functions that take DataFrames and return structured results (no UI, no LLM), so every number is reproducible and testable.

- **Revenue and profit:** monthly trend with month-over-month and year-over-year change, growth/decline summary, volatility.
- **Categories, products, regions, segments:** performance tables, shares, and "what changed between two months" tables whose changes add up exactly to the net change.
- **Customers:** top customers, revenue concentration, repeat rate, time between orders, new/lost/reduced/increased buckets, and a high-value-customer check.
- **Order volume vs average order value:** an exact arithmetic split of a revenue change.

**Automated revenue-change investigation** (`python -m src.analytics.revenue_drop --month 2025-03`) answers questions like *"Why did revenue decrease in March?"*. The output separates **facts** (computed from the data), **hypotheses** (possible reasons, marked as not confirmed) and **limitations**. For March 2025 versus February 2025 it finds: revenue down 8.8%; Electronics down 23.7% (about 91% of all category decreases) after four consecutive monthly declines; average order value down 7.5%; and high-value customers spending well below their usual level. It also reports that the overall change is within normal month-to-month variation, and it makes no causal claims.

Two safeguards against misleading conclusions: comparisons for high-value customers are calibrated against earlier months (top customers buy in lumps, so a naive comparison would overstate a drop), and the report states when a change is within normal variation.

Testing: 25 tests use hand-calculable toy data for exact formulas and the real database for reconciliation with plain SQL. The tests were also checked by deliberately injecting bugs into the code to confirm they fail.

---

## Getting started

### Requirements

- Python 3.10 or newer
- Git

### Setup

```bash
git clone https://github.com/sanjeetdwivedi21/InsightIQ.git
cd InsightIQ
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate
```

Install dependencies and create your environment file:

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set your provider, model and key (keep the key only in `.env`, which is git-ignored):

```text
LLM_PROVIDER=nvidia
LLM_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
NVIDIA_API_KEY=your-key-here
```

Check the connection:

```bash
python -m src.ai.llm_client
```

Model IDs change over time. If you get a 404 or 410 error, pick a current model ID from the provider's catalog.

### Build the database

Run from the project root:

```bash
python -m database.generate_data      # writes CSV files to data/raw/
python -m database.setup_database     # cleans data, builds SQLite DB, runs 18 checks
python -m database.validate_data      # quality checks plus scenario report
```

Expected: `18/18 checks passed`.

### Run the tests

```bash
python -m pytest tests/test_database.py -v
```

Expected: `10 passed`. Use `python -m pytest`, not plain `pytest`, so the project modules are found.

```bash
python -m pytest tests/test_sql.py -v     # offline: validator, executor, engine (30 tests)
python -m tests.live_eval                 # live benchmark against your configured model
python -m tests.live_eval --model <id>    # compare another model
python -m pytest tests/test_analytics.py -v                  # analytics engine (25 tests)
python -m src.analytics.revenue_drop --month 2025-03         # explain a month's revenue change
```

---

## Project structure

```
InsightIQ/
├── app.py                      # (planned) Streamlit entry point
├── requirements.txt
├── .env.example
├── config/settings.py          # typed settings loaded from .env
├── data/
│   ├── raw/                    # generated CSVs
│   └── processed/              # SQLite database (git-ignored)
├── database/
│   ├── schema.sql              # tables, indexes, vw_sales view
│   ├── generate_data.py        # synthetic data generator
│   ├── setup_database.py       # clean, load, validate
│   ├── validate_data.py        # data-quality checks and scenario report
│   └── connection.py           # read-only and write connections
├── src/
│   ├── models.py               # result containers
│   ├── ai/                     # llm_client, prompts, sql_generator
│   ├── sql/                    # sql_validator, query_executor, schema_provider
│   ├── services/nl2sql.py      # question -> SQL -> result pipeline
│   ├── analytics/              # revenue, product, customer analysis; revenue_drop investigation
│   └── visualization/ reports/ # (planned)
├── pages/                      # (planned) Streamlit pages
└── tests/
    ├── test_database.py
    ├── test_sql.py
    ├── test_analytics.py
    └── live_eval.py            # benchmark with reference queries
```

Folders under `src/` and `pages/` exist as placeholders and are filled in as later phases are completed.

---

## LLM configuration

The provider is chosen in `.env`, so switching models does not require code changes elsewhere. All LLM calls go through one small client (`src/ai/llm_client.py`) with retries on rate limits and clear error messages.

```text
LLM_PROVIDER=nvidia        # nvidia | anthropic | openai
LLM_MODEL=<model id>
```

NVIDIA NIM uses an OpenAI-compatible endpoint. The Anthropic and OpenAI paths are implemented but have only been tested with a simulated client so far.

## Security design

Implemented and tested:

- Read-only database connection for all queries.
- SQL validator: a single `SELECT` only; write, DDL and admin statements rejected.
- Query timeout and row limit.
- The LLM only produces SQL text; nothing it returns is executed as code.
- Secrets live in `.env`, which is git-ignored; keys are held as secret values and are not printed.

Planned for Phase 9:

- Full security review, dependency audit and logging review.

Nothing in the planned list should be treated as verified until it has been implemented and tested.

---

## Roadmap

After the nine core phases, planned extensions include context-aware follow-up questions, anomaly detection, trend detection, customer segmentation, what-if analysis, SQL query explanations and exportable executive reports.

---

## Author

**Sanjeet Dwivedi** – BCA student and aspiring data analyst

- GitHub: [github.com/sanjeetdwivedi21](https://github.com/sanjeetdwivedi21)
- LinkedIn: [linkedin.com/in/sanjeetdwivedi-854539434](https://www.linkedin.com/in/sanjeetdwivedi-854539434)
- Email: sanjeetdwivedi2005@gmail.com

The dataset is entirely synthetic and generated for demonstration. It contains no real customer or business data.
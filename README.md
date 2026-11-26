# InsightIQ – AI Business Analyst Assistant

Ask business questions in plain English and get back SQL, analysis, charts, insights and reports.

InsightIQ is a portfolio project that mimics what a business analyst does every day: write SQL, analyse the results, build charts and explain what the numbers mean. Non-technical users should be able to ask *"Why did revenue decrease in March?"* and get an answer that is backed by real data.

> **Project status: in development (Phase 2 of 9 complete).**
> The data layer is finished and tested. The AI, analytics, visualization and dashboard layers are planned and are being built phase by phase. Sections below clearly mark what is **done** and what is **planned**.

---

## Status

| Phase | What | Status |
|---|---|---|
| 1 | System architecture and project setup | Done |
| 2 | Synthetic dataset and SQLite database | Done (18/18 data-quality checks, 10/10 tests pass) |
| 3 | Natural language to SQL engine and SQL validator | Planned |
| 4 | Python business analytics engine (incl. revenue-drop investigation) | Planned |
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
| AI (planned) | Configurable LLM provider (Anthropic Claude by default; OpenAI and NVIDIA NIM planned), LangChain |
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

An API key is not needed yet. It will be required from Phase 3, when the LLM is integrated. Keep it only in `.env`, which is git-ignored.

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
├── src/                        # (planned) ai, analytics, sql, visualization, reports, services
├── pages/                      # (planned) Streamlit pages
└── tests/
    └── test_database.py
```

Folders under `src/` and `pages/` exist as placeholders and are filled in as later phases are completed.

---

## LLM configuration (planned for Phase 3)

The provider is chosen in `.env`, so switching models should not require code changes elsewhere.

```text
LLM_PROVIDER=anthropic     # anthropic | openai | nvidia (nvidia to be added)
LLM_MODEL=<model id>
```

NVIDIA NIM exposes an OpenAI-compatible endpoint, so it will be added as another provider option.

## Security design

Implemented now:

- Read-only database connection for all queries.
- Secrets live in `.env`, which is git-ignored.

Planned for Phase 3 and 9:

- SQL validator: a single `SELECT` only, with write and DDL statements rejected.
- Query timeout and row limits.
- No execution of LLM-generated code.
- API keys never written to logs.

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

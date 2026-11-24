# make_structure.py  (ek baar chalao, phir delete kar do)
from pathlib import Path

folders = [
    ".streamlit", "config", "data/raw", "data/processed", "database",
    "src/services", "src/ai", "src/analytics", "src/sql",
    "src/visualization", "src/reports", "src/utils", "pages", "tests", "logs",
]
files = [
    "app.py", "README.md", "src/models.py",
    "database/connection.py", "database/generate_data.py", "database/setup_database.py",
    "src/services/orchestrator.py",
    "src/ai/llm_client.py", "src/ai/prompts.py", "src/ai/sql_generator.py",
    "src/ai/insight_generator.py", "src/ai/conversation.py",
    "src/analytics/revenue_analysis.py", "src/analytics/customer_analysis.py",
    "src/analytics/product_analysis.py", "src/analytics/business_metrics.py",
    "src/analytics/revenue_drop.py",
    "src/sql/sql_validator.py", "src/sql/query_executor.py",
    "src/visualization/chart_generator.py",
    "src/reports/report_generator.py",
    "src/utils/logger.py", "src/utils/helpers.py",
    "pages/1_Dashboard.py", "pages/2_AI_Analyst.py", "pages/3_Data_Explorer.py",
    "pages/4_Query_History.py", "pages/5_Reports.py",
    "tests/test_database.py", "tests/test_sql.py",
    "tests/test_analytics.py", "tests/test_insights.py",
]
# Python ko in folders ko "package" samajhne ke liye
init_dirs = ["config", "src", "src/services", "src/ai", "src/analytics",
             "src/sql", "src/visualization", "src/reports", "src/utils", "database"]

for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).touch(exist_ok=True)
for d in init_dirs:
    (Path(d) / "__init__.py").touch(exist_ok=True)

print("Structure ready!")
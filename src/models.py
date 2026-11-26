"""
src/models.py

Plain data containers passed between modules (standard-library dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SQLResult:
    """What the executor returns for one successful query."""
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    elapsed_ms: float
    truncated: bool = False          # True if the row limit cut the result short


@dataclass
class QueryOutcome:
    """Everything the UI needs to know about one question."""
    question: str
    # success | empty | cannot_answer | rejected | invalid_question | error
    status: str
    sql: str = ""
    dataframe: pd.DataFrame | None = None
    row_count: int = 0
    elapsed_ms: float = 0.0
    truncated: bool = False
    message: str = ""                # short, user-friendly explanation
    error: str = ""                  # technical detail when something failed
    attempts: int = 0                # SQL attempts made (1 = worked first time)
    llm_calls: int = 0
    sql_history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("success", "empty")
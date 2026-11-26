"""
src/ai/sql_generator.py

Asks the LLM to turn a question into SQL, and extracts the SQL from its reply.
It never executes anything: validation and execution are separate modules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.ai.prompts import REPAIR_PROMPT, SQL_SYSTEM_PROMPT
from src.sql.schema_provider import get_schema_text


@dataclass
class Generation:
    sql: str | None = None
    cannot_answer_reason: str | None = None
    raw: str = ""


_FENCE_RE = re.compile(r"```(?:sqlite|sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# A bare (unfenced) reply that starts like SQL is passed on, even if it is a write statement,
# so the validator can reject it properly instead of it being mistaken for "no SQL found".
_START_RE = re.compile(r"^\s*(?:select|with|insert|update|delete|drop|alter|create|replace|pragma|attach|vacuum)\b",
                       re.IGNORECASE)
_CANNOT_RE = re.compile(r"^\s*CANNOT_ANSWER\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> Generation:
    """Pull the SQL (or a CANNOT_ANSWER reason) out of the model's reply."""
    raw = text or ""
    stripped = raw.strip()
    m = _CANNOT_RE.match(stripped)
    if m:
        return Generation(cannot_answer_reason=m.group(1).strip() or "The question cannot be answered.", raw=raw)
    fence = _FENCE_RE.search(stripped)
    candidate = fence.group(1).strip() if fence else (stripped if _START_RE.match(stripped) else "")
    candidate = candidate.rstrip().rstrip(";").strip()
    return Generation(sql=candidate or None, raw=raw)


class SQLGenerator:
    def __init__(self, llm=None, schema_text: str | None = None) -> None:
        self._llm = llm
        self._schema_text = schema_text

    @property
    def llm(self):
        if self._llm is None:
            from src.ai.llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    def _system(self) -> str:
        schema = self._schema_text or get_schema_text()
        return SQL_SYSTEM_PROMPT.format(schema=schema)

    @staticmethod
    def _history_messages(history: list[tuple[str, str]] | None) -> list[dict]:
        msgs: list[dict] = []
        for q, sql in (history or [])[-3:]:                 # last 3 turns is plenty for follow-ups
            msgs.append({"role": "user", "content": f"Question: {q}"})
            msgs.append({"role": "assistant", "content": f"```sql\n{sql}\n```"})
        return msgs

    def generate(self, question: str, history: list[tuple[str, str]] | None = None) -> Generation:
        messages = self._history_messages(history) + [{"role": "user", "content": f"Question: {question}"}]
        return extract_sql(self.llm.generate(self._system(), messages, temperature=0.0))

    def repair(self, question: str, bad_sql: str, error: str,
               history: list[tuple[str, str]] | None = None) -> Generation:
        messages = self._history_messages(history) + [
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": f"```sql\n{bad_sql}\n```"},
            {"role": "user", "content": REPAIR_PROMPT.format(error=error)},
        ]
        return extract_sql(self.llm.generate(self._system(), messages, temperature=0.0))
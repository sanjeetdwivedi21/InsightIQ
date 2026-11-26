"""
src/services/nl2sql.py

The natural-language-to-SQL pipeline:

  question -> LLM -> SQL -> validate -> execute (read-only) -> QueryOutcome

If the SQL is repairable-wrong (unknown table, SQL error), the LLM gets the error
message and ONE chance to fix it. Forbidden operations (DELETE, DROP, ...) are
rejected immediately and never sent back for "fixing".
"""
from __future__ import annotations

import logging
import re

from src.ai.sql_generator import SQLGenerator
from src.models import QueryOutcome
from src.sql.query_executor import QueryExecutionError, QueryTimeoutError, execute_query
from src.sql.sql_validator import SQLValidationError, validate_sql

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 500
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_question(question: str) -> str:
    return _CONTROL_CHARS.sub("", question or "").strip()


class NL2SQLEngine:
    def __init__(self, generator: SQLGenerator | None = None, max_repairs: int = 1) -> None:
        self.generator = generator or SQLGenerator()
        self.max_repairs = max_repairs

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> QueryOutcome:
        question = clean_question(question)
        out = QueryOutcome(question=question, status="error")

        if not question:
            out.status, out.message = "invalid_question", "Please type a question."
            return out
        if len(question) > MAX_QUESTION_LENGTH:
            out.status = "invalid_question"
            out.message = f"Please keep your question under {MAX_QUESTION_LENGTH} characters."
            return out

        from src.ai.llm_client import LLMError   # local import keeps this module light for tests

        try:
            gen = self.generator.generate(question, history)
            out.llm_calls += 1
            for attempt in range(1, self.max_repairs + 2):
                if gen.cannot_answer_reason:
                    out.status = "cannot_answer"
                    out.message = f"This question can't be answered from the business database: {gen.cannot_answer_reason}"
                    return self._done(out)
                if not gen.sql:
                    error, repairable, kind = "No SQL query was found in the model's reply.", True, "error"
                else:
                    out.sql = gen.sql
                    out.sql_history.append(gen.sql)
                    out.attempts = attempt
                    error, repairable, kind = self._try_run(gen.sql, out)
                    if error is None:
                        return self._done(out)
                if not repairable or attempt > self.max_repairs:
                    out.error = error
                    out.status = "rejected" if kind == "validation" else "error"
                    if kind == "validation":
                        out.message = "That query was blocked for safety or validity: " + error
                    elif kind == "timeout":
                        out.message = error + " Try a narrower question."
                    else:
                        out.message = "I couldn't produce a working query for that question. Try rephrasing it."
                    return self._done(out)
                gen = self.generator.repair(question, gen.sql or "", error, history)
                out.llm_calls += 1
        except LLMError as exc:
            out.status, out.error = "error", str(exc)
            out.message = "The AI service is unavailable right now. Please try again in a moment."
        return self._done(out)

    @staticmethod
    def _try_run(sql: str, out: QueryOutcome) -> tuple[str | None, bool, str]:
        """Validate and execute. Returns (error_text, repairable, kind); error_text None = success."""
        try:
            validate_sql(sql)
            result = execute_query(sql)
        except SQLValidationError as exc:
            return str(exc), exc.repairable, "validation"
        except QueryTimeoutError as exc:
            return str(exc), False, "timeout"
        except QueryExecutionError as exc:
            return f"Database error: {exc}", True, "database"
        out.sql = result.sql
        out.dataframe = result.dataframe
        out.row_count = result.row_count
        out.elapsed_ms = result.elapsed_ms
        out.truncated = result.truncated
        out.status = "empty" if result.row_count == 0 else "success"
        out.message = "The query ran but returned no rows." if result.row_count == 0 else ""
        return None, False, ""

    @staticmethod
    def _done(out: QueryOutcome) -> QueryOutcome:
        logger.info("question=%r status=%s attempts=%d llm_calls=%d", out.question[:80], out.status,
                    out.attempts, out.llm_calls)
        return out
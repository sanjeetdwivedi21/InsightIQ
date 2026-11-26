"""
src/sql/sql_validator.py

Decides whether a piece of (LLM-generated) SQL is safe to run. Rules:

  1. Exactly ONE statement, and it must start with SELECT or WITH.
  2. No write / schema / admin keywords anywhere (INSERT, UPDATE, DELETE, DROP,
     ALTER, CREATE, PRAGMA, ATTACH, ...), checked on real tokens only, so a
     keyword inside a string literal like 'DROP' is fine.
  3. Only the business tables and the vw_sales view may be queried
     (CTE names defined in the query itself are also allowed).
  4. No SQLite internals (sqlite_*, pragma_*), no schema-qualified names
     (main.orders), no table-valued functions, no dangerous functions.
  5. Comments are stripped before checking, so they cannot hide anything.

This is ONE layer of defence. The database connection is also read-only and the
executor enforces a timeout and a row limit.
"""
from __future__ import annotations

import re

ALLOWED_TABLES = {"customers", "products", "orders", "order_items", "payments", "vw_sales"}

MAX_SQL_LENGTH = 5000

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "BEGIN", "COMMIT",
    "ROLLBACK", "SAVEPOINT", "RELEASE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "MERGE",
    "UPSERT", "EXPLAIN",
}
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "fts3_tokenizer",
                       "randomblob", "zeroblob"}

# Words that end a table reference (so they are not mistaken for an alias).
_CLAUSE_WORDS = {"WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "EXCEPT", "INTERSECT",
                 "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "OUTER", "ON",
                 "USING", "WINDOW", "SELECT"}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_NUM_RE = re.compile(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")


class SQLValidationError(Exception):
    """SQL was rejected. `repairable` = worth asking the LLM to fix it (False for forbidden operations)."""

    def __init__(self, message: str, repairable: bool = False) -> None:
        super().__init__(message)
        self.repairable = repairable


Token = tuple[str, str]  # (kind, text); kind: word | string | qident | number | punct


def tokenize(sql: str) -> tuple[list[Token], str]:
    """Split SQL into tokens, dropping comments. Returns (tokens, cleaned_sql_without_comments)."""
    tokens: list[Token] = []
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                raise SQLValidationError("Unterminated comment in SQL.", repairable=True)
            i = j + 2
            out.append(" ")
            continue
        if c in ("'", '"', "`"):
            j = i + 1
            while True:
                k = sql.find(c, j)
                if k == -1:
                    raise SQLValidationError("Unterminated quote in SQL.", repairable=True)
                if k + 1 < n and sql[k + 1] == c:     # doubled quote = escaped quote
                    j = k + 2
                    continue
                break
            tokens.append(("string" if c == "'" else "qident", sql[i + 1:k].replace(c + c, c)))
            out.append(sql[i:k + 1])
            i = k + 1
            continue
        if c == "[":
            k = sql.find("]", i + 1)
            if k == -1:
                raise SQLValidationError("Unterminated [identifier] in SQL.", repairable=True)
            tokens.append(("qident", sql[i + 1:k]))
            out.append(sql[i:k + 1])
            i = k + 1
            continue
        if c.isspace():
            out.append(c)
            i += 1
            continue
        m = _WORD_RE.match(sql, i)
        if m:
            tokens.append(("word", m.group()))
            out.append(m.group())
            i = m.end()
            continue
        m = _NUM_RE.match(sql, i)
        if m and m.group():
            tokens.append(("number", m.group()))
            out.append(m.group())
            i = m.end()
            continue
        tokens.append(("punct", c))
        out.append(c)
        i += 1
    return tokens, "".join(out)


def _is_word(tok: Token, *words: str) -> bool:
    return tok[0] == "word" and tok[1].upper() in words


def _is_punct(tok: Token, ch: str) -> bool:
    return tok[0] == "punct" and tok[1] == ch


def _cte_names(tokens: list[Token]) -> set[str]:
    """Names defined by  WITH name AS (...) , name2 AS (...)  patterns."""
    names: set[str] = set()
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok[0] not in ("word", "qident") or i == 0:
            continue
        prev = tokens[i - 1]
        if not (_is_word(prev, "WITH", "RECURSIVE") or _is_punct(prev, ",")):
            continue
        j = i + 1
        if j < n and _is_punct(tokens[j], "("):          # optional column list
            depth = 0
            while j < n:
                if _is_punct(tokens[j], "("):
                    depth += 1
                elif _is_punct(tokens[j], ")"):
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
        if j + 1 < n and _is_word(tokens[j], "AS"):
            k = j + 1
            while k < n and _is_word(tokens[k], "NOT", "MATERIALIZED"):
                k += 1
            if k < n and _is_punct(tokens[k], "("):
                names.add(tok[1].lower())
    return names


def _referenced_tables(tokens: list[Token]) -> set[str]:
    tables: set[str] = set()
    n = len(tokens)
    for idx, tok in enumerate(tokens):
        if not _is_word(tok, "FROM", "JOIN"):
            continue
        is_from = tok[1].upper() == "FROM"
        j = idx + 1
        while j < n:
            t = tokens[j]
            if _is_punct(t, "("):
                nxt = tokens[j + 1] if j + 1 < n else ("punct", "")
                if _is_word(nxt, "SELECT", "WITH") or _is_punct(nxt, "("):
                    break                                   # subquery: its own FROM is checked later
                raise SQLValidationError("Parenthesised table lists are not supported; use JOIN.",
                                         repairable=True)
            if t[0] not in ("word", "qident"):
                raise SQLValidationError("Could not understand a table reference in FROM/JOIN.",
                                         repairable=True)
            name = t[1]
            j += 1
            if j < n and _is_punct(tokens[j], "."):
                raise SQLValidationError("Schema-qualified table names (like main.orders) are not allowed.",
                                         repairable=True)
            if j < n and _is_punct(tokens[j], "("):
                raise SQLValidationError(f"Table-valued function '{name}' is not allowed.", repairable=True)
            tables.add(name.lower())
            if j < n and _is_word(tokens[j], "AS"):
                j += 2
            elif j < n and tokens[j][0] in ("word", "qident") and not (
                    tokens[j][0] == "word" and tokens[j][1].upper() in _CLAUSE_WORDS):
                j += 1
            if is_from and j < n and _is_punct(tokens[j], ","):
                j += 1
                continue
            break
    return tables


def validate_sql(sql: str) -> str:
    """Return the cleaned SQL (comments and trailing semicolon removed) or raise SQLValidationError."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("The SQL query is empty.", repairable=True)
    if len(sql) > MAX_SQL_LENGTH:
        raise SQLValidationError(f"The SQL query is too long (limit {MAX_SQL_LENGTH} characters).",
                                 repairable=True)

    tokens, cleaned = tokenize(sql)

    # drop trailing semicolons, then no semicolon may remain
    while tokens and _is_punct(tokens[-1], ";"):
        tokens.pop()
    cleaned = cleaned.strip().rstrip(";").strip()
    if not tokens:
        raise SQLValidationError("The SQL query is empty.", repairable=True)
    if any(_is_punct(t, ";") for t in tokens):
        raise SQLValidationError("Only a single SQL statement is allowed.", repairable=True)

    if not _is_word(tokens[0], "SELECT", "WITH"):
        raise SQLValidationError("Only SELECT queries are allowed.", repairable=False)

    for i, tok in enumerate(tokens):
        kind, text = tok
        if kind == "word":
            up = text.upper()
            if up in FORBIDDEN_KEYWORDS:
                nxt = tokens[i + 1] if i + 1 < len(tokens) else ("", "")
                if up == "REPLACE" and _is_punct(nxt, "("):
                    continue                                 # REPLACE(text, a, b) is a harmless function
                raise SQLValidationError(f"The keyword {up} is not allowed. Only read-only SELECT queries can run.",
                                         repairable=False)
            if text.lower() in FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(f"The function {text} is not allowed.", repairable=False)
        if kind in ("word", "qident"):
            low = text.lower()
            if low.startswith("sqlite_") or low.startswith("pragma_"):
                raise SQLValidationError("Access to SQLite internal objects is not allowed.", repairable=False)

    unknown = _referenced_tables(tokens) - _cte_names(tokens) - ALLOWED_TABLES
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_TABLES))
        raise SQLValidationError(f"Unknown or not allowed table(s): {', '.join(sorted(unknown))}. "
                                 f"Allowed: {allowed}.", repairable=True)
    return cleaned
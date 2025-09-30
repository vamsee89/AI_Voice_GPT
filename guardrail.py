import re
import sqlparse
from typing import Dict, Any, Tuple

# ----------------- Detection Helpers -----------------

def detect_sql_injection(sql: str) -> Tuple[bool, str]:
    """Detect SQL injection and dangerous patterns."""
    patterns = [
        r";", r"--", r"/\*", r"\bEXEC\b", r"\bUNION\b", r"\bDROP\b",
        r"\bINSERT\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bMERGE\b",
        r"xp_", r"pg_read_file"
    ]
    for p in patterns:
        if re.search(p, sql, re.IGNORECASE):
            return True, f"SQL Injection pattern: {p}"
    return False, ""


def detect_direct_pii(nl_question: str) -> bool:
    """Detect if the question references a specific member by name or SSN."""
    # Proper name detection
    if re.search(r"\b[A-Z][a-z]{2,}\b", nl_question):
        return True
    # SSN patterns
    if re.search(r"\d{3}-\d{2}-\d{4}", nl_question):
        return True
    return False


def detect_aggregation(nl_question: str) -> bool:
    """Detect if aggregation is being asked."""
    agg_terms = ["average", "avg", "count", "sum", "total", "min", "max", "distribution"]
    return any(term in nl_question.lower() for term in agg_terms)


def detect_group_scope(nl_question: str) -> bool:
    """Detect if the aggregation is group-based (safe)."""
    group_terms = ["plan", "region", "product", "department", "segment", "team"]
    return any(term in nl_question.lower() for term in group_terms)


# ----------------- Guardrail Engine -----------------

def guardrail_check(nl_question: str, sql: str) -> Dict[str, Any]:
    """Check natural language + SQL query against guardrail rules."""

    # 1. Block SQL injection
    inj, reason = detect_sql_injection(sql)
    if inj:
        return {"status": "blocked", "reason": reason}

    # 2. Block non-SELECT
    parsed = sqlparse.parse(sql)
    if parsed and parsed[0].get_type() != "SELECT":
        return {"status": "blocked", "reason": "Only SELECT queries allowed"}

    # 3. PII detection
    if detect_direct_pii(nl_question):
        if detect_aggregation(nl_question):
            # Aggregated but person-specific → block
            if not detect_group_scope(nl_question):
                return {"status": "blocked", "reason": "Aggregated query over a single member (PII)"}
            else:
                return {"status": "ok", "reason": "Group aggregation allowed"}
        else:
            return {"status": "blocked", "reason": "Direct PII/PHI lookup not allowed"}

    # 4. Aggregation without PII → allow
    if detect_aggregation(nl_question):
        return {"status": "ok", "reason": "Aggregate query allowed"}

    # 5. Default → allow
    return {"status": "ok", "reason": "Non-sensitive query allowed"}


# ----------------- Example Usage -----------------

examples = [
    ("Show John comment", "SELECT comment FROM member_profile WHERE first_name='John';"),
    ("What is the average CSAT for John", "SELECT AVG(csat) FROM survey WHERE first_name='John';"),
    ("What is the average CSAT for Gold plan members", "SELECT AVG(csat) FROM survey WHERE plan='Gold';"),
    ("Count of members by product", "SELECT product, COUNT(*) FROM survey GROUP BY product;")
]

for q, s in examples:
    print(q, "=>", guardrail_check(q, s))

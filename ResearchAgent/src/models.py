from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Model registry — change models in one place, not scattered across agents.
#
# Strategy: gpt-4o only where the user sees the output or reasoning depth
# directly affects quality.  Everything structural stays on gpt-4o-mini.
#
#   gpt-4o      — 30k  TPM tier-1 · $2.50 / $10  per 1M tokens
#   gpt-4o-mini — 200k TPM tier-1 · $0.15 / $0.60 per 1M tokens
#
# Usage map
# ─────────────────────────────────────────────────────────────────────────
# ROUTER  (mini, t=0)   supervisor routing, search queries, deep questions
# STRICT  (mini, t=0)   all structured-output tasks (plan, assemble, experts)
# ANALYST (4o,   t=0.7) answer_deep_questions — research depth feeds writing
# WRITER  (4o,   t=0.7) write_section         — the prose the user reads
# ─────────────────────────────────────────────────────────────────────────
# ANALYST and WRITER run in separate phases (research then writing), so the
# 30k TPM budget has time to reset between them.
# ---------------------------------------------------------------------------

ROUTER  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
STRICT  = ChatOpenAI(model="gpt-4o-mini", temperature=0,   max_tokens=4096)
ANALYST = ChatOpenAI(model="gpt-4o",      temperature=0.7)
WRITER  = ChatOpenAI(model="gpt-4o",      temperature=0.7)

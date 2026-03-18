from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Model registry
#
# All agents import from here so model changes happen in one place.
#
# gpt-4o-mini vs gpt-4o
#   gpt-4o      — 30k  TPM on tier 1, $2.50/$10 per 1M tokens
#   gpt-4o-mini — 200k TPM on tier 1, $0.15/$0.60 per 1M tokens
#
# Parallel section writes easily exceed 30k TPM, so gpt-4o-mini is the
# right default here.  Swap `WRITER` to gpt-4o if you upgrade your tier
# and want higher-quality prose.
# ---------------------------------------------------------------------------

# Routing / structured decisions (supervisor, expert routing)
ROUTER = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Creative / generative tasks (expert breakdown, research answers, writing)
CREATIVE = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Strict structured output (plan_report, write_section schema)
STRICT = ChatOpenAI(model="gpt-4o-mini", temperature=0)
